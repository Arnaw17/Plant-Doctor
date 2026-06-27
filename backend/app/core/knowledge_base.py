from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


SeedGuide = Dict[str, str]


SEED_GUIDES: List[SeedGuide] = [
    {
        "id": "water-root-airflow",
        "title": "Watering works better when roots can dry between cycles",
        "url": "https://old.reddit.com/r/houseplants/comments/1jrkugs/you_dont_suck_at_keeping_strings_of_pearls_alive/",
        "topic": "water",
        "summary": (
            "Community advice from a String of Pearls care post: plants can tolerate water, but they do not "
            "like wet roots for long. Better drainage, a faster-drying mix, and the right pot matter more than "
            "waiting for visible shriveling."
        ),
        "evidence": "SoPs actually love water, but they hate having wet roots for very long.",
    },
    {
        "id": "light-brighter-than-you-think",
        "title": "Most indoor plants want more light than the room seems to offer",
        "url": "https://old.reddit.com/r/houseplants/comments/1masf6r/my_mums_orbifolia_that_she_waters_with_tap_water_and_gives_two_hours_of/",
        "topic": "light",
        "summary": (
            "A recurring r/houseplants theme is that bright indirect light, window rotation, and supplemental grow "
            "lights can reverse stretching and improve leaf quality. Several showcase posts describe plants thriving "
            "after being moved closer to windows or under grow lights."
        ),
        "evidence": "It gets virtually zero usable light, and puts it in the window for a couple of hours.",
    },
    {
        "id": "pest-quarantine-early",
        "title": "Quarantine new plants and inspect for mites, thrips, and webbing early",
        "url": "https://old.reddit.com/r/houseplants/comments/1m6mepr/reminder_to_always_quarantine_any_new_plants/",
        "topic": "pest",
        "summary": (
            "The subreddit repeatedly recommends isolating new plants, checking stems and leaf undersides, and "
            "treating quickly when webbing, sticky residue, or tiny moving pests appear. Early quarantine often "
            "prevents a small problem from becoming a room-wide outbreak."
        ),
        "evidence": "Reminder to always quarantine any new plants.",
    },
    {
        "id": "repotting-drainage-mix",
        "title": "Repotting works when the soil and pot match the plant's airflow needs",
        "url": "https://old.reddit.com/r/houseplants/comments/1khyqtc/the_day_i_repotted_my_concerningly_phallic_cactus_vs_one_week_later/",
        "topic": "nutrient",
        "summary": (
            "Repotting stories in r/houseplants often point to the same fix: a better-draining mix, a pot that "
            "evaporates moisture at the right rate, and a more stable root environment. When the root zone dries "
            "too slowly, even tough plants struggle."
        ),
        "evidence": "Everything you've ever heard about watering String of Pearls is wrong.",
    },
    {
        "id": "humidity-fern-rhythm",
        "title": "Humidity-loving plants respond to steady routines",
        "url": "https://old.reddit.com/r/houseplants/comments/1jbzh5m/do_you_have_fern_envy_how_i_care_for_them_below/",
        "topic": "water",
        "summary": (
            "Fern and moisture-sensitive plant posts in the subreddit emphasize repeatable care rhythms: check the "
            "weight of the pot, water when the medium is ready, and keep ambient humidity stable rather than "
            "over-correcting after visible stress appears."
        ),
        "evidence": "Everyday I lift the bottom of each hanging basket and if it feels light, I water.",
    },
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _embed_text(text: str, dimensions: int = 48) -> List[float]:
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dimensions
        vector[bucket] += 1.0
    norm = sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _collection_text(seed: SeedGuide) -> str:
    return " ".join([seed["title"], seed["topic"], seed["summary"], seed["evidence"]])


def _seed_to_hit(seed: SeedGuide, note_suffix: str = "") -> Dict[str, str]:
    note = seed["summary"]
    if seed.get("evidence"):
        note = f'{note} Key phrase: {seed["evidence"]}'
    if note_suffix:
        note = f"{note} {note_suffix}".strip()
    return {
        "title": seed["title"],
        "url": seed["url"],
        "topic": seed["topic"],
        "note": note,
        "document": _collection_text(seed),
        "distance": "",
    }


def _fallback_search(query: str, issue_category: Optional[str], species: Optional[str], limit: int) -> List[Dict[str, str]]:
    query_tokens = set(_tokenize(" ".join(part for part in [query, issue_category or "", species or ""] if part)))
    scored: List[tuple[float, SeedGuide]] = []
    for seed in SEED_GUIDES:
        seed_tokens = set(_tokenize(_collection_text(seed)))
        overlap = len(query_tokens & seed_tokens)
        bonus = 0.0
        if issue_category and seed["topic"] == issue_category:
            bonus += 2.0
        if species and species.lower() in seed["document"].lower() if False else False:
            bonus += 0.0
        scored.append((overlap + bonus, seed))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_seed_to_hit(seed) for score, seed in scored[: max(1, limit)] if score >= 0]


@lru_cache(maxsize=1)
def get_care_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="plant_care_guides")
    if collection.count() == 0:
        collection.add(
            ids=[seed["id"] for seed in SEED_GUIDES],
            documents=[_collection_text(seed) for seed in SEED_GUIDES],
            metadatas=[
                {
                    "title": seed["title"],
                    "url": seed["url"],
                    "topic": seed["topic"],
                    "summary": seed["summary"],
                    "evidence": seed["evidence"],
                }
                for seed in SEED_GUIDES
            ],
            embeddings=[_embed_text(_collection_text(seed)) for seed in SEED_GUIDES],
        )
    return collection


def search_care_knowledge(
    query: str,
    issue_category: Optional[str] = None,
    species: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, str]]:
    search_terms = " ".join(term for term in [species or "", issue_category or "", query or ""] if term)
    try:
        results = get_care_collection().query(
            query_embeddings=[_embed_text(search_terms or query or issue_category or species or "plant care")],
            n_results=max(1, limit),
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits: List[Dict[str, str]] = []
        for index, metadata in enumerate(metadatas):
            summary = str(metadata.get("summary", ""))
            evidence = str(metadata.get("evidence", ""))
            note = summary
            if evidence:
                note = f"{summary} Key phrase: {evidence}"
            hits.append(
                {
                    "title": str(metadata.get("title", f"Guide {index + 1}")),
                    "url": str(metadata.get("url", "")),
                    "topic": str(metadata.get("topic", "general")),
                    "note": note,
                    "distance": str(distances[index]) if index < len(distances) else "",
                    "document": str(documents[index]) if index < len(documents) else "",
                }
            )
        return hits
    except BaseException:
        return _fallback_search(query=search_terms or query or "", issue_category=issue_category, species=species, limit=limit)


def format_citations(hits: List[Dict[str, str]], label: str = "Sources") -> str:
    if not hits:
        return ""
    lines = [f"{label}:"]
    for index, hit in enumerate(hits, start=1):
        title = hit.get("title", f"Source {index}")
        url = hit.get("url", "")
        note = hit.get("note", "").strip()
        if note:
            lines.append(f"- {title}: {note} ({url})")
        else:
            lines.append(f"- {title}: {url}")
    return "\n".join(lines)