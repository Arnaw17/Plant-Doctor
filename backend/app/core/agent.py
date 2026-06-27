from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.knowledge_base import format_citations, search_care_knowledge

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ISSUE_CATEGORIES = ("water", "light", "pest", "nutrient", "disease")

DEFAULT_FOLLOWUP_QUESTIONS: Dict[str, List[str]] = {
    "water": [
        "How often do you water, and does the pot fully drain?",
        "Are the leaves soft and yellow or crisp and brown?",
        "Has the soil stayed wet for more than 2-3 days?",
    ],
    "light": [
        "What direction does the window face and how many hours of bright light does it get?",
        "Has the plant stretched, leaned, or produced smaller leaves recently?",
        "Is it sitting close to glass or tucked far from the window?",
    ],
    "pest": [
        "Do you see sticky residue, webbing, bumps, or tiny moving insects?",
        "Have any nearby plants developed similar symptoms?",
        "Did you recently bring in a new plant or move this one outdoors?",
    ],
    "nutrient": [
        "Have you fertilized in the last month, and with what strength?",
        "Are the newest leaves pale, small, or distorted?",
        "Is the plant root-bound or due for fresh soil?",
    ],
    "disease": [
        "Are there spots, mold, mushy stems, or rapidly spreading damage?",
        "Have the affected leaves been wet for long periods?",
        "Has the issue spread from one leaf to many in the last week?",
    ],
}

RECOVERY_TIMES = {
    "water": "7-14 days for early stress, 3-4 weeks for fuller recovery",
    "light": "2-4 weeks for visible improvement, longer for new growth",
    "pest": "1-3 weeks to stop spread, 4-6 weeks for cleanup",
    "nutrient": "2-4 weeks after feeding and soil correction",
    "disease": "1-2 weeks to slow spread, 3-6 weeks for stabilization",
}

WATCH_FOR = {
    "water": "new leaf firmness, less drooping, and soil drying at a healthier pace",
    "light": "upright growth, richer color, and smaller gaps between leaves",
    "pest": "no new specks, webbing, or sticky residue, plus cleaner new growth",
    "nutrient": "greener new leaves, better size, and fewer deformed tips",
    "disease": "stopped spread, drier lesions, and healthier unaffected leaves",
}


class IntakeAnalysis(BaseModel):
    species: str = Field(default="Unknown")
    issue_category: str = Field(default="water")
    followup_questions: List[str] = Field(default_factory=list)
    summary: str = Field(default="")


class CarePlanDraft(BaseModel):
    care_plan: str
    expected_recovery_time: str
    one_week_watch_for: str
    summary: str
    checklist_items: List[Dict[str, object]] = Field(default_factory=list)


class CheckInReview(BaseModel):
    comparison_summary: str
    plan_update: str
    health_status: str


class AgentState(TypedDict, total=False):
    messages: Sequence[BaseMessage]
    species: str
    location: str
    issue_category: str
    care_plan: str
    diagnosis_ready: bool
    image_data: Optional[str]
    followup_answers: Dict[str, str]


def _trim(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _category_from_text(text: str) -> str:
    lowered = text.lower()
    weighted_keywords = {
        "water": ["dry", "droop", "drooping", "wilting", "yellow", "overwater", "underwater", "mushy"],
        "light": ["light", "shade", "sun", "window", "stretch", "leggy", "pale"],
        "pest": ["pest", "bug", "bugs", "web", "webbing", "mite", "mites", "sticky", "scale", "aphid"],
        "nutrient": ["nutrient", "fertil", "deficiency", "pale", "chlorosis"],
        "disease": ["disease", "spot", "spots", "mold", "fungus", "fungal", "blight", "rot", "rotting"],
    }
    scores = {category: 0 for category in ISSUE_CATEGORIES}
    for category, keywords in weighted_keywords.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[category] += 1

    if max(scores.values()) == 0:
        return "water"
    return max(scores, key=scores.get)


def _species_from_text(text: str) -> str:
    lowered = text.lower()
    common_species = [
        "pothos",
        "snake plant",
        "monstera",
        "peace lily",
        "spider plant",
        "ficus",
        "philodendron",
        "succulent",
        "cactus",
        "fern",
        "orchid",
        "tomato",
        "rose",
    ]
    for name in common_species:
        if name in lowered:
            return name.title()
    return "Unknown"


def _followup_questions(issue_category: str) -> List[str]:
    return DEFAULT_FOLLOWUP_QUESTIONS.get(issue_category, DEFAULT_FOLLOWUP_QUESTIONS["water"])[:3]


def _care_plan_text(species: str, issue_category: str, answers: Optional[Dict[str, str]], location: Optional[str]) -> CarePlanDraft:
    answers = answers or {}
    base_steps = {
        "water": [
            "Check the top 2-3 cm of soil before watering again.",
            "If the soil is soggy, pause watering and improve drainage immediately.",
            "If the soil is dry, water thoroughly until excess runs out the bottom.",
        ],
        "light": [
            "Move the plant to brighter indirect light or closer to the window if it has been stretching.",
            "Rotate the pot weekly so growth stays even.",
            "Avoid harsh direct afternoon sun unless the species tolerates it well.",
        ],
        "pest": [
            "Isolate the plant from others right away.",
            "Wipe visible pests off leaves and stems, then treat with insecticidal soap or neem as appropriate.",
            "Repeat treatment every 5-7 days until no pests remain.",
        ],
        "nutrient": [
            "Pause strong fertilizer and inspect whether the plant needs fresh potting mix.",
            "Feed at half strength only after the plant is stable and roots are not stressed.",
            "Flush the soil if fertilizer salt buildup is likely.",
        ],
        "disease": [
            "Remove the worst affected leaves with clean tools.",
            "Keep foliage dry and improve airflow around the plant.",
            "If the issue keeps spreading, treat with an appropriate fungicide or bacterial control for the species.",
        ],
    }
    steps = base_steps.get(issue_category, base_steps["water"])
    followup_summary = "; ".join(f"{k}: {v}" for k, v in answers.items()) if answers else "No follow-up answers provided yet."
    location_text = f" in {location}" if location else ""
    care_plan = (
        f"For {species}{location_text}, the issue looks most consistent with a {issue_category} problem.\n\n"
        "Step-by-step actions:\n"
        f"1. {steps[0]}\n"
        f"2. {steps[1]}\n"
        f"3. {steps[2]}\n"
        "4. Keep the plant stable for the next week and avoid adding extra stress like repotting unless root rot is obvious.\n\n"
        f"Relevant context from your answers: {followup_summary}\n\n"
        f"Expected recovery time: {RECOVERY_TIMES[issue_category]}.\n"
        f"What to look for in 1 week: {WATCH_FOR[issue_category]}."
    )
    checklist_items = [
        {"id": 1, "text": steps[0], "done": False},
        {"id": 2, "text": steps[1], "done": False},
        {"id": 3, "text": steps[2], "done": False},
        {"id": 4, "text": "Recheck progress in one week and compare new growth, leaf color, and droop.", "done": False},
    ]
    return CarePlanDraft(
        care_plan=care_plan,
        expected_recovery_time=RECOVERY_TIMES[issue_category],
        one_week_watch_for=WATCH_FOR[issue_category],
        summary=f"Generated care plan for a likely {issue_category} issue.",
        checklist_items=checklist_items,
    )


def _build_intake_summary(species: str, issue_category: str, questions: List[str], location: Optional[str]) -> str:
    location_text = f" in {location}" if location else ""
    question_lines = "\n".join([f"- {question}" for question in questions])
    return (
        f"I’ve identified the plant as {species}{location_text} with a likely {issue_category} issue.\n"
        "Before I prescribe a care plan, answer these 3 questions:\n"
        f"{question_lines}"
    )


def _knowledge_search_text(species: str, issue_category: str, location: Optional[str], message: str) -> str:
    parts = [species, issue_category, location or "", message]
    return " ".join(part for part in parts if part)


def _append_source_block(text: str, citations: List[Dict[str, str]], heading: str = "Sources") -> str:
    if not citations:
        return text
    return f"{text}\n\n{format_citations(citations, label=heading)}"


def _vision_model():
    if os.getenv("GROQ_API_KEY"):
        model_name = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        return ChatGroq(model=model_name, temperature=0.2)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=0.2, api_key=openai_key)

    return None


def analyze_plant_intake(
    message: str,
    image_data: Optional[str] = None,
    species_hint: Optional[str] = None,
    location: Optional[str] = None,
) -> Dict[str, object]:
    model = _vision_model()
    text = _trim(message)
    species = species_hint or _species_from_text(text)
    issue_category = _category_from_text(text)
    followup_questions = _followup_questions(issue_category)

    if model and image_data:
        prompt = (
            "Identify the plant species if missing, infer the issue category from the photo and message, "
            "and produce a short response asking exactly 2-3 clarifying questions before treatment."
        )
        try:
            structured = model.with_structured_output(IntakeAnalysis)
            analysis = structured.invoke(
                [
                    {
                        "role": "system",
                        "content": "You are an expert plant diagnostic assistant.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Message: {text}\nKnown species: {species}\nLocation: {location or 'Unknown'}\n{prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        ],
                    },
                ]
            )
            species = analysis.species or species
            issue_category = analysis.issue_category if analysis.issue_category in ISSUE_CATEGORIES else issue_category
            followup_questions = analysis.followup_questions[:3] or followup_questions
        except Exception:
            pass

    citations = search_care_knowledge(
        query=_knowledge_search_text(species, issue_category, location, text),
        issue_category=issue_category,
        species=species,
        limit=3,
    )
    summary = _append_source_block(
        _build_intake_summary(species, issue_category, followup_questions, location),
        citations,
        heading="Evidence from r/houseplants",
    )
    return {
        "species": species,
        "issue_category": issue_category,
        "followup_questions": followup_questions,
        "summary": summary,
        "citations": citations,
    }


def generate_care_plan(
    species: str,
    issue_category: str,
    answers: Optional[Dict[str, str]] = None,
    location: Optional[str] = None,
) -> Dict[str, str]:
    category = issue_category if issue_category in ISSUE_CATEGORIES else "water"
    draft = _care_plan_text(species or "Unknown", category, answers, location)
    citations = search_care_knowledge(
        query=_knowledge_search_text(species or "Unknown", category, location, draft.care_plan),
        issue_category=category,
        species=species,
        limit=3,
    )
    plan_text = _append_source_block(draft.care_plan, citations, heading="Community references")
    return {**draft.model_dump(), "care_plan": plan_text, "citations": citations}


def compare_check_in(
    species: str,
    issue_category: str,
    current_plan: str,
    previous_photo_data: Optional[str],
    new_photo_data: str,
) -> Dict[str, str]:
    model = _vision_model()
    overlay_notes = [
        "Left image is the earlier photo; right image is the newest check-in.",
        "Watch the same symptoms the care plan targets so changes are easy to compare week over week.",
    ]
    citations = search_care_knowledge(
        query=_knowledge_search_text(species, issue_category, None, current_plan),
        issue_category=issue_category,
        species=species,
        limit=2,
    )
    if model and new_photo_data:
        try:
            structured = model.with_structured_output(CheckInReview)
            review = structured.invoke(
                [
                    {
                        "role": "system",
                        "content": "You review a weekly plant check-in photo and update a care plan.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Species: {species}\nIssue category: {issue_category}\nCurrent care plan: {current_plan}\n"
                                    f"Prior weekly photo available: {'yes' if previous_photo_data else 'no'}\n"
                                    "Assess whether the plant is improving, worsening, or stable. "
                                    "Focus on yellowing, droop, spots, pest signs, and new growth. "
                                    "If a prior photo exists, compare against it from memory and the current care plan."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{new_photo_data}"}},
                        ],
                    },
                ]
            )
            overlay_notes = [
                "Left image is the earlier photo; right image is the newest check-in.",
                f"Agent focus: {review.health_status.replace('_', ' ')}.",
                review.comparison_summary,
            ]
            return {**review.model_dump(), "citations": citations, "overlay_notes": overlay_notes}
        except Exception:
            pass

    comparison_summary = (
        f"Saved the new weekly check-in photo for a {species} with a likely {issue_category} issue. "
        "Weekly vision review was unavailable, so the plan is being carried forward using the existing care notes."
    )
    plan_update = (
        f"Continue the current {issue_category} recovery plan, keep tracking the same symptoms, "
        f"and recheck for {WATCH_FOR[issue_category]}."
    )
    return {
        "comparison_summary": comparison_summary,
        "plan_update": plan_update,
        "health_status": "monitoring",
        "citations": citations,
        "overlay_notes": overlay_notes,
    }


class PlantDoctorAgent:
    def invoke(self, state: AgentState) -> Dict[str, object]:
        messages = list(state.get("messages", []))
        latest_user = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                latest_user = _trim(str(message.content))
                break

        if state.get("followup_answers"):
            care_plan = generate_care_plan(
                species=state.get("species", "Unknown"),
                issue_category=state.get("issue_category", "water"),
                answers=state.get("followup_answers"),
                location=state.get("location"),
            )
            response = AIMessage(content=care_plan["care_plan"])
            return {
                "messages": messages + [response],
                "species": state.get("species", "Unknown"),
                "location": state.get("location", "Unknown"),
                "issue_category": state.get("issue_category", "water"),
                "care_plan": care_plan["care_plan"],
                "diagnosis_ready": True,
                "citations": care_plan.get("citations", []),
            }

        analysis = analyze_plant_intake(
            message=latest_user,
            image_data=state.get("image_data"),
            species_hint=state.get("species") or None,
            location=state.get("location") or None,
        )
        response = AIMessage(content=analysis["summary"])
        return {
            "messages": messages + [response],
            "species": analysis["species"],
            "location": state.get("location", "Unknown"),
            "issue_category": analysis["issue_category"],
            "care_plan": "",
            "diagnosis_ready": False,
            "citations": analysis.get("citations", []),
        }


plant_doctor_agent = PlantDoctorAgent()