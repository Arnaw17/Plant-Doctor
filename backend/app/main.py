from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.agent import (
    UPLOAD_DIR,
    analyze_plant_intake,
    compare_check_in,
    generate_care_plan,
    plant_doctor_agent,
)
from app.core.reminders import (
  acknowledge_reminder,
  build_public_recovered_gallery,
  list_user_reminders,
  scan_due_care_reminders,
  start_reminder_scheduler,
  stop_reminder_scheduler,
)
from app.database import SessionLocal, engine
from app.models.models import CheckIn, Diagnosis, Photo, Plant, User
from app.security import create_access_token, decode_access_token, hash_password, verify_password

app = FastAPI(
    title="AI Plant Doctor API",
    description="Stateful plant registration and care workflow",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


class ChatMessageInput(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessageInput] = Field(default_factory=list)
    image_data: Optional[str] = None
    species_hint: Optional[str] = None
    location: Optional[str] = None
    stage: str = "intake"
    followup_answers: Optional[Dict[str, str]] = None


class SignupRequest(BaseModel):
    email: str
    password: str


class FollowUpAnswers(BaseModel):
    answers: Dict[str, str]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _save_photo_bytes(data: bytes, original_name: str, prefix: str) -> Path:
    suffix = Path(original_name).suffix or ".jpg"
    filename = f"{prefix}_{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / filename
    path.write_bytes(data)
    return path


async def _persist_photo(
    db: Session,
    plant_id: int,
    photo: Optional[UploadFile],
    image_data: Optional[str],
    prefix: str,
    raw_bytes: Optional[bytes] = None,
    original_name: Optional[str] = None,
) -> tuple[Optional[Photo], Optional[str]]:
    if raw_bytes is None and photo is None and not image_data:
        return None, None

    if raw_bytes is not None:
        data = raw_bytes
        name_hint = original_name or f"{prefix}.jpg"
    elif photo is not None:
        data = await photo.read()
        name_hint = photo.filename or f"{prefix}.jpg"
    else:
        try:
            data = base64.b64decode(image_data or "")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        name_hint = f"{prefix}.jpg"

    path = _save_photo_bytes(data, name_hint, prefix)
    row = Photo(plant_id=plant_id, url=f"/uploads/{path.name}", file_path=str(path))
    db.add(row)
    db.flush()
    return row, base64.b64encode(data).decode("ascii")


def _get_latest_diagnosis(db: Session, plant_id: int) -> Optional[Diagnosis]:
    return (
        db.query(Diagnosis)
        .filter(Diagnosis.plant_id == plant_id)
        .order_by(Diagnosis.id.desc())
        .first()
    )


def _get_photo_map(db: Session, photo_ids: List[int]) -> Dict[int, Photo]:
  if not photo_ids:
    return {}
  photos = db.query(Photo).filter(Photo.id.in_(photo_ids)).all()
  return {photo.id: photo for photo in photos}


def _serialize_check_in(db: Session, check_in: CheckIn) -> Dict[str, object]:
  photo_ids = [photo_id for photo_id in [check_in.photo_id, check_in.previous_photo_id] if photo_id]
  photo_map = _get_photo_map(db, photo_ids)
  current_photo = photo_map.get(check_in.photo_id) if check_in.photo_id else None
  previous_photo = photo_map.get(check_in.previous_photo_id) if check_in.previous_photo_id else None
  return {
    "id": check_in.id,
    "photo_url": current_photo.url if current_photo else None,
    "previous_photo_url": previous_photo.url if previous_photo else None,
    "user_note": check_in.user_note,
    "comparison_summary": check_in.comparison_summary,
    "plan_update": check_in.plan_update,
    "health_status": check_in.health_status,
    "created_at": check_in.created_at.strftime("%Y-%m-%d %H:%M:%S") if check_in.created_at else None,
  }


def _plant_status_label(plant: Plant, diagnosis: Optional[Diagnosis]) -> str:
    status = (plant.diagnosis_status or "").lower()
    issue = (plant.issue_category or (diagnosis.issue_category if diagnosis else "") or "").lower()
    if status in {"questions_pending"}:
        return "diagnosing"
    if status in {"critical"} or issue == "disease":
        return "critical"
    if status in {"active", "monitoring"}:
        return "recovering"
    if status in {"healthy", "stable"}:
        return "healthy"
    return "recovering" if diagnosis else "diagnosing"


def _check_in_focus(issue_category: str) -> str:
    focus_map = {
        "water": "Is the leaf yellowing reducing?",
        "light": "Is new growth less stretched and more upright?",
        "pest": "Are there fewer visible pests or fresh spots?",
        "nutrient": "Are the newest leaves greener and larger?",
        "disease": "Has the spread slowed or stopped?",
    }
    return focus_map.get(issue_category, "Is the plant looking better than last week?")


def _serialize_plant(db: Session, plant: Plant) -> Dict[str, object]:
    diagnosis = _get_latest_diagnosis(db, plant.id)
    latest_photo = (
        db.query(Photo)
        .filter(Photo.plant_id == plant.id)
        .order_by(Photo.id.desc())
        .first()
    )
    check_ins = (
        db.query(CheckIn)
        .filter(CheckIn.plant_id == plant.id)
        .order_by(CheckIn.id.desc())
        .all()
    )
    latest_check_in = _serialize_check_in(db, check_ins[0]) if check_ins else None
    return {
        "id": plant.id,
        "user_id": plant.user_id,
        "nickname": plant.nickname,
        "species": plant.species,
        "location": plant.location,
        "problem_description": plant.problem_description,
        "issue_category": plant.issue_category,
        "diagnosis_status": plant.diagnosis_status,
        "status_label": _plant_status_label(plant, diagnosis),
        "current_care_plan": plant.current_care_plan,
        "care_plan_items": plant.care_plan_items or [],
        "expected_recovery_time": plant.expected_recovery_time,
        "one_week_watch_for": plant.one_week_watch_for,
        "followup_questions": plant.followup_questions or [],
        "followup_answers": plant.followup_answers or {},
        "latest_photo_url": latest_photo.url if latest_photo else None,
        "latest_check_in": latest_check_in,
        "next_watering_at": plant.next_watering_at.strftime("%Y-%m-%d %H:%M:%S") if plant.next_watering_at else None,
        "next_fertilizing_at": plant.next_fertilizing_at.strftime("%Y-%m-%d %H:%M:%S") if plant.next_fertilizing_at else None,
        "latest_diagnosis": {
            "id": diagnosis.id,
            "status": diagnosis.status,
            "issue_category": diagnosis.issue_category,
            "species": diagnosis.species,
            "care_plan": diagnosis.care_plan,
        }
        if diagnosis
        else None,
        "check_in_count": len(check_ins),
    }
    from app.models.models import Base
from app.database import engine
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

@app.on_event("startup")
def _startup() -> None:
    # Create all tables first
    Base.metadata.create_all(bind=engine)

    # Reset sequences only if the tables exist
    with engine.begin() as connection:
        for table_name in (
            "users",
            "plants",
            "diagnoses",
            "photos",
            "check_ins",
            "reminders",
        ):
@app.on_event("startup")
def _startup() -> None:
    # Create all database tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Reset sequences safely
    with engine.begin() as connection:
        for table_name in (
            "users",
            "plants",
            "diagnoses",
            "photos",
            "check_ins",
            "reminders",
        ):
            try:
                connection.execute(
                    text(
                        f"""
                        SELECT setval(
                            pg_get_serial_sequence('{table_name}', 'id'),
                            COALESCE((SELECT MAX(id) FROM {table_name}), 1),
                            (SELECT MAX(id) IS NOT NULL FROM {table_name})
                        )
                        """
                    )
                )
            except ProgrammingError:
                # Ignore if the table or sequence does not exist yet
                pass

    start_reminder_scheduler()
    scan_due_care_reminders()


@app.on_event("shutdown")
def _shutdown() -> None:
    stop_reminder_scheduler()


@app.get("/")
def read_root():
    return RedirectResponse(url="/app")


@app.get("/api/health")
def health():
    return {"status": "online", "agent": "AI Plant Doctor API"}


@app.post("/api/auth/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(email=email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(str(user.id), user.email)
    return {"user_id": user.id, "email": user.email, "access_token": token, "token_type": "bearer"}


@app.post("/api/auth/login")
def login(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(str(user.id), user.email)
    return {"user_id": user.id, "email": user.email, "access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {"user_id": current_user.id, "email": current_user.email}


@app.post("/api/plants/register")
async def register_plant(
    problem_description: str = Form(...),
    nickname: Optional[str] = Form(None),
    species: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    image_data: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    plant = Plant(
        user_id=current_user.id,
        nickname=nickname,
        species=species or "Unknown",
        location=location,
        problem_description=problem_description,
        issue_category=None,
        diagnosis_status="questions_pending",
        followup_questions=[],
        care_plan_items=[],
    )
    db.add(plant)
    db.flush()

    registration_photo_b64 = image_data
    photo_bytes = None
    photo_name = None
    if photo is not None:
        photo_bytes = await photo.read()
        photo_name = photo.filename or f"plant_{plant.id}.jpg"
        registration_photo_b64 = base64.b64encode(photo_bytes).decode("ascii")

    try:
      analysis = analyze_plant_intake(
        message=problem_description,
        image_data=registration_photo_b64,
        species_hint=species,
        location=location,
      )
    except BaseException:
      fallback_species = species or "Unknown"
      fallback_issue = "water"
      fallback_questions = [
        "How often do you water, and does the pot fully drain?",
        "Are the leaves soft and yellow or crisp and brown?",
        "Has the soil stayed wet for more than 2-3 days?",
      ]
      fallback_summary = (
        f"I’ve identified the plant as {fallback_species}{f' in {location}' if location else ''} with a likely {fallback_issue} issue.\n"
        "Before I prescribe a care plan, answer these 3 questions:\n"
        + "\n".join(f"- {question}" for question in fallback_questions)
      )
      analysis = {
        "species": fallback_species,
        "issue_category": fallback_issue,
        "followup_questions": fallback_questions,
        "summary": fallback_summary,
        "citations": [],
      }

    plant_species = species or str(analysis["species"]) or "Unknown"
    plant.species = plant_species
    plant.issue_category = str(analysis["issue_category"])
    plant.followup_questions = analysis["followup_questions"]

    stored_photo, photo_b64 = await _persist_photo(
        db,
        plant.id,
        None if photo_bytes is not None else photo,
        None if photo_bytes is not None else image_data,
        f"plant_{plant.id}",
        raw_bytes=photo_bytes,
        original_name=photo_name,
    )

    diagnosis = Diagnosis(
        plant_id=plant.id,
        diagnosis_result=str(analysis["summary"]),
        issue_category=str(analysis["issue_category"]),
        species=plant_species,
        followup_questions=analysis["followup_questions"],
        status="questions_pending",
    )
    db.add(diagnosis)
    db.commit()
    db.refresh(plant)
    db.refresh(diagnosis)

    return {
        "plant": _serialize_plant(db, plant),
        "diagnosis_id": diagnosis.id,
        "uploaded_photo_url": stored_photo.url if stored_photo else None,
        "followup_questions": analysis["followup_questions"],
        "assistant_message": analysis["summary"],
        "image_data_saved": bool(photo_b64),
    }


@app.post("/api/plants/{plant_id}/follow-up")
def submit_followup(
    plant_id: int,
    payload: FollowUpAnswers,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    plant = db.query(Plant).filter(Plant.id == plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    if plant.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")

    diagnosis = _get_latest_diagnosis(db, plant_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="No diagnosis exists for this plant")

    issue_category = diagnosis.issue_category or plant.issue_category or "water"
    species = plant.species or diagnosis.species or "Unknown"
    plan = generate_care_plan(species=species, issue_category=issue_category, answers=payload.answers, location=plant.location)
    now = datetime.utcnow()
    watering_days = {"water": 3, "light": 5, "pest": 7, "nutrient": 4, "disease": 4}

    plant.followup_answers = payload.answers
    plant.current_care_plan = plan["care_plan"]
    plant.care_plan_items = plan["checklist_items"]
    plant.expected_recovery_time = plan["expected_recovery_time"]
    plant.one_week_watch_for = plan["one_week_watch_for"]
    plant.diagnosis_status = "active"
    plant.issue_category = issue_category
    plant.next_watering_at = now + timedelta(days=watering_days.get(issue_category, 7))
    if issue_category == "nutrient" or "feed" in plan["care_plan"].lower():
      plant.next_fertilizing_at = now + timedelta(days=14)
    diagnosis.followup_answers = payload.answers
    diagnosis.care_plan = plan["care_plan"]
    diagnosis.expected_recovery_time = plan["expected_recovery_time"]
    diagnosis.one_week_watch_for = plan["one_week_watch_for"]
    diagnosis.status = "active"
    db.commit()

    return {
        "plant_id": plant.id,
        "care_plan": plan["care_plan"],
        "checklist_items": plan["checklist_items"],
        "expected_recovery_time": plan["expected_recovery_time"],
        "one_week_watch_for": plan["one_week_watch_for"],
        "status": "active",
    }


@app.post("/api/plants/{plant_id}/check-ins")
async def weekly_check_in(
    plant_id: int,
    photo: UploadFile = File(...),
    note: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    plant = db.query(Plant).filter(Plant.id == plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    if plant.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")

    latest_photo = (
        db.query(Photo)
        .filter(Photo.plant_id == plant.id)
        .order_by(Photo.id.desc())
        .first()
    )
    latest_diagnosis = _get_latest_diagnosis(db, plant.id)
    issue_category = plant.issue_category or (latest_diagnosis.issue_category if latest_diagnosis else "water")
    species = plant.species or (latest_diagnosis.species if latest_diagnosis else "Unknown")
    current_plan = plant.current_care_plan or (latest_diagnosis.care_plan if latest_diagnosis and latest_diagnosis.care_plan else "")

    stored_photo, new_photo_b64 = await _persist_photo(db, plant.id, photo, None, f"checkin_{plant.id}")
    previous_b64 = None
    if latest_photo:
        try:
            previous_b64 = base64.b64encode(Path(latest_photo.file_path).read_bytes()).decode("ascii")
        except FileNotFoundError:
            previous_b64 = None

    review = compare_check_in(
        species=species,
        issue_category=issue_category,
        current_plan=current_plan,
        previous_photo_data=previous_b64,
        new_photo_data=new_photo_b64 or "",
    )
    if note:
        review["comparison_summary"] = f"User note: {note}. {review['comparison_summary']}"

    comparison_media = {
      "plant_id": plant.id,
      "current_photo_url": stored_photo.url if stored_photo else None,
      "previous_photo_url": latest_photo.url if latest_photo else None,
      "overlay_notes": review.get("overlay_notes", []),
      "citations": review.get("citations", []),
    }

    check_in = CheckIn(
        plant_id=plant.id,
        photo_id=stored_photo.id if stored_photo else None,
        previous_photo_id=latest_photo.id if latest_photo else None,
        user_note=note,
        comparison_summary=review["comparison_summary"],
        plan_update=review["plan_update"],
        health_status=review["health_status"],
    )
    db.add(check_in)
    db.flush()

    if current_plan:
        plant.current_care_plan = f"{current_plan}\n\nWeekly check-in update: {review['plan_update']}"
    else:
        plant.current_care_plan = review["plan_update"]
    plant.diagnosis_status = review["health_status"]
    if latest_diagnosis:
        latest_diagnosis.status = review["health_status"]

    db.commit()

    return {
        "check_in_id": check_in.id,
        "comparison_summary": review["comparison_summary"],
        "plan_update": review["plan_update"],
        "health_status": review["health_status"],
        "comparison_focus": _check_in_focus(issue_category),
        "latest_photo_url": stored_photo.url if stored_photo else None,
      "comparison_media": comparison_media,
    }


@app.get("/api/plants/{plant_id}")
def get_plant(plant_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    plant = db.query(Plant).filter(Plant.id == plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    if plant.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")
    return _serialize_plant(db, plant)


@app.get("/api/users/{user_id}/plants")
def list_user_plants(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed")

    plants = db.query(Plant).filter(Plant.user_id == user_id).order_by(Plant.id.desc()).all()
    return {"user_id": user_id, "plants": [_serialize_plant(db, plant) for plant in plants]}


@app.get("/api/me/plants")
def my_plants(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    plants = db.query(Plant).filter(Plant.user_id == current_user.id).order_by(Plant.id.desc()).all()
    return {"user_id": current_user.id, "plants": [_serialize_plant(db, plant) for plant in plants]}


@app.get("/api/plants/{plant_id}/photos")
def plant_photos(plant_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    plant = db.query(Plant).filter(Plant.id == plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    if plant.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")

    photos = (
        db.query(Photo)
        .filter(Photo.plant_id == plant_id)
        .order_by(Photo.id.asc())
        .all()
    )
    return {
        "plant_id": plant_id,
        "photos": [
            {
                "id": photo.id,
                "url": photo.url,
                "file_path": photo.file_path,
                "uploaded_at": photo.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if photo.uploaded_at else None,
            }
            for photo in photos
        ],
    }


@app.patch("/api/plants/{plant_id}/checklist/{item_id}")
def toggle_checklist_item(
    plant_id: int,
    item_id: int,
    done: bool = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    plant = db.query(Plant).filter(Plant.id == plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    if plant.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")

    items = list(plant.care_plan_items or [])
    updated = False
    for item in items:
        if int(item.get("id", 0)) == item_id:
            item["done"] = done
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Checklist item not found")

    plant.care_plan_items = items
    db.commit()
    return {"plant_id": plant_id, "checklist_items": items}


@app.get("/api/history")
def get_diagnostic_history(db: Session = Depends(get_db)):
    records = (
        db.query(Diagnosis)
        .join(Plant, Diagnosis.plant_id == Plant.id)
        .order_by(Diagnosis.id.desc())
        .all()
    )

    history_list = []
    for diagnosis in records:
        plant = diagnosis.plant
        history_list.append(
            {
                "diagnosis_id": diagnosis.id,
                "plant_id": diagnosis.plant_id,
                "species": plant.species if plant else "Unknown",
                "location": plant.location if plant else "Unknown",
                "issue_category": diagnosis.issue_category,
                "care_plan": diagnosis.care_plan,
                "status": diagnosis.status,
                "diagnosis_result": diagnosis.diagnosis_result,
                "created_at": diagnosis.created_at.strftime("%Y-%m-%d %H:%M:%S") if diagnosis.created_at else None,
            }
        )

    return {"history": history_list}


@app.get("/api/me/reminders")
def my_reminders(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
  return {"reminders": list_user_reminders(db, current_user.id)}


@app.post("/api/reminders/{reminder_id}/ack")
def acknowledge_user_reminder(reminder_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
  reminder = acknowledge_reminder(db, reminder_id, current_user.id)
  if not reminder:
    raise HTTPException(status_code=404, detail="Reminder not found")
  return {"reminder": reminder}


@app.get("/api/public/recovered-gallery")
def public_recovered_gallery(symptom: Optional[str] = None, limit: int = 12, db: Session = Depends(get_db)):
  safe_limit = max(1, min(limit, 24))
  return {
    "symptom": symptom or "all",
    "gallery": build_public_recovered_gallery(db, symptom=symptom, limit=safe_limit),
  }


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    formatted_messages = []
    for message in payload.history:
        if message.role == "user":
            formatted_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            formatted_messages.append(AIMessage(content=message.content))

    formatted_messages.append(HumanMessage(content=payload.message))

    if payload.followup_answers or payload.stage == "followup":
        answers = payload.followup_answers or {"response": payload.message}
        analysis = analyze_plant_intake(
            message=payload.message,
            image_data=payload.image_data,
            species_hint=payload.species_hint,
            location=payload.location,
        )
        plan = generate_care_plan(
            species=str(analysis["species"]),
            issue_category=str(analysis["issue_category"]),
            answers=answers,
            location=payload.location,
        )
        return {
            "response": plan["care_plan"],
            "extracted_data": {
                "species": analysis["species"],
                "location": payload.location or "Unknown",
                "issue_category": analysis["issue_category"],
                "followup_questions": analysis["followup_questions"],
                "care_plan": plan["care_plan"],
                "expected_recovery_time": plan["expected_recovery_time"],
                "one_week_watch_for": plan["one_week_watch_for"],
            },
          "citations": plan.get("citations", []),
            "next_step": "care_plan",
        }

    initial_state = {
        "messages": formatted_messages,
        "species": payload.species_hint or "",
        "location": payload.location or "",
        "issue_category": "unknown",
        "diagnosis_ready": False,
        "image_data": payload.image_data,
    }
    final_output = plant_doctor_agent.invoke(initial_state)
    analysis = analyze_plant_intake(
        message=payload.message,
        image_data=payload.image_data,
        species_hint=payload.species_hint,
        location=payload.location,
    )

    return {
        "response": final_output["messages"][-1].content if final_output.get("messages") else analysis["summary"],
        "extracted_data": {
            "species": final_output.get("species", analysis["species"]),
            "location": final_output.get("location", payload.location or "Unknown"),
            "issue_category": final_output.get("issue_category", analysis["issue_category"]),
            "followup_questions": analysis["followup_questions"],
            "saved_to_db": False,
        },
      "citations": final_output.get("citations", analysis.get("citations", [])),
        "next_step": "followup_questions",
    }


APP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Plant Doctor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f8f0;
      --bg-2: #e7f1e2;
      --panel: rgba(255, 255, 255, 0.86);
      --panel-strong: #ffffff;
      --panel-soft: #f7fbf4;
      --text: #1f2d22;
      --muted: #66786b;
      --accent: #3f8f52;
      --accent-2: #89b86a;
      --warn: #b87a28;
      --danger: #b54b4b;
      --line: rgba(44, 73, 53, 0.12);
      --shadow: 0 24px 60px rgba(39, 73, 50, 0.12);
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family: "Segoe UI", "Aptos", system-ui, sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(137, 184, 106, 0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(63, 143, 82, 0.16), transparent 26%),
        linear-gradient(180deg, var(--bg), var(--bg-2));
      min-height:100vh;
    }
    a { color:inherit; }
    .wrap { max-width:1440px; margin:0 auto; padding:28px 22px 34px; }
    .shell { position:relative; }
    .shell::before,
    .shell::after {
      content:"";
      position:fixed;
      inset:auto;
      width:380px;
      height:380px;
      border-radius:50%;
      filter:blur(24px);
      pointer-events:none;
      opacity:.35;
      z-index:-1;
    }
    .shell::before { top:-120px; left:-110px; background:rgba(137,184,106,.34); }
    .shell::after { right:-120px; bottom:-120px; background:rgba(63,143,82,.18); }
    .topbar {
      display:flex;
      gap:16px;
      justify-content:space-between;
      align-items:flex-start;
      margin-bottom:20px;
      padding:18px 20px;
      border:1px solid var(--line);
      border-radius:26px;
      background:rgba(255,255,255,.58);
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
    }
    .brand { font-size:28px; font-weight:800; letter-spacing:-0.04em; }
    .brand-copy { margin-top:6px; max-width:700px; color:var(--muted); line-height:1.5; }
    .muted { color:var(--muted); }
    .grid { display:grid; gap:16px; }
    .hero {
      display:grid;
      grid-template-columns: 1fr;
      gap:16px;
      margin-bottom:18px;
    }
    .card {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:24px;
      padding:18px;
      box-shadow:var(--shadow);
      backdrop-filter: blur(16px);
    }
    .card.strong { background:var(--panel-strong); }
    .card.soft { background:var(--panel-soft); }
    .two { grid-template-columns: 1fr; }
    .three { grid-template-columns: 1fr; }
    .workspace { align-items:start; }
    .left-rail { position: static; }
    .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    input, textarea, button, select {
      width:100%;
      border-radius:16px;
      border:1px solid var(--line);
      background:#fff;
      color:var(--text);
      padding:12px 14px;
      font:inherit;
    }
    textarea { min-height:110px; resize:vertical; }
    button {
      cursor:pointer;
      background:linear-gradient(135deg, #2f7e45, #7eb15f);
      color:white;
      border:0;
      font-weight:700;
      box-shadow:0 12px 26px rgba(63,143,82,.24);
    }
    button.secondary { background:#edf4e9; color:var(--text); border:1px solid var(--line); box-shadow:none; }
    button.ghost { background:transparent; border:1px solid var(--line); color:var(--text); box-shadow:none; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .pill {
      display:inline-flex;
      padding:7px 12px;
      border-radius:999px;
      font-size:12px;
      font-weight:700;
      background:#edf4e9;
      color:var(--accent);
      border:1px solid var(--line);
    }
    .pill.healthy { background:rgba(63,143,82,.12); color:var(--accent); }
    .pill.recovering { background:rgba(126,177,95,.16); color:#4f7f33; }
    .pill.critical { background:rgba(181,75,75,.12); color:var(--danger); }
    .pill.diagnosing { background:rgba(184,122,40,.14); color:var(--warn); }
    .card-list { display:grid; gap:12px; }
    .plant-card {
      padding:14px;
      border-radius:18px;
      border:1px solid var(--line);
      background:linear-gradient(180deg, rgba(255,255,255,.95), rgba(242,248,238,.98));
      cursor:pointer;
    }
    .plant-card.active { outline:2px solid rgba(63,143,82,.28); }
    .plant-card img { width:100%; height:150px; object-fit:cover; border-radius:14px; margin-bottom:10px; background:#eef4ea; }
    .detail { display:grid; gap:16px; }
    .detail-grid { display:grid; grid-template-columns: 1fr; gap:16px; }
    .section-head {
      display:flex;
      align-items:flex-end;
      justify-content:space-between;
      gap:12px;
      margin-bottom:12px;
    }
    .section-head h3 {
      margin:0;
      letter-spacing:-0.03em;
    }
    .preview {
      width:100%;
      max-height:240px;
      object-fit:cover;
      border-radius:18px;
      background:#eef4ea;
      border:1px solid var(--line);
    }
    .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }
    .gallery img { width:100%; height:150px; object-fit:cover; border-radius:14px; border:1px solid var(--line); }
    .comparison { display:grid; gap:12px; }
    .comparison-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .compare-card {
      position:relative;
      overflow:hidden;
      border-radius:20px;
      border:1px solid var(--line);
      background:linear-gradient(180deg, rgba(255,255,255,.96), rgba(242,248,238,.98));
    }
    .compare-card img {
      width:100%;
      display:block;
      aspect-ratio: 4 / 5;
      object-fit:cover;
      background:#eef4ea;
    }
    .compare-caption {
      position:absolute;
      left:12px;
      right:12px;
      bottom:12px;
      padding:10px 12px;
      border-radius:14px;
      background:rgba(18, 31, 21, 0.72);
      color:#fff;
      font-size:13px;
      line-height:1.45;
      backdrop-filter: blur(10px);
    }
    .compare-caption strong { display:block; margin-bottom:3px; }
    .compare-notes { display:grid; gap:8px; }
    .compare-note {
      padding:10px 12px;
      border-radius:14px;
      background:#f6faf2;
      border:1px solid var(--line);
      color:var(--text);
      font-size:14px;
      line-height:1.5;
    }
    .community-toolbar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
    .chip-btn {
      width:auto;
      padding:8px 12px;
      border-radius:999px;
      background:#edf4e9;
      color:var(--text);
      border:1px solid var(--line);
      box-shadow:none;
      font-size:13px;
    }
    .community-grid { display:grid; gap:12px; }
    .community-card {
      display:grid;
      gap:10px;
      padding:14px;
      border-radius:18px;
      border:1px solid var(--line);
      background:linear-gradient(180deg, rgba(255,255,255,.96), rgba(242,248,238,.98));
    }
    .community-card .pair {
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:10px;
    }
    .community-card img {
      width:100%;
      height:160px;
      object-fit:cover;
      border-radius:14px;
      border:1px solid var(--line);
      background:#eef4ea;
    }
    .reminders-list { display:grid; gap:10px; }
    .reminder-item {
      padding:12px;
      border-radius:14px;
      background:#f6faf2;
      border:1px solid var(--line);
      display:grid;
      gap:8px;
    }
    .reminder-head { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
    .checklist { display:grid; gap:10px; }
    .check-item {
      display:flex;
      gap:10px;
      align-items:flex-start;
      padding:10px 12px;
      border-radius:14px;
      background:#f5faf1;
      border:1px solid var(--line);
    }
    .check-item.done { opacity:.72; text-decoration:line-through; }
    .hidden { display:none !important; }
    .statusline { font-size:14px; color:var(--muted); margin-top:8px; white-space:pre-wrap; line-height:1.55; }
    .progress { width:100%; height:10px; border-radius:999px; overflow:hidden; background:#e4eee0; border:1px solid var(--line); }
    .progress > div { height:100%; width:0%; background:linear-gradient(90deg,#2f7e45,#86b860); }
    .small { font-size:13px; }
    .question { padding:10px 12px; border-radius:14px; background:#f6faf2; border:1px solid var(--line); margin-bottom:10px; }
    .split { display:grid; gap:12px; grid-template-columns: 1fr 1fr; }
    .headline {
      display:grid;
      gap:14px;
      # padding:10px 0 2px;
    }
    .eyebrow {
      display:inline-flex;
      align-items:center;
      gap:8px;
      width:max-content;
      padding:7px 12px;
      border-radius:999px;
      background:rgba(63,143,82,.12);
      color:var(--accent);
      border:1px solid rgba(63,143,82,.14);
      font-size:12px;
      font-weight:700;
      letter-spacing:.03em;
      text-transform:uppercase;
    }
    .hero-title {
      font-size:clamp(34px, 5vw, 58px);
      line-height:1.02;
      letter-spacing:-0.05em;
      margin:0;
    }
    .hero-copy { margin:0; max-width:72ch; color:var(--muted); font-size:16px; line-height:1.65; }
    .stats { display:grid; grid-template-columns:1fr; gap:12px; }
    .stat {
      padding:14px;
      border-radius:18px;
      background:rgba(255,255,255,.72);
      border:1px solid var(--line);
    }
    .stat strong { display:block; font-size:20px; margin-bottom:4px; }
    .stack { display:grid; gap:16px; }
    .side-panel {
      display:grid;
      gap:12px;
      align-content:start;
    }
    .leaf-band {
      min-height:240px;
      border-radius:24px;
      padding:18px;
      background:
        linear-gradient(160deg, rgba(63,143,82,.95), rgba(126,177,95,.88)),
        radial-gradient(circle at top right, rgba(255,255,255,.2), transparent 30%);
      color:white;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
      box-shadow:0 20px 50px rgba(47,126,69,.24);
    }
    .leaf-band .muted { color:rgba(255,255,255,.84); }
    .leaf-chip {
      display:inline-flex;
      width:max-content;
      padding:8px 12px;
      border-radius:999px;
      background:rgba(255,255,255,.18);
      border:1px solid rgba(255,255,255,.18);
      font-size:12px;
      font-weight:700;
      letter-spacing:.03em;
      text-transform:uppercase;
    }
    @media (max-width: 1080px) {
      .hero, .two, .detail-grid, .split, .three, .stats { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap shell">
    <div class="topbar">
      <div>
        <div class="brand">Plant Doctor</div>
        <div class="brand-copy">A calm plant care workspace for diagnosis, follow-up questions, care plans, and weekly recovery tracking.</div>
      </div>
      <div class="row">
        <span id="userBadge" class="pill diagnosing">Signed out</span>
        <button id="logoutBtn" class="ghost hidden" onclick="logout()">Logout</button>
      </div>
    </div>

    <div class="hero">
      <div class="card strong headline">
        <div class="eyebrow">Plant doctor workspace</div>
        <h1 class="hero-title">See what your plant needs, then track recovery with one focused dashboard.</h1>
        <p class="hero-copy">Register a plant with a photo, answer clarifying questions, generate a care checklist, and compare weekly check-ins without leaving the screen.</p>
        <div class="stats">
          <div class="stat"><strong>Diagnose</strong><span class="muted small">Capture symptoms and image context.</span></div>
          <div class="stat"><strong>Prescribe</strong><span class="muted small">Turn answers into a practical care plan.</span></div>
          <div class="stat"><strong>Recover</strong><span class="muted small">Review photos and progress over time.</span></div>
        </div>
      </div>
      <div class="side-panel">
        <div class="leaf-band">
          <div>
            <div class="leaf-chip">Calm plant care</div>
            <h3 style="margin:14px 0 8px; font-size:26px; line-height:1.05;">A softer, more botanical layout for the same workflow.</h3>
            <p class="muted" style="margin:0;">Designed to feel like a plant health panel instead of a generic admin dashboard.</p>
          </div>
          <div class="row" style="justify-content:space-between; align-items:end;">
            <div>
              <div class="muted small">Focus</div>
              <strong>Diagnosis, care, recovery</strong>
            </div>
            <div class="pill" style="background:rgba(255,255,255,.18); color:white; border-color:rgba(255,255,255,.18);">Live</div>
          </div>
        </div>
        <div id="authView" class="grid three">
          <div class="card strong">
            <h3>Sign up</h3>
            <input id="signupEmail" placeholder="Email" />
            <div style="height:10px"></div>
            <input id="signupPassword" type="password" placeholder="Password" />
            <div style="height:10px"></div>
            <button onclick="signup()">Create account</button>
            <div id="signupStatus" class="statusline"></div>
          </div>
          <div class="card strong">
            <h3>Login</h3>
            <input id="loginEmail" placeholder="Email" />
            <div style="height:10px"></div>
            <input id="loginPassword" type="password" placeholder="Password" />
            <div style="height:10px"></div>
            <button onclick="login()">Login</button>
            <div id="loginStatus" class="statusline"></div>
          </div>
          <div class="card soft">
            <h3>What this app does</h3>
            <div class="muted small">Register a plant with a photo, answer follow-up questions, get a checklist care plan, track weekly recovery photos, and keep a visual gallery per plant.</div>
          </div>
          <div class="card soft">
            <div class="section-head" style="margin-bottom:10px;">
              <h3 style="margin:0;">Community recovered gallery</h3>
              <span class="pill recovering">Public</span>
            </div>
            <div class="community-toolbar">
              <button class="chip-btn" onclick="setCommunitySymptom('all')">All</button>
              <button class="chip-btn" onclick="setCommunitySymptom('water')">Water</button>
              <button class="chip-btn" onclick="setCommunitySymptom('light')">Light</button>
              <button class="chip-btn" onclick="setCommunitySymptom('pest')">Pest</button>
              <button class="chip-btn" onclick="setCommunitySymptom('nutrient')">Nutrient</button>
              <button class="chip-btn" onclick="setCommunitySymptom('disease')">Disease</button>
            </div>
            <div id="communityGallery" class="community-grid">
              <div class="muted small">Loading recovered cases...</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="dashboardView" class="grid two workspace hidden" style="margin-top:16px;">
      <div class="grid left-rail" style="gap:16px;">
        <div class="card">
          <div class="section-head">
            <h3>Register plant</h3>
            <span class="pill">Intake</span>
          </div>
          <div class="split">
            <input id="plantNickname" placeholder="Nickname (optional)" />
            <input id="plantSpecies" placeholder="Species (optional)" />
          </div>
          <div style="height:10px"></div>
          <input id="plantLocation" placeholder="Location" />
          <div style="height:10px"></div>
          <textarea id="plantProblem" placeholder="Describe what is wrong"></textarea>
          <div style="height:10px"></div>
          <input id="plantPhoto" type="file" accept="image/*" onchange="previewFile(event,'plantPreview')" />
          <div style="height:10px"></div>
          <img id="plantPreview" class="preview hidden" alt="Preview" />
          <div style="height:10px"></div>
          <div class="progress"><div id="registerProgress"></div></div>
          <div style="height:10px"></div>
          <button onclick="registerPlant()">Upload and diagnose</button>
          <div id="registerStatus" class="statusline"></div>
        </div>

        <div class="card">
          <div class="section-head" style="margin-bottom:10px;">
            <h3>My plants</h3>
            <button id="refreshBtn" class="secondary" onclick="loadPlants()">Refresh</button>
          </div>
          <div class="muted small" style="margin-bottom:12px;">Select a plant to open its diagnosis, follow-ups, and recovery history.</div>
          <div id="plantList" class="card-list"></div>
        </div>
      </div>

      <div class="detail">
        <div class="card">
          <div class="section-head" style="align-items:flex-start; margin-bottom:10px;">
            <div>
              <h3 id="detailTitle" style="margin:0;">Select a plant</h3>
              <div id="detailMeta" class="muted small"></div>
            </div>
            <span id="detailStatus" class="pill diagnosing">diagnosing</span>
          </div>
          <div style="height:12px"></div>
          <div class="detail-grid">
            <div>
              <img id="detailPhoto" class="preview hidden" alt="Latest plant photo" />
              <div id="detailDescription" class="statusline"></div>
            </div>
            <div class="card" style="margin:0;">
              <h4 style="margin-top:0;">Weekly check-in</h4>
              <div id="comparisonFocus" class="muted small">Upload a weekly photo and one-line update.</div>
              <div style="height:10px"></div>
              <input id="checkInNote" placeholder="One-line update" />
              <div style="height:10px"></div>
              <input id="checkInPhoto" type="file" accept="image/*" onchange="previewFile(event,'checkInPreview')" />
              <div style="height:10px"></div>
              <img id="checkInPreview" class="preview hidden" alt="Check-in preview" />
              <div style="height:10px"></div>
              <div class="progress"><div id="checkInProgress"></div></div>
              <div style="height:10px"></div>
              <button onclick="submitCheckIn()">Submit check-in</button>
              <div id="checkInStatus" class="statusline"></div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-head">
            <h3>Follow-up questions</h3>
            <span class="pill diagnosing">Clarify</span>
          </div>
          <div id="followupBox" class="muted small">Register a plant to get clarifying questions before treatment.</div>
          <div style="height:12px"></div>
          <div id="followupFields"></div>
          <div style="height:10px"></div>
          <button id="followupBtn" class="hidden" onclick="submitFollowup()">Generate care plan</button>
        </div>

        <div class="card">
          <div class="section-head">
            <h3>Care plan</h3>
            <span class="pill healthy">Treatment</span>
          </div>
          <div id="carePlanText" class="statusline">No care plan yet.</div>
          <div style="height:12px"></div>
          <div id="careChecklist" class="checklist"></div>
          <div style="height:12px"></div>
          <div class="split">
            <div class="card" style="margin:0;">
              <div class="muted small">Expected recovery</div>
              <div id="recoveryTime">-</div>
            </div>
            <div class="card" style="margin:0;">
              <div class="muted small">Watch for in 1 week</div>
              <div id="oneWeekWatch">-</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-head">
            <h3>Recovery gallery</h3>
            <span class="pill recovering">Timeline</span>
          </div>
          <div id="gallery" class="gallery"></div>
        </div>

        <div class="card">
          <div class="section-head">
            <h3>Photo diff</h3>
            <span class="pill">Last vs current</span>
          </div>
          <div id="photoDiff" class="comparison">
            <div class="muted small">Submit a weekly check-in to compare the last two photos with agent notes.</div>
          </div>
        </div>

        <div class="card">
          <div class="section-head">
            <h3>Upcoming reminders</h3>
            <span class="pill diagnosing">Scheduler</span>
          </div>
          <div id="remindersList" class="reminders-list">
            <div class="muted small">Login to see watering and fertilizing nudges.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const state = {
      token: localStorage.getItem('plant_token') || '',
      user: null,
      plants: [],
      selectedPlant: null,
      gallery: [],
      lastCheckInComparison: null,
      communitySymptom: 'all'
    };

    function headers(json = false) {
      const h = {};
      if (state.token) h.Authorization = `Bearer ${state.token}`;
      if (json) h['Content-Type'] = 'application/json';
      return h;
    }

    async function api(path, options = {}) {
      const res = await fetch(path, { ...options, headers: { ...headers(options.json), ...(options.headers || {}) } });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      return res.json();
    }

    function setCommunitySymptom(symptom) {
      state.communitySymptom = symptom;
      loadCommunityGallery();
    }

    function setAuthUI(loggedIn) {
      document.getElementById('authView').classList.toggle('hidden', loggedIn);
      document.getElementById('dashboardView').classList.toggle('hidden', !loggedIn);
      document.getElementById('logoutBtn').classList.toggle('hidden', !loggedIn);
    }

    function setUserBadge(text, cls) {
      const el = document.getElementById('userBadge');
      el.textContent = text;
      el.className = `pill ${cls || 'diagnosing'}`;
    }

    function setStatus(id, message, isError = false) {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = message || '';
      el.style.color = isError ? '#ff9c9c' : '';
    }

    function previewFile(event, targetId) {
      const file = event.target.files && event.target.files[0];
      const img = document.getElementById(targetId);
      if (!file) {
        img.classList.add('hidden');
        img.src = '';
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        img.src = reader.result;
        img.classList.remove('hidden');
      };
      reader.readAsDataURL(file);
    }

    function clearInputAndPreview(inputId, previewId) {
      const input = document.getElementById(inputId);
      const preview = document.getElementById(previewId);
      if (input) input.value = '';
      if (preview) {
        preview.src = '';
        preview.classList.add('hidden');
      }
    }

    async function signup() {
      try {
        setStatus('signupStatus', 'Creating account...');
        const email = document.getElementById('signupEmail').value.trim();
        const password = document.getElementById('signupPassword').value;
        const data = await api('/api/auth/signup', { method: 'POST', json: true, body: JSON.stringify({ email, password }) });
        state.token = data.access_token;
        localStorage.setItem('plant_token', state.token);
        setStatus('signupStatus', 'Account created.');
        await boot();
      } catch (err) {
        setStatus('signupStatus', err.message, true);
      }
    }

    async function login() {
      try {
        setStatus('loginStatus', 'Logging in...');
        const email = document.getElementById('loginEmail').value.trim();
        const password = document.getElementById('loginPassword').value;
        const data = await api('/api/auth/login', { method: 'POST', json: true, body: JSON.stringify({ email, password }) });
        state.token = data.access_token;
        localStorage.setItem('plant_token', state.token);
        setStatus('loginStatus', 'Logged in.');
        await boot();
      } catch (err) {
        setStatus('loginStatus', err.message, true);
      }
    }

    function logout() {
      state.token = '';
      state.user = null;
      state.plants = [];
      state.selectedPlant = null;
      state.lastCheckInComparison = null;
      localStorage.removeItem('plant_token');
      setAuthUI(false);
      setUserBadge('Signed out', 'diagnosing');
      document.getElementById('remindersList').innerHTML = '<div class="muted small">Login to see watering and fertilizing nudges.</div>';
    }

    function renderCommunityGallery(items) {
      const container = document.getElementById('communityGallery');
      if (!container) return;
      if (!items || !items.length) {
        container.innerHTML = '<div class="muted small">No recovered cases yet for this symptom.</div>';
        return;
      }
      container.innerHTML = items.map(item => `
        <div class="community-card">
          <div class="row" style="justify-content:space-between; align-items:center;">
            <strong>${item.title}</strong>
            <span class="pill recovering">${item.symptom || 'general'}</span>
          </div>
          <div class="pair">
            <div>
              <div class="muted small" style="margin-bottom:6px;">Before</div>
              <img src="${item.before_photo_url}" alt="Recovered case before photo" />
            </div>
            <div>
              <div class="muted small" style="margin-bottom:6px;">After</div>
              <img src="${item.after_photo_url}" alt="Recovered case after photo" />
            </div>
          </div>
          <div class="statusline" style="margin-top:0;">${item.recovery_note}</div>
          <div class="muted small">Recovered ${item.recovered_at || 'recently'} · ${item.check_in_count || 0} check-ins</div>
        </div>
      `).join('');
    }

    async function loadCommunityGallery() {
      try {
        const symptom = state.communitySymptom || 'all';
        const data = await api(`/api/public/recovered-gallery?symptom=${encodeURIComponent(symptom)}&limit=6`);
        renderCommunityGallery(data.gallery || []);
      } catch (err) {
        const container = document.getElementById('communityGallery');
        if (container) container.innerHTML = `<div class="muted small">${err.message}</div>`;
      }
    }

    function renderReminders(reminders) {
      const container = document.getElementById('remindersList');
      if (!container) return;
      if (!reminders || !reminders.length) {
        container.innerHTML = '<div class="muted small">No reminders due right now.</div>';
        return;
      }
      container.innerHTML = reminders.map(reminder => `
        <div class="reminder-item">
          <div class="reminder-head">
            <div>
              <strong>${reminder.title}</strong>
              <div class="muted small">Due ${reminder.due_at || 'soon'} · ${reminder.reminder_type}</div>
            </div>
            ${reminder.acknowledged_at ? '<span class="pill healthy">Done</span>' : `<button class="secondary" style="width:auto;" onclick="ackReminder(${reminder.id})">Done</button>`}
          </div>
          <div class="statusline" style="margin-top:0;">${reminder.message}</div>
        </div>
      `).join('');
    }

    async function loadReminders() {
      const container = document.getElementById('remindersList');
      if (!state.token) {
        if (container) container.innerHTML = '<div class="muted small">Login to see watering and fertilizing nudges.</div>';
        return;
      }
      try {
        const data = await api('/api/me/reminders');
        renderReminders(data.reminders || []);
      } catch (err) {
        if (container) container.innerHTML = `<div class="muted small">${err.message}</div>`;
      }
    }

    async function ackReminder(reminderId) {
      await api(`/api/reminders/${reminderId}/ack`, { method: 'POST' });
      await loadReminders();
    }

    function renderPlants() {
      const container = document.getElementById('plantList');
      if (!state.plants.length) {
        container.innerHTML = '<div class="muted small">No plants yet.</div>';
        return;
      }
      container.innerHTML = state.plants.map(plant => {
        const photo = plant.latest_photo_url ? plant.latest_photo_url : '';
        return `
          <div class="plant-card ${state.selectedPlant && state.selectedPlant.id === plant.id ? 'active' : ''}" onclick="selectPlant(${plant.id})">
            ${photo ? `<img src="${photo}" alt="Plant photo" />` : '<div class="preview" style="height:140px"></div>'}
            <div class="row" style="justify-content:space-between;">
              <strong>${plant.nickname || plant.species || 'Plant #' + plant.id}</strong>
              <span class="pill ${plant.status_label || 'diagnosing'}">${plant.status_label || 'diagnosing'}</span>
            </div>
            <div class="muted small">${plant.species || 'Unknown species'}${plant.location ? ' · ' + plant.location : ''}</div>
          </div>`;
      }).join('');
    }

    async function loadPlants(preferredPlantId = null) {
      const refreshBtn = document.getElementById('refreshBtn');
      if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Refreshing...';
      }
      try {
        const data = await api('/api/me/plants');
        state.plants = data.plants || [];
        if (preferredPlantId) {
          state.selectedPlant = state.plants.find(p => p.id === preferredPlantId) || (state.plants.length ? state.plants[0] : null);
        } else if (!state.selectedPlant || !state.plants.some(p => p.id === state.selectedPlant.id)) {
          state.selectedPlant = state.plants.length ? state.plants[0] : null;
        }
        renderPlants();
        if (state.selectedPlant) {
          await selectPlant(state.selectedPlant.id, false);
        } else {
          document.getElementById('detailTitle').textContent = 'Select a plant';
          document.getElementById('detailMeta').textContent = '';
          document.getElementById('detailStatus').textContent = 'diagnosing';
          document.getElementById('detailStatus').className = 'pill diagnosing';
          document.getElementById('detailDescription').textContent = '';
          document.getElementById('comparisonFocus').textContent = 'Upload a weekly photo and one-line update.';
          document.getElementById('carePlanText').textContent = 'No care plan yet.';
          document.getElementById('recoveryTime').textContent = '-';
          document.getElementById('oneWeekWatch').textContent = '-';
          document.getElementById('followupBox').textContent = 'Register a plant to get clarifying questions before treatment.';
          document.getElementById('followupFields').innerHTML = '';
          document.getElementById('followupBtn').classList.add('hidden');
          document.getElementById('careChecklist').innerHTML = '';
          document.getElementById('gallery').innerHTML = '';
          document.getElementById('photoDiff').innerHTML = '<div class="muted small">Submit a weekly check-in to compare the last two photos with agent notes.</div>';
        }
      } finally {
        if (refreshBtn) {
          refreshBtn.disabled = false;
          refreshBtn.textContent = 'Refresh';
        }
      }
    }

    function renderCarePlan(plant) {
      const text = plant.current_care_plan || 'No care plan yet.';
      document.getElementById('carePlanText').textContent = text;
      document.getElementById('recoveryTime').textContent = plant.expected_recovery_time || '-';
      document.getElementById('oneWeekWatch').textContent = plant.one_week_watch_for || '-';
      const checklist = document.getElementById('careChecklist');
      const items = plant.care_plan_items || [];
      if (!items.length) {
        checklist.innerHTML = '<div class="muted small">Checklist appears after follow-up answers are submitted.</div>';
        return;
      }
      checklist.innerHTML = items.map(item => `
        <label class="check-item ${item.done ? 'done' : ''}">
          <input type="checkbox" ${item.done ? 'checked' : ''} onchange="toggleChecklist(${plant.id}, ${item.id}, this.checked)" />
          <div>
            <div>${item.text}</div>
          </div>
        </label>`).join('');
    }

    function renderComparison(plant) {
      const panel = document.getElementById('photoDiff');
      const latest = state.lastCheckInComparison && state.lastCheckInComparison.plant_id === plant.id
        ? state.lastCheckInComparison
        : (plant.latest_check_in || null);
      const previousPhotoUrl = latest ? (latest.previous_photo_url || latest.previous_url || null) : null;
      const currentPhotoUrl = latest ? (latest.current_photo_url || latest.photo_url || null) : null;

      if (!latest || (!previousPhotoUrl && !currentPhotoUrl)) {
        panel.innerHTML = '<div class="muted small">Submit a weekly check-in to compare the last two photos with agent notes.</div>';
        return;
      }

      const notes = latest.overlay_notes || [];
      panel.innerHTML = `
        <div class="comparison-grid">
          <div class="compare-card">
            ${previousPhotoUrl ? `<img src="${previousPhotoUrl}" alt="Previous plant photo" />` : '<div class="preview" style="height:100%; min-height:220px"></div>'}
            <div class="compare-caption"><strong>Previous</strong>Earlier reference photo from the last check-in.</div>
          </div>
          <div class="compare-card">
            ${currentPhotoUrl ? `<img src="${currentPhotoUrl}" alt="Current plant photo" />` : '<div class="preview" style="height:100%; min-height:220px"></div>'}
            <div class="compare-caption"><strong>Current</strong>Newest photo from the agent review.</div>
          </div>
        </div>
        <div class="compare-notes">
          ${(notes.length ? notes : ['No overlay notes were returned for this check-in.']).map(note => `<div class="compare-note">${note}</div>`).join('')}
        </div>
      `;
    }

    function renderQuestions(plant) {
      const box = document.getElementById('followupBox');
      const fields = document.getElementById('followupFields');
      const questions = plant.followup_questions || [];
      if (!questions.length) {
        box.textContent = 'No pending follow-up questions.';
        fields.innerHTML = '';
        document.getElementById('followupBtn').classList.add('hidden');
        return;
      }
      box.textContent = 'Answer these before treatment:';
      fields.innerHTML = questions.map((q, idx) => `
        <div class="question">
          <div class="small muted">Question ${idx + 1}</div>
          <div style="margin-bottom:8px">${q}</div>
          <input data-q="${idx}" placeholder="Your answer" />
        </div>`).join('');
      document.getElementById('followupBtn').classList.remove('hidden');
    }

    async function selectPlant(plantId, reload = true) {
      const plant = state.plants.find(p => p.id === plantId) || await api(`/api/plants/${plantId}`);
      state.selectedPlant = plant;
      if (reload) {
        const full = await api(`/api/plants/${plantId}`);
        state.selectedPlant = full;
      }
      const p = state.selectedPlant;
      document.getElementById('detailTitle').textContent = p.nickname || p.species || `Plant #${p.id}`;
      document.getElementById('detailMeta').textContent = `${p.species || 'Unknown species'}${p.location ? ' · ' + p.location : ''}`;
      document.getElementById('detailStatus').textContent = p.status_label || 'diagnosing';
      document.getElementById('detailStatus').className = `pill ${p.status_label || 'diagnosing'}`;
      document.getElementById('detailDescription').textContent = p.problem_description || '';
      clearInputAndPreview('plantPhoto', 'plantPreview');
      clearInputAndPreview('checkInPhoto', 'checkInPreview');
      document.getElementById('checkInNote').value = '';
      const latest = document.getElementById('detailPhoto');
      if (p.latest_photo_url) {
        latest.src = p.latest_photo_url;
        latest.classList.remove('hidden');
      } else {
        latest.classList.add('hidden');
      }
      document.getElementById('comparisonFocus').textContent = p.one_week_watch_for ? `Check-in focus: ${p.one_week_watch_for}` : 'Upload a weekly photo and one-line update.';
      renderQuestions(p);
      renderCarePlan(p);
      renderComparison(p);
      await loadGallery(plantId);
      renderPlants();
    }

    async function loadGallery(plantId) {
      const data = await api(`/api/plants/${plantId}/photos`);
      state.gallery = data.photos || [];
      const gallery = document.getElementById('gallery');
      if (!state.gallery.length) {
        gallery.innerHTML = '<div class="muted small">No photos yet.</div>';
        return;
      }
      gallery.innerHTML = state.gallery.map(photo => `<img src="${photo.url}" title="${photo.uploaded_at || ''}" alt="Plant photo" />`).join('');
    }

    async function registerPlant() {
      const form = new FormData();
      form.append('problem_description', document.getElementById('plantProblem').value);
      form.append('nickname', document.getElementById('plantNickname').value);
      form.append('species', document.getElementById('plantSpecies').value);
      form.append('location', document.getElementById('plantLocation').value);
      const file = document.getElementById('plantPhoto').files[0];
      if (file) form.append('photo', file);

      const status = document.getElementById('registerStatus');
      const bar = document.getElementById('registerProgress');
      status.textContent = 'Uploading...';
      bar.style.width = '0%';
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/plants/register');
        if (state.token) xhr.setRequestHeader('Authorization', `Bearer ${state.token}`);
        xhr.upload.onprogress = e => { if (e.lengthComputable) bar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`; };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(xhr.responseText || xhr.statusText));
          }
        };
        xhr.onerror = () => reject(new Error('Upload failed'));
        xhr.send(form);
      }).then(async data => {
        status.textContent = 'Plant registered. Answer the follow-up questions below.';
        bar.style.width = '100%';
        await loadPlants(data.plant.id);
        await loadReminders();
      }).catch(err => {
        status.textContent = err.message;
        bar.style.width = '0%';
      });
    }

    async function submitFollowup() {
      const plant = state.selectedPlant;
      if (!plant) return;
      const inputs = document.querySelectorAll('#followupFields input[data-q]');
      const answers = {};
      inputs.forEach((input, idx) => answers[`question_${idx + 1}`] = input.value.trim());
      const data = await api(`/api/plants/${plant.id}/follow-up`, { method: 'POST', json: true, body: JSON.stringify({ answers }) });
      document.getElementById('carePlanText').textContent = data.care_plan;
      document.getElementById('recoveryTime').textContent = data.expected_recovery_time;
      document.getElementById('oneWeekWatch').textContent = data.one_week_watch_for;
      const checklist = document.getElementById('careChecklist');
      checklist.innerHTML = (data.checklist_items || []).map(item => `
        <label class="check-item ${item.done ? 'done' : ''}">
          <input type="checkbox" ${item.done ? 'checked' : ''} onchange="toggleChecklist(${plant.id}, ${item.id}, this.checked)" />
          <div><div>${item.text}</div></div>
        </label>`).join('');
      await loadPlants();
      await loadReminders();
    }

    async function toggleChecklist(plantId, itemId, done) {
      const form = new URLSearchParams();
      form.append('done', done ? 'true' : 'false');
      await api(`/api/plants/${plantId}/checklist/${itemId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: form.toString()
      });
      await loadPlants(plantId);
    }

    async function submitCheckIn() {
      const plant = state.selectedPlant;
      if (!plant) return;
      const form = new FormData();
      const file = document.getElementById('checkInPhoto').files[0];
      if (!file) {
        document.getElementById('checkInStatus').textContent = 'Choose a photo first.';
        return;
      }
      form.append('photo', file);
      form.append('note', document.getElementById('checkInNote').value || '');
      const status = document.getElementById('checkInStatus');
      const bar = document.getElementById('checkInProgress');
      status.textContent = 'Uploading weekly check-in...';
      bar.style.width = '0%';
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `/api/plants/${plant.id}/check-ins`);
        if (state.token) xhr.setRequestHeader('Authorization', `Bearer ${state.token}`);
        xhr.upload.onprogress = e => { if (e.lengthComputable) bar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`; };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(xhr.responseText || xhr.statusText));
          }
        };
        xhr.onerror = () => reject(new Error('Upload failed'));
        xhr.send(form);
      }).then(async data => {
        status.textContent = `${data.comparison_focus}: ${data.comparison_summary}`;
        bar.style.width = '100%';
        state.lastCheckInComparison = data.comparison_media || null;
        await loadPlants(plant.id);
        clearInputAndPreview('checkInPhoto', 'checkInPreview');
        document.getElementById('checkInNote').value = '';
        await loadReminders();
      }).catch(err => {
        status.textContent = err.message;
      });
    }

    async function boot() {
      await loadCommunityGallery();
      if (!state.token) {
        setAuthUI(false);
        setUserBadge('Signed out', 'diagnosing');
        return;
      }
      try {
        state.user = await api('/api/auth/me');
        setAuthUI(true);
        setUserBadge(state.user.email, 'healthy');
        await loadPlants();
        await loadReminders();
      } catch (err) {
        setStatus('signupStatus', err.message, true);
        setStatus('loginStatus', err.message, true);
        logout();
      }
    }

    boot();
  </script>
</body>
</html>
"""


@app.get("/app", response_class=HTMLResponse)
def app_page():
    return HTMLResponse(APP_HTML)
