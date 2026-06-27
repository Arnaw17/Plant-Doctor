from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.models import CheckIn, Diagnosis, Photo, Plant, Reminder


scheduler = BackgroundScheduler(timezone="UTC")

ACTIVE_STATUSES = {"active", "monitoring", "healthy", "stable", "recovering"}
RECOVERED_STATUSES = {"healthy", "stable", "recovered"}
WATER_INTERVAL_DAYS = {
    "water": 3,
    "light": 5,
    "pest": 7,
    "nutrient": 4,
    "disease": 4,
}
FERTILIZE_INTERVAL_DAYS = {
    "water": 30,
    "light": 28,
    "pest": 30,
    "nutrient": 14,
    "disease": 21,
}


def _now() -> datetime:
    return datetime.utcnow()


def _safe_text(value: Optional[str]) -> str:
    return (value or "").strip()


def _photo_urls(db: Session, plant_id: int) -> List[str]:
    photos = (
        db.query(Photo)
        .filter(Photo.plant_id == plant_id)
        .order_by(Photo.id.asc())
        .all()
    )
    return [photo.url for photo in photos]


def _latest_check_in(db: Session, plant_id: int) -> Optional[CheckIn]:
    return (
        db.query(CheckIn)
        .filter(CheckIn.plant_id == plant_id)
        .order_by(CheckIn.id.desc())
        .first()
    )


def _watering_due_at(plant: Plant) -> Optional[datetime]:
    if plant.next_watering_at:
        return plant.next_watering_at
    base = plant.last_watered or plant.updated_at or plant.created_at
    if not base:
        return None
    days = WATER_INTERVAL_DAYS.get((plant.issue_category or "water").lower(), 7)
    return base + timedelta(days=days)


def _fertilizing_due_at(plant: Plant) -> Optional[datetime]:
    if plant.next_fertilizing_at:
        return plant.next_fertilizing_at
    care_plan = _safe_text(plant.current_care_plan).lower()
    if plant.issue_category != "nutrient" and "feed" not in care_plan and "fertiliz" not in care_plan:
        return None
    base = plant.updated_at or plant.created_at
    if not base:
        return None
    days = FERTILIZE_INTERVAL_DAYS.get((plant.issue_category or "nutrient").lower(), 21)
    return base + timedelta(days=days)


def _create_reminder(db: Session, plant: Plant, reminder_type: str, due_at: datetime, title: str, message: str) -> bool:
    recent_reminder = (
        db.query(Reminder)
        .filter(
            Reminder.plant_id == plant.id,
            Reminder.reminder_type == reminder_type,
            Reminder.created_at >= due_at - timedelta(hours=12),
        )
        .order_by(Reminder.id.desc())
        .first()
    )
    if recent_reminder:
        return False

    reminder = Reminder(
        plant_id=plant.id,
        user_id=plant.user_id,
        reminder_type=reminder_type,
        title=title,
        message=message,
        due_at=due_at,
        sent_at=_now(),
    )
    db.add(reminder)

    interval_days = WATER_INTERVAL_DAYS.get((plant.issue_category or "water").lower(), 7)
    if reminder_type == "watering":
        plant.next_watering_at = due_at + timedelta(days=interval_days)
    elif reminder_type == "fertilizing":
        interval_days = FERTILIZE_INTERVAL_DAYS.get((plant.issue_category or "nutrient").lower(), 21)
        plant.next_fertilizing_at = due_at + timedelta(days=interval_days)

    return True


def scan_due_care_reminders() -> Dict[str, int]:
    db = SessionLocal()
    created = 0
    touched = 0
    try:
        now = _now()
        plants = db.query(Plant).filter(Plant.diagnosis_status.in_(ACTIVE_STATUSES)).all()
        for plant in plants:
            watering_due = _watering_due_at(plant)
            if watering_due and watering_due <= now:
                title = "Watering reminder"
                message = f"Water {plant.nickname or plant.species or f'Plant #{plant.id}'} and check that the pot drains well."
                if _create_reminder(db, plant, "watering", watering_due, title, message):
                    created += 1
                    touched += 1

            fertilizing_due = _fertilizing_due_at(plant)
            if fertilizing_due and fertilizing_due <= now:
                title = "Fertilizing reminder"
                message = f"Feed {plant.nickname or plant.species or f'Plant #{plant.id}'} at the current plan strength if it is stable enough."
                if _create_reminder(db, plant, "fertilizing", fertilizing_due, title, message):
                    created += 1
                    touched += 1

        if touched:
            db.commit()
        return {"scanned": len(plants), "created": created}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def start_reminder_scheduler() -> None:
    if scheduler.running:
        return
    scheduler.add_job(
        scan_due_care_reminders,
        trigger="interval",
        minutes=15,
        id="plant-care-reminder-scan",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()


def stop_reminder_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)


def serialize_reminder(reminder: Reminder) -> Dict[str, object]:
    return {
        "id": reminder.id,
        "plant_id": reminder.plant_id,
        "reminder_type": reminder.reminder_type,
        "title": reminder.title,
        "message": reminder.message,
        "due_at": reminder.due_at.strftime("%Y-%m-%d %H:%M:%S") if reminder.due_at else None,
        "sent_at": reminder.sent_at.strftime("%Y-%m-%d %H:%M:%S") if reminder.sent_at else None,
        "acknowledged_at": reminder.acknowledged_at.strftime("%Y-%m-%d %H:%M:%S") if reminder.acknowledged_at else None,
        "created_at": reminder.created_at.strftime("%Y-%m-%d %H:%M:%S") if reminder.created_at else None,
    }


def list_user_reminders(db: Session, user_id: int) -> List[Dict[str, object]]:
    reminders = (
        db.query(Reminder)
        .filter(Reminder.user_id == user_id)
        .order_by(Reminder.due_at.desc(), Reminder.id.desc())
        .all()
    )
    return [serialize_reminder(reminder) for reminder in reminders]


def acknowledge_reminder(db: Session, reminder_id: int, user_id: int) -> Optional[Dict[str, object]]:
    reminder = (
        db.query(Reminder)
        .filter(Reminder.id == reminder_id, Reminder.user_id == user_id)
        .first()
    )
    if not reminder:
        return None
    reminder.acknowledged_at = _now()
    db.commit()
    db.refresh(reminder)
    return serialize_reminder(reminder)


def build_public_recovered_gallery(db: Session, symptom: Optional[str] = None, limit: int = 12) -> List[Dict[str, object]]:
    plants = db.query(Plant).order_by(Plant.id.desc()).all()
    symptom_filter = _safe_text(symptom).lower()
    gallery: List[Dict[str, object]] = []

    for plant in plants:
        latest_check_in = _latest_check_in(db, plant.id)
        diagnosis = (
            db.query(Diagnosis)
            .filter(Diagnosis.plant_id == plant.id)
            .order_by(Diagnosis.id.desc())
            .first()
        )
        if plant.diagnosis_status not in RECOVERED_STATUSES and (
            not latest_check_in or (latest_check_in.health_status or "").lower() not in RECOVERED_STATUSES
        ):
            continue

        issue_category = (plant.issue_category or (diagnosis.issue_category if diagnosis else "") or "").lower()
        if symptom_filter and symptom_filter not in issue_category and symptom_filter != "all":
            continue

        urls = _photo_urls(db, plant.id)
        if len(urls) < 2:
            continue

        gallery.append(
            {
                "case_id": plant.id,
                "title": f"Recovered case #{plant.id}",
                "species": plant.species or "Unknown",
                "symptom": plant.issue_category or "general",
                "before_photo_url": urls[-2],
                "after_photo_url": urls[-1],
                "recovery_note": (latest_check_in.comparison_summary if latest_check_in else plant.problem_description or "Recovered through the care plan."),
                "recovery_status": latest_check_in.health_status if latest_check_in and latest_check_in.health_status else plant.diagnosis_status,
                "recovered_at": latest_check_in.created_at.strftime("%Y-%m-%d %H:%M:%S") if latest_check_in and latest_check_in.created_at else None,
                "check_in_count": len(plant.check_ins),
            }
        )
        if len(gallery) >= limit:
            break

    return gallery