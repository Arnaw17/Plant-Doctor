from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    plants = relationship("Plant", back_populates="owner")


class Plant(Base):
    __tablename__ = "plants"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    nickname = Column(String, nullable=True)
    species = Column(String, nullable=True)
    location = Column(String, nullable=True)
    problem_description = Column(Text, nullable=True)
    issue_category = Column(String, nullable=True)
    diagnosis_status = Column(String, nullable=True, default="pending_followup")
    current_care_plan = Column(Text, nullable=True)
    care_plan_items = Column(JSON, nullable=True)
    expected_recovery_time = Column(String, nullable=True)
    one_week_watch_for = Column(Text, nullable=True)
    followup_questions = Column(JSON, nullable=True)
    followup_answers = Column(JSON, nullable=True)
    last_watered = Column(DateTime, default=func.now())
    next_watering_at = Column(DateTime, nullable=True)
    next_fertilizing_at = Column(DateTime, nullable=True)
    next_check_in_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    owner = relationship("User", back_populates="plants")
    photos = relationship("Photo", back_populates="plant")
    diagnoses = relationship("Diagnosis", back_populates="plant")
    check_ins = relationship("CheckIn", back_populates="plant")


class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"), nullable=False)
    url = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=func.now())

    plant = relationship("Plant", back_populates="photos")


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"), nullable=False)
    diagnosis_result = Column(Text, nullable=False)
    issue_category = Column(String, nullable=True)
    species = Column(String, nullable=True)
    followup_questions = Column(JSON, nullable=True)
    followup_answers = Column(JSON, nullable=True)
    care_plan = Column(Text, nullable=True)
    expected_recovery_time = Column(String, nullable=True)
    one_week_watch_for = Column(Text, nullable=True)
    status = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    plant = relationship("Plant", back_populates="diagnoses")


class CheckIn(Base):
    __tablename__ = "check_ins"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"), nullable=False)
    photo_id = Column(Integer, ForeignKey("photos.id"), nullable=True)
    previous_photo_id = Column(Integer, ForeignKey("photos.id"), nullable=True)
    user_note = Column(Text, nullable=True)
    comparison_summary = Column(Text, nullable=False)
    plan_update = Column(Text, nullable=False)
    health_status = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())

    plant = relationship("Plant", back_populates="check_ins")
    photo = relationship("Photo", foreign_keys=[photo_id])


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reminder_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    due_at = Column(DateTime, nullable=False)
    sent_at = Column(DateTime, default=func.now())
    acknowledged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    plant = relationship("Plant")
    user = relationship("User")