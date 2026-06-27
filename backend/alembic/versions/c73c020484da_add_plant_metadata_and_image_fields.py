"""add_plant_metadata_and_image_fields

Revision ID: c73c020484da
Revises: 866ffaf45e9a
Create Date: 2026-06-16 10:44:36.313713

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c73c020484da'
down_revision: Union[str, Sequence[str], None] = '866ffaf45e9a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("plants", sa.Column("nickname", sa.String(), nullable=True))
    op.add_column("plants", sa.Column("problem_description", sa.Text(), nullable=True))
    op.add_column("plants", sa.Column("issue_category", sa.String(), nullable=True))
    op.add_column("plants", sa.Column("diagnosis_status", sa.String(), nullable=True))
    op.add_column("plants", sa.Column("current_care_plan", sa.Text(), nullable=True))
    op.add_column("plants", sa.Column("expected_recovery_time", sa.String(), nullable=True))
    op.add_column("plants", sa.Column("one_week_watch_for", sa.Text(), nullable=True))
    op.add_column("plants", sa.Column("followup_questions", sa.JSON(), nullable=True))
    op.add_column("plants", sa.Column("followup_answers", sa.JSON(), nullable=True))
    op.add_column("plants", sa.Column("next_check_in_at", sa.DateTime(), nullable=True))
    op.add_column("plants", sa.Column("created_at", sa.DateTime(), nullable=True))
    op.add_column("plants", sa.Column("updated_at", sa.DateTime(), nullable=True))

    op.add_column("diagnoses", sa.Column("species", sa.String(), nullable=True))
    op.add_column("diagnoses", sa.Column("followup_questions", sa.JSON(), nullable=True))
    op.add_column("diagnoses", sa.Column("followup_answers", sa.JSON(), nullable=True))
    op.add_column("diagnoses", sa.Column("expected_recovery_time", sa.String(), nullable=True))
    op.add_column("diagnoses", sa.Column("one_week_watch_for", sa.Text(), nullable=True))
    op.add_column("diagnoses", sa.Column("updated_at", sa.DateTime(), nullable=True))

    op.create_table(
        "check_ins",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plant_id", sa.Integer(), nullable=False),
        sa.Column("photo_id", sa.Integer(), nullable=True),
        sa.Column("previous_photo_id", sa.Integer(), nullable=True),
        sa.Column("comparison_summary", sa.Text(), nullable=False),
        sa.Column("plan_update", sa.Text(), nullable=False),
        sa.Column("health_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["plant_id"], ["plants.id"]),
        sa.ForeignKeyConstraint(["photo_id"], ["photos.id"]),
        sa.ForeignKeyConstraint(["previous_photo_id"], ["photos.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("check_ins")

    op.drop_column("diagnoses", "updated_at")
    op.drop_column("diagnoses", "one_week_watch_for")
    op.drop_column("diagnoses", "expected_recovery_time")
    op.drop_column("diagnoses", "followup_answers")
    op.drop_column("diagnoses", "followup_questions")
    op.drop_column("diagnoses", "species")

    op.drop_column("plants", "updated_at")
    op.drop_column("plants", "created_at")
    op.drop_column("plants", "next_check_in_at")
    op.drop_column("plants", "followup_answers")
    op.drop_column("plants", "followup_questions")
    op.drop_column("plants", "one_week_watch_for")
    op.drop_column("plants", "expected_recovery_time")
    op.drop_column("plants", "current_care_plan")
    op.drop_column("plants", "diagnosis_status")
    op.drop_column("plants", "issue_category")
    op.drop_column("plants", "problem_description")
    op.drop_column("plants", "nickname")
