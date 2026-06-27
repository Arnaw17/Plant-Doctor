"""add_reminders_and_due_dates

Revision ID: 6e7f8a9b0c1d
Revises: 3b7d2f4e1c9a
Create Date: 2026-06-22 12:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "6e7f8a9b0c1d"
down_revision: Union[str, Sequence[str], None] = "3b7d2f4e1c9a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("plants", sa.Column("next_watering_at", sa.DateTime(), nullable=True))
    op.add_column("plants", sa.Column("next_fertilizing_at", sa.DateTime(), nullable=True))
    op.create_table(
        "reminders",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("plant_id", sa.Integer(), sa.ForeignKey("plants.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("reminder_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("due_at", sa.DateTime(), nullable=False),
        sa.Column("sent_at", sa.DateTime(), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(op.f("ix_reminders_id"), "reminders", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_reminders_id"), table_name="reminders")
    op.drop_table("reminders")
    op.drop_column("plants", "next_fertilizing_at")
    op.drop_column("plants", "next_watering_at")
