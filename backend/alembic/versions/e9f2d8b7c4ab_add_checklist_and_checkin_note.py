"""add_checklist_and_checkin_note

Revision ID: e9f2d8b7c4ab
Revises: c73c020484da
Create Date: 2026-06-16 11:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e9f2d8b7c4ab"
down_revision: Union[str, Sequence[str], None] = "c73c020484da"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("plants", sa.Column("care_plan_items", sa.JSON(), nullable=True))
    op.add_column("check_ins", sa.Column("user_note", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("check_ins", "user_note")
    op.drop_column("plants", "care_plan_items")
