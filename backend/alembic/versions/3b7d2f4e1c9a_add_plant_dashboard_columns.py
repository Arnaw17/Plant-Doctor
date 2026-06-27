"""add_plant_dashboard_columns

Revision ID: 3b7d2f4e1c9a
Revises: e9f2d8b7c4ab
Create Date: 2026-06-16 11:26:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "3b7d2f4e1c9a"
down_revision: Union[str, Sequence[str], None] = "e9f2d8b7c4ab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
