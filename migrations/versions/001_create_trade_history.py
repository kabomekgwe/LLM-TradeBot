"""Create trade_history table and TimescaleDB hypertable

Revision ID: 001
Revises:
Create Date: 2025-12-28

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create trade_history table and convert to TimescaleDB hypertable."""

    # Create trade_history table
    op.create_table(
        'trade_history',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('trade_id', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('side', sa.String(length=10), nullable=False),
        sa.Column('order_type', sa.String(length=10), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('exit_price', sa.Float(), nullable=True),
        sa.Column('realized_pnl', sa.Float(), server_default='0.0'),
        sa.Column('pnl_pct', sa.Float(), server_default='0.0'),
        sa.Column('fees', sa.Float(), server_default='0.0'),
        sa.Column('market_regime', sa.String(length=20), nullable=True),
        sa.Column('bull_confidence', sa.Float(), nullable=True),
        sa.Column('bear_confidence', sa.Float(), nullable=True),
        sa.Column('decision_confidence', sa.Float(), nullable=True),
        sa.Column('won', sa.Boolean(), server_default='false'),
        sa.Column('closed', sa.Boolean(), server_default='false'),
        sa.Column('close_timestamp', sa.DateTime(), nullable=True),
        sa.Column('agent_votes', JSON, nullable=True),
        sa.Column('signals', JSON, nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_trade_history_trade_id', 'trade_history', ['trade_id'], unique=True)
    op.create_index('ix_trade_history_symbol', 'trade_history', ['symbol'], unique=False)
    op.create_index('ix_trade_history_timestamp', 'trade_history', ['timestamp'], unique=False)

    # Convert to TimescaleDB hypertable
    # This enables time-series optimizations for timestamp-based queries
    op.execute("""
        SELECT create_hypertable('trade_history', 'timestamp', if_not_exists => TRUE);
    """)


def downgrade() -> None:
    """Drop trade_history table."""
    op.drop_index('ix_trade_history_timestamp', table_name='trade_history')
    op.drop_index('ix_trade_history_symbol', table_name='trade_history')
    op.drop_index('ix_trade_history_trade_id', table_name='trade_history')
    op.drop_table('trade_history')
