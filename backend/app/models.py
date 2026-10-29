
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, UniqueConstraint
)

from app.database import Base


class PriceBar(Base):
    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "interval", "timestamp",
            name="uq_price_bar",
        ),
    )

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), index=True, nullable=False)
    interval = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class ForecastRecord(Base):
    __tablename__ = "forecast_records"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), index=True, nullable=False)
    interval = Column(String(10), nullable=False)
    model_name = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    forecast_timestamp = Column(DateTime(timezone=True), nullable=False)
    predicted_close = Column(Float, nullable=False)
