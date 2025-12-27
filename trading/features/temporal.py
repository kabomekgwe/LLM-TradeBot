"""Time-based features using pandas_market_calendars.

Extracts trading session, day-of-week, hour-of-day, weekend flags,
and holiday detection from timestamp data.
"""

import logging
import pandas as pd
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)


class TemporalFeatures:
    """Extract time-based features from timestamps."""

    def __init__(self, calendar_name: str = 'NYSE'):
        """Initialize temporal feature extractor.

        Args:
            calendar_name: Market calendar to use (default: NYSE)
                          Options: NYSE, NASDAQ, LSE, TSX, BMF, HKEX, etc.
        """
        try:
            self.calendar = mcal.get_calendar(calendar_name)
            self.calendar_name = calendar_name
            logger.info(f"Initialized {calendar_name} market calendar")
        except Exception as e:
            logger.warning(f"Could not load {calendar_name} calendar: {e}. Using fallback.")
            self.calendar = None

    @staticmethod
    def get_trading_session(hour_utc: int) -> str:
        """Map UTC hour to major trading session.

        Args:
            hour_utc: Hour in UTC (0-23)

        Returns:
            Session name: 'Asia', 'London', 'New_York', or 'Off_Hours'
        """
        if 0 <= hour_utc < 8:
            return 'Asia'  # Tokyo open 0:00-8:00 UTC
        elif 8 <= hour_utc < 12:
            return 'London'  # London open 8:00-16:00 UTC (pre-NY)
        elif 12 <= hour_utc < 21:
            return 'New_York'  # NY overlap + exclusive 12:00-21:00 UTC
        else:
            return 'Off_Hours'  # 21:00-24:00 UTC

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to DataFrame.

        Args:
            df: DataFrame with 'timestamp' column (datetime type)

        Returns:
            DataFrame with added temporal features

        Raises:
            ValueError: If 'timestamp' column missing or not datetime
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logger.warning("Converting 'timestamp' to datetime")
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter

        # Weekend flag
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Trading session
        df['session'] = df['hour'].apply(self.get_trading_session)

        # One-hot encode sessions
        df['session_asia'] = (df['session'] == 'Asia').astype(int)
        df['session_london'] = (df['session'] == 'London').astype(int)
        df['session_newyork'] = (df['session'] == 'New_York').astype(int)
        df['session_offhours'] = (df['session'] == 'Off_Hours').astype(int)

        # Holiday detection (if calendar available)
        if self.calendar is not None:
            try:
                # Get trading schedule for date range
                start_date = df['timestamp'].min().date()
                end_date = df['timestamp'].max().date()
                schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

                # Mark holidays (dates NOT in schedule)
                df['is_holiday'] = (~df['timestamp'].dt.date.isin(schedule.index.date)).astype(int)

                logger.debug(f"Detected {df['is_holiday'].sum()} holiday candles")
            except Exception as e:
                logger.warning(f"Could not detect holidays: {e}")
                df['is_holiday'] = 0
        else:
            df['is_holiday'] = 0

        logger.info(f"Added temporal features to {len(df)} candles")

        return df
