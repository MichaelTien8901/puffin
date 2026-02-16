"""Trading session management and market calendar."""

import asyncio
import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)


class TradingSession:
    """Manages trading session timing and market calendar."""

    # NYSE regular trading hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    # Extended hours
    PREMARKET_OPEN = time(4, 0)
    AFTERHOURS_CLOSE = time(20, 0)

    # Market holidays (major US holidays)
    # These are approximate - use a proper calendar API for production
    MARKET_HOLIDAYS = [
        # 2024
        "2024-01-01",  # New Year's Day
        "2024-01-15",  # MLK Day
        "2024-02-19",  # Presidents Day
        "2024-03-29",  # Good Friday
        "2024-05-27",  # Memorial Day
        "2024-06-19",  # Juneteenth
        "2024-07-04",  # Independence Day
        "2024-09-02",  # Labor Day
        "2024-11-28",  # Thanksgiving
        "2024-12-25",  # Christmas
        # 2025
        "2025-01-01",
        "2025-01-20",
        "2025-02-17",
        "2025-04-18",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-12-25",
        # 2026
        "2026-01-01",
        "2026-01-19",
        "2026-02-16",
        "2026-04-03",
        "2026-05-25",
        "2026-06-19",
        "2026-07-03",
        "2026-09-07",
        "2026-11-26",
        "2026-12-25",
    ]

    def __init__(self, timezone: str = "America/New_York", extended_hours: bool = False):
        """Initialize trading session.

        Args:
            timezone: Timezone for market hours (default Eastern Time).
            extended_hours: Include pre-market and after-hours (default False).
        """
        self.timezone = ZoneInfo(timezone)
        self.extended_hours = extended_hours
        self._holidays = set(self.MARKET_HOLIDAYS)

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Check if market is currently open.

        Args:
            dt: Datetime to check (default now).

        Returns:
            True if market is open, False otherwise.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone)
        else:
            dt = dt.astimezone(self.timezone)

        # Check if weekend
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check if holiday
        if dt.date().isoformat() in self._holidays:
            return False

        # Check time
        current_time = dt.time()

        if self.extended_hours:
            return self.PREMARKET_OPEN <= current_time <= self.AFTERHOURS_CLOSE
        else:
            return self.MARKET_OPEN <= current_time < self.MARKET_CLOSE

    def next_open(self, dt: datetime | None = None) -> datetime:
        """Get next market open time.

        Args:
            dt: Reference datetime (default now).

        Returns:
            Datetime of next market open.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone)
        else:
            dt = dt.astimezone(self.timezone)

        # Start from current date
        check_date = dt.date()
        open_time = self.PREMARKET_OPEN if self.extended_hours else self.MARKET_OPEN

        # If today and before open time, return today's open
        if dt.time() < open_time and check_date.weekday() < 5 and check_date.isoformat() not in self._holidays:
            return datetime.combine(check_date, open_time, tzinfo=self.timezone)

        # Otherwise, find next trading day
        for _ in range(10):  # Check up to 10 days ahead
            check_date += timedelta(days=1)

            # Skip weekends
            if check_date.weekday() >= 5:
                continue

            # Skip holidays
            if check_date.isoformat() in self._holidays:
                continue

            # Found next trading day
            return datetime.combine(check_date, open_time, tzinfo=self.timezone)

        # Fallback (shouldn't reach here)
        raise ValueError("Could not find next market open within 10 days")

    def next_close(self, dt: datetime | None = None) -> datetime:
        """Get next market close time.

        Args:
            dt: Reference datetime (default now).

        Returns:
            Datetime of next market close.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone)
        else:
            dt = dt.astimezone(self.timezone)

        close_time = self.AFTERHOURS_CLOSE if self.extended_hours else self.MARKET_CLOSE

        # If market is open today, return today's close
        if self.is_market_open(dt):
            return datetime.combine(dt.date(), close_time, tzinfo=self.timezone)

        # Otherwise, return next market day's close
        next_open_dt = self.next_open(dt)
        return datetime.combine(next_open_dt.date(), close_time, tzinfo=self.timezone)

    def supports_extended_hours(self) -> bool:
        """Check if extended hours are supported.

        Returns:
            True if extended hours are enabled.
        """
        return self.extended_hours

    async def wait_for_open(self, check_interval: float = 60.0):
        """Async wait until market opens.

        Args:
            check_interval: Seconds between checks (default 60).
        """
        while not self.is_market_open():
            next_open_dt = self.next_open()
            now = datetime.now(self.timezone)
            wait_seconds = (next_open_dt - now).total_seconds()

            logger.info(f"Market closed. Next open: {next_open_dt} ({wait_seconds/3600:.1f} hours)")

            # Wait either check_interval or until next open, whichever is shorter
            await asyncio.sleep(min(check_interval, max(0, wait_seconds)))

        logger.info("Market is now open")

    def wait_for_open_sync(self, check_interval: float = 60.0):
        """Synchronous wait until market opens.

        Args:
            check_interval: Seconds between checks (default 60).
        """
        import time as time_module

        while not self.is_market_open():
            next_open_dt = self.next_open()
            now = datetime.now(self.timezone)
            wait_seconds = (next_open_dt - now).total_seconds()

            logger.info(f"Market closed. Next open: {next_open_dt} ({wait_seconds/3600:.1f} hours)")

            # Wait either check_interval or until next open, whichever is shorter
            time_module.sleep(min(check_interval, max(0, wait_seconds)))

        logger.info("Market is now open")

    def get_trading_days(self, start: datetime, end: datetime) -> list[datetime]:
        """Get all trading days between start and end dates.

        Args:
            start: Start date.
            end: End date.

        Returns:
            List of trading day datetimes.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=self.timezone)
        if end.tzinfo is None:
            end = end.replace(tzinfo=self.timezone)

        trading_days = []
        current = start.date()
        end_date = end.date()

        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                # Skip holidays
                if current.isoformat() not in self._holidays:
                    trading_days.append(
                        datetime.combine(current, self.MARKET_OPEN, tzinfo=self.timezone)
                    )

            current += timedelta(days=1)

        return trading_days

    def is_trading_day(self, dt: datetime | None = None) -> bool:
        """Check if given date is a trading day.

        Args:
            dt: Date to check (default today).

        Returns:
            True if trading day, False otherwise.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone)

        # Check weekend
        if dt.weekday() >= 5:
            return False

        # Check holiday
        if dt.date().isoformat() in self._holidays:
            return False

        return True

    def time_until_open(self, dt: datetime | None = None) -> timedelta:
        """Get time until next market open.

        Args:
            dt: Reference datetime (default now).

        Returns:
            Timedelta until next open.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone)

        next_open_dt = self.next_open(dt)
        return next_open_dt - dt

    def time_until_close(self, dt: datetime | None = None) -> timedelta:
        """Get time until next market close.

        Args:
            dt: Reference datetime (default now).

        Returns:
            Timedelta until next close.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone)

        next_close_dt = self.next_close(dt)
        return next_close_dt - dt

    def get_session_schedule(self, date: datetime | None = None) -> dict:
        """Get trading session schedule for a given date.

        Args:
            date: Date to get schedule for (default today).

        Returns:
            Dict with open/close times, or empty if not a trading day.
        """
        if date is None:
            date = datetime.now(self.timezone)
        elif date.tzinfo is None:
            date = date.replace(tzinfo=self.timezone)

        if not self.is_trading_day(date):
            return {"trading_day": False}

        schedule = {
            "trading_day": True,
            "date": date.date().isoformat(),
            "regular_open": datetime.combine(date.date(), self.MARKET_OPEN, tzinfo=self.timezone),
            "regular_close": datetime.combine(date.date(), self.MARKET_CLOSE, tzinfo=self.timezone),
        }

        if self.extended_hours:
            schedule["premarket_open"] = datetime.combine(
                date.date(), self.PREMARKET_OPEN, tzinfo=self.timezone
            )
            schedule["afterhours_close"] = datetime.combine(
                date.date(), self.AFTERHOURS_CLOSE, tzinfo=self.timezone
            )

        return schedule
