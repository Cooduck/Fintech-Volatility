"""Local market bars, study windows and the archived announcement replay."""

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import time

import exchange_calendars as xc
import numpy as np
import pandas as pd

from config import BASE_DATA

TECH = ["ret1", "ret2", "vol5", "vol20", "volume_ratio", "rsi14", "spread"]
MARKET = ["qqq_ret1", "qqq_vol5", "qqq_vol20", "rel_vol_ratio"]
SCHEDULE = xc.get_calendar("XNYS", start="2015-01-01", end="2026-11-21").schedule


def fetch(url, headers=None):
    """Fetch a historical source, retrying only transient transport or HTTP failures."""
    request = Request(
        url, headers={"User-Agent": "Mozilla/5.0 university financial research", **(headers or {})}
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in {408, 429} and error.code < 500:
                raise
            if attempt == 3:
                raise
        except (URLError, OSError):
            if attempt == 3:
                raise
        time.sleep(2**attempt)


def announcement():
    """Return the archived release used by the fixed NVDA case study."""
    return {
        "utc": "2026-11-12T13:48:00Z",
        "url": "https://www.globenewswire.com/news-release/2026/11/12/3401635/0/en/"
               "nvidia-announces-ceo-transition.html",
    }


def bars(symbol, interval):
    frame = pd.read_csv(BASE_DATA / f"{symbol}_{interval}.csv", index_col="utc")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def price_panel(grain, symbol="NVDA", continuous=False, *, read_bars=bars, schedule_table=SCHEDULE):
    """Build daily or half-day features and realized labels from completed market bars."""
    if grain not in {"daily", "halfday"}:
        raise ValueError(f"Unsupported price granularity: {grain}")
    daily = read_bars(symbol, "1d").copy()
    daily.index = daily.index.tz_convert("America/New_York").tz_localize(None).normalize()
    hourly = read_bars(symbol, "1h") if grain == "halfday" else None
    first_hour = hourly.index.min() if hourly is not None else None
    three_hours = pd.Timedelta(hours=3)
    rows = []
    for i, date in enumerate(schedule_table.index):
        if i < 2 or date not in daily.index or schedule_table.index[i - 1] not in daily.index:
            continue
        schedule = schedule_table.iloc[i]
        previous = schedule_table.iloc[i - 1]
        previous_close = daily.loc[schedule_table.index[i - 1], "close"]
        if grain == "daily":
            values = daily.loc[date]
            volatility = abs(np.log(values["close"] / previous_close))
            windows = [
                ("D", schedule_table.iloc[i - 2]["close"], previous["close"], schedule["close"],
                 values, volatility, previous_close)
            ]
        else:
            if schedule.open < first_hour:
                continue
            windows = []
            split = schedule.open + three_hours
            morning_start = previous.open + three_hours if continuous else previous["close"]
            for session, start, end, news_start in [
                ("AM", schedule.open, split, morning_start),
                ("PM", split, schedule["close"], schedule.open),
            ]:
                period = hourly.reindex(pd.date_range(start, end, freq="1h", inclusive="left"))
                if period.empty or period.isna().any().any():
                    continue
                opening = period.open.iloc[0]
                prior = previous_close if session == "AM" else opening
                changes = np.diff(np.log(np.r_[opening, period.close]))
                overnight = np.log(opening / previous_close) ** 2 if session == "AM" else 0
                volatility = np.sqrt(np.sum(changes ** 2) + overnight)
                values = dict(
                    intraday_rv=np.sqrt(np.sum(changes ** 2)), open=opening,
                    close=period.close.iloc[-1], high=period.high.max(),
                    low=period.low.min(), volume=period.volume.sum(),
                )
                windows.append((session, news_start, start, end, values, volatility, prior))
        for session, start, cut, end, values, volatility, prior in windows:
            spread = np.log(values["high"] / values["low"])
            range_volatility = spread / np.sqrt(4 * np.log(2))
            rows.append(dict(
                date=str(date.date()), session=session, start=start, cut=cut, end=end,
                sigma_intraday=max(values.get("intraday_rv", volatility), range_volatility),
                sigma=max(volatility, range_volatility), ret=np.log(values["close"] / prior),
                volume=values["volume"], range_now=spread, previous_close=previous_close,
            ))

    if not rows:
        raise ValueError(f"No complete {grain} market windows for {symbol}.")
    frame = pd.DataFrame(rows)
    # Predictors are lagged so the current window's realized outcomes cannot enter a decision.
    frame["ret1"] = frame.ret.shift()
    frame["ret2"] = frame.ret.shift(2)
    frame["spread"] = frame.range_now.shift()
    session = frame.groupby("session")
    for lag in (5, 20):
        frame[f"vol{lag}"] = session.ret.transform(lambda s: s.shift().rolling(lag).std())
    if grain == "halfday":
        reference = session.volume.transform(lambda s: s.shift().rolling(20).mean())
        frame["volume_ratio"] = (frame.volume / reference).shift()
    else:
        frame["volume_ratio"] = frame.volume.shift() / frame.volume.shift(2).rolling(20).mean()
    gains = frame.ret.clip(lower=0).shift().rolling(14).mean()
    losses = (-frame.ret.clip(upper=0)).shift().rolling(14).mean()
    frame["rsi14"] = 100 * gains / (gains + losses).replace(0, np.nan)
    frame["past_mean"] = session.sigma.transform(lambda s: s.shift().rolling(20).mean())
    deviation = session.sigma.transform(lambda s: s.shift().rolling(20).std())
    frame["z"] = (frame.sigma - frame.past_mean) / deviation.replace(0, np.nan)
    frame["is_afternoon_session"] = frame.session.eq("PM").astype(int)
    previous_intraday = session.sigma_intraday
    mean = previous_intraday.transform(lambda s: s.shift().rolling(20).mean())
    deviation = previous_intraday.transform(lambda s: s.shift().rolling(20).std())
    frame["z_intraday"] = (frame.sigma_intraday - mean) / deviation.replace(0, np.nan)
    frame["sigma_lag1"] = frame.sigma.shift()
    frame["market_known_at"] = frame.end.shift()
    return frame
