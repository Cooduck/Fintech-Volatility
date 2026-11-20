"""V4 point-in-time market features and local event clustering."""

from collections import deque

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from collect_data import DATA as BASE_DATA
from panels import SCHEDULE, bars, price_panel
from risk_evaluation import MARKET
from v3_features import DECAY, aggregate
from v4_collect import DATA

BASE = ["is_earnings_window", "is_earnings_pre1d", "vix_close", "vix_ret1", "vix_z20"]
CLUSTERS = ["UniqueEventCount", "ClusterImpactSum", "NoveltyRatio"]


def price_reader(symbol, interval):
    path = DATA / f"{symbol}_{interval}.csv"
    if not path.exists():
        return bars(symbol, interval)
    data = pd.read_csv(path, index_col="utc")
    data.index = pd.to_datetime(data.index, utc=True)
    return data.sort_index()


def market_frame(
    symbol, reader=price_reader, schedule=SCHEDULE, forecast_date=None, qqq=None
):
    def panel(ticker):
        return price_panel(
            "halfday",
            ticker,
            continuous=True,
            read_bars=reader,
            schedule_table=schedule,
            forecast_date=forecast_date,
        )

    data = panel(symbol)
    qqq = panel("QQQ") if qqq is None else qqq
    names = ["ret1", "vol5", "vol20", "sigma_lag1", "market_known_at"]
    ref = qqq[["cut", "session", *names]].rename(columns={k: f"qqq_{k}" for k in names})
    frame = data.merge(ref, on=["cut", "session"], validate="one_to_one")
    frame["rel_vol_ratio"] = frame.sigma_lag1 / frame.qqq_sigma_lag1.replace(0, np.nan)
    frame["market_known_at"] = frame[["market_known_at", "qqq_market_known_at"]].max(
        axis=1
    )
    frame[MARKET] = frame[MARKET].round(7).astype("float32")
    return frame.assign(symbol=symbol)


def vix_features(vix, schedule):
    vix = vix.sort_values("DATE").copy()
    vix["date_vix"] = vix.DATE.astype(str)
    vix["vix_close"] = vix.CLOSE
    vix["vix_ret1"] = np.log(vix.CLOSE / vix.CLOSE.shift())
    history = vix.CLOSE.shift().rolling(20)
    vix["vix_z20"] = (vix.CLOSE - history.mean()) / history.std().replace(0, np.nan)
    # VIX daily file may include trading after the NYSE equity close.
    # Use the latest completed prior session, with 16:15 ET availability.
    vix["vix_known_at"] = (
        pd.to_datetime(vix.DATE + " 16:15")
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    return vix[["date_vix", "vix_known_at", "vix_close", "vix_ret1", "vix_z20"]]


def add_market_context(frame, symbol, vix, releases, notices, schedule=SCHEDULE):
    sessions = schedule.index.strftime("%Y-%m-%d")
    prev_close = schedule.close.shift().to_dict()
    filing_time = pd.to_datetime(
        releases.known_at, utc=True, format="mixed"
    ).sort_values()
    notices = notices[notices.symbol.eq(symbol)].copy()
    notices["known"] = (
        (pd.to_datetime(notices.notice_date) + pd.Timedelta(days=1))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    notices["next_am"] = [
        sessions[sessions.searchsorted(day, side="right")]
        if sessions.searchsorted(day, side="right") < len(sessions)
        else ""
        for day in notices.release_date
    ]
    rows = []
    for point in frame.itertuples(index=False):
        start = prev_close[pd.Timestamp(point.date)]
        observed = filing_time[filing_time.gt(start) & filing_time.le(point.cut)]
        approved = notices[notices.known.le(point.cut)]
        current = approved[approved.next_am.eq(point.date)]
        pre = approved[approved.release_date.eq(point.date)]
        used = pd.concat([observed, current.known, pre.known])
        rows.append(
            dict(
                is_earnings_window=int(bool(len(observed) or len(current))),
                is_earnings_pre1d=int(bool(len(pre))),
                confirmed_earnings=int(bool(len(observed))),
                earnings_known_at=used.max() if len(used) else pd.NaT,
            )
        )
    enriched = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    enriched["earnings_known_at"] = pd.to_datetime(enriched.earnings_known_at, utc=True)
    enriched["cut"] = enriched.cut.dt.as_unit("ns")
    reference = vix_features(vix, schedule)
    reference["vix_known_at"] = reference.vix_known_at.dt.as_unit("ns")
    merged = pd.merge_asof(
        enriched.sort_values("cut"),
        reference,
        left_on="cut",
        right_on="vix_known_at",
        direction="backward",
    )
    assert (merged.date_vix.isna() | merged.date_vix.lt(merged.date)).all()
    assert (
        merged.earnings_known_at.isna() | merged.earnings_known_at.le(merged.cut)
    ).all()
    return merged.sort_values("cut")


def cluster_features(frame, news):
    substantive = news[
        news.event_type.ne("market_noise") & news.impact_level.ge(1)
    ].copy()
    history = {session: deque(maxlen=3) for session in frame.session.unique()}
    substantive["terms"] = substantive.title.map(lambda t: t.split(" | ", 1)[-1])
    asof = substantive["asof"]
    out = []
    for row in frame.itertuples(index=False):
        lo, hi = asof.searchsorted([row.start, row.cut], side="right")
        current = substantive.iloc[lo:hi]
        previous = [title for block in history[row.session] for title in block]
        history[row.session].append(current.terms.tolist())
        count = len(current)
        if not count:
            out.append([0.0, 0.0, 0.0])
            continue
        text = current.terms.tolist() + previous
        matrix = TfidfVectorizer(
            ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
        ).fit_transform(text)
        related = (matrix[:count] @ matrix[:count].T).toarray() > 0.65
        n_clusters, labels = connected_components(related, directed=False)
        weights = (
            current.confidence
            * current.source_weight
            * np.exp(-DECAY * (row.cut - current["asof"]).dt.total_seconds() / 3600)
        )
        severity = (weights * (current.impact_level - 1).clip(lower=0)).to_numpy()
        maxima = np.zeros(n_clusters)
        np.maximum.at(maxima, labels, severity)
        similarities = np.zeros(n_clusters)
        if previous:
            lookup = (matrix[:count] @ matrix[count:].T).toarray().max(axis=1)
            np.maximum.at(similarities, labels, lookup)
        out.append(
            [np.log1p(n_clusters), np.log1p(maxima.sum()), (similarities < 0.55).mean()]
        )
    return pd.DataFrame(out, index=frame.index, columns=CLUSTERS)


def build_features(raw, news, lexicon, vix, releases, notices, schedule=SCHEDULE):
    frame = aggregate(raw.sort_values("cut"), news, lexicon)
    frame[CLUSTERS] = cluster_features(frame, news)
    frame = frame[frame.session.eq("AM")].copy()
    frame = add_market_context(
        frame, raw.symbol.iloc[0], vix, releases, notices, schedule
    )
    assert (frame.LatestNews.isna() | frame.LatestNews.le(frame.cut)).all()
    assert (frame.market_known_at.isna() | frame.market_known_at.le(frame.cut)).all()
    return frame


def inputs(symbol):
    return (
        pd.read_csv(BASE_DATA / "lexicon.csv"),
        pd.read_csv(DATA / "VIX.csv"),
        pd.read_csv(DATA / f"{symbol}_earnings.csv"),
        pd.read_csv(DATA / "earnings_notices.csv"),
    )
