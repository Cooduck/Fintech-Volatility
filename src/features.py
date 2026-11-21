"""Point-in-time news, market context and event-cluster features."""

from collections import deque
import re

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ARCHIVE_DATA as DATA, BASE_DATA
from market_data import MARKET, SCHEDULE, bars, price_panel


BASE_SENTIMENT = [
    "SentMean",
    "SentStd",
    "Momentum",
    "NegRatio",
    "NewsShock",
    "Interaction",
]
SENTIMENT = BASE_SENTIMENT + ["has_news", "NoiseShare"]
TAIL = ["SubstantiveImpactSum", "MinSentiment", "HasLevel3Event"]
SOURCE_TIERS = {
    "SEC filing": 1.0,
    "CNBC": 1.0,
    "AP race call": 1.0,
    "NVIDIA via GlobeNewswire": 1.0,
    "PR Newswire": 1.0,
    "Reuters": 1.0,
    "WSJ": 1.0,
    "MarketWatch": 1.0,
    "Barron's": 1.0,
    "Bloomberg": 1.0,
    "Yahoo Finance": 0.3,
    "Finnhub": 0.3,
    "techcrunch.com": 0.3,
    "www.thurrott.com": 0.3,
    "The Motley Fool": 0.1,
    "siliconangle.com": 0.3,
    "financefeeds.com": 0.3,
    "wccftech.com": 0.3,
    "www.networkworld.com": 0.3,
    "Tom's Hardware": 0.3,
    "247wallst.com": 0.1,
    "finbold.com": 0.1,
    "investorplace.com": 0.1,
}
AFTER_MARKET = re.compile(
    r"\b(?:clos(?:ed|es|ing) (?:up|down|higher|lower|at)|"
    r"ended (?:the day|the session|trading)|finished (?:the day|the session)|"
    r"at (?:the )?(?:market )?close|by (?:the )?(?:market )?close|"
    r"after (?:the )?(?:closing bell|market closed)|"
    r"on the day|at the end of (?:the )?(?:day|session))\b",
    re.IGNORECASE,
)
DECAY = 0.25


def timestamp_news(index, clock):
    news = index.copy()
    news["published"] = pd.to_datetime(news.published, utc=True, format="mixed")
    news["available"] = pd.to_datetime(news.utc, utc=True, format="mixed")
    news["source_weight"] = news.source.map(SOURCE_TIERS)
    if news.source_weight.isna().any():
        raise ValueError(
            f"Unclassified sources: {news.loc[news.source_weight.isna(), 'source'].unique()}"
        )
    news["delay_hours"] = (news.available - news.published).dt.total_seconds() / 3600
    if (news.delay_hours.dropna() < 0).any():
        raise ValueError("Availability precedes first publication")
    late = news.delay_hours.gt(1)
    hindsight = news.title.str.contains(AFTER_MARKET, na=False)
    news["action"] = np.select(
        [late & hindsight, late & ~hindsight], ["exclude", "rollback"], default="keep"
    )
    if "lead_key" in news:
        refined = news.lead_key.fillna("").ne("")
        # The title can be backdated for sensitivity analysis. The lead cannot.
        news.loc[refined & news.action.eq("rollback"), "action"] = "keep"
    news["asof"] = news.available
    if clock == "bounded":
        news = news[news.action.ne("exclude")].copy()
        rolled = news.action.eq("rollback")
        news.loc[rolled, "asof"] = news.loc[rolled, "published"]
    return news.sort_values(["asof", "source", "key"]).reset_index(drop=True)


def aggregate(frame, news, lexicon):
    """News rows must belong to one ticker. The cutoff is right-inclusive."""
    asof = news["asof"]
    polarity = news.sentiment_score.to_numpy()
    impact = news.impact_level.to_numpy()
    confidence = news.confidence.to_numpy()
    source_weight = news.source_weight.to_numpy()
    substantive = news.event_type.ne("market_noise").to_numpy() & (impact >= 1)
    tier1 = substantive & (impact == 3) & (source_weight == 1)
    by_year = {}
    for year in asof.dt.year.unique():
        by_year[year] = (
            set(lexicon.Word[lexicon.Positive.between(1, year - 1)]),
            set(lexicon.Word[lexicon.Negative.between(1, year - 1)]),
        )
    lex = []
    for time, text in zip(asof, news.title):
        words = re.findall(r"[A-Z]+", text.split(" | ", 1)[-1].upper())
        positive, negative = by_year[time.year]
        lex.append(
            (sum(w in positive for w in words) - sum(w in negative for w in words))
            / max(1, len(words))
        )
    lex = np.asarray(lex)
    records = []
    for row in frame.itertuples(index=False):
        lo, hi = asof.searchsorted([row.start, row.cut], side="right")
        indices = np.arange(lo, hi)
        relevant = indices[substantive[indices]]
        ages = (row.cut - asof.iloc[lo:hi]).dt.total_seconds().to_numpy() / 3600
        decay = np.exp(-DECAY * ages)
        active = indices[tier1[indices]]
        mean = std = negative = cumulative = tail = 0.0
        if len(relevant):
            weight = confidence[relevant] * decay[relevant - lo]
            weight *= source_weight[relevant]
            if weight.sum() > 0:
                mean = np.average(polarity[relevant], weights=weight)
                std = np.sqrt(
                    np.average(np.square(polarity[relevant] - mean), weights=weight)
                )
                negative = np.average(polarity[relevant] < -0.5, weights=weight)
            cumulative = np.log1p(np.dot(weight, np.maximum(0.0, impact[relevant] - 1)))
            tail = polarity[relevant].min()
        records.append(
            dict(
                TotalNews=len(indices),
                N=len(relevant),
                SentMean=mean,
                SentStd=std,
                NegRatio=negative,
                NoiseShare=float((~substantive[indices]).mean()) if hi > lo else 0.0,
                has_news=int(bool(len(relevant))),
                SubstantiveImpactSum=cumulative,
                MinSentiment=tail,
                HasLevel3Event=int(bool(len(active))),
                Lexicon=np.average(lex[indices], weights=decay / decay.max())
                if hi > lo
                else 0.0,
                LatestTier1=asof.iloc[active[-1]] if len(active) else pd.NaT,
                LatestNews=asof.iloc[hi - 1] if hi > lo else pd.NaT,
            )
        )
    out = pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    by_session = out.groupby("session")
    out["Momentum"] = out.SentMean - by_session.SentMean.transform(
        lambda s: s.shift().rolling(3).mean()
    )
    out["NewsShock"] = out.N / (
        1 + by_session.N.transform(lambda s: s.shift().rolling(20).mean())
    )
    out["Interaction"] = out.SentMean.abs() * out.NewsShock
    fields = SENTIMENT + TAIL + ["Lexicon"]
    out[fields] = out[fields].round(7).astype("float32")
    return out


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
    symbol, reader=price_reader, schedule=SCHEDULE, qqq=None
):
    def panel(ticker):
        return price_panel(
            "halfday",
            ticker,
            continuous=True,
            read_bars=reader,
            schedule_table=schedule,
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


def with_target(frame, target):
    if target == "B1":
        return frame
    return frame.assign(
        z=frame.z_intraday,
        sigma=frame.sigma_intraday,
        y=(frame.z_intraday > 1.5).astype(int),
    )


def derive(frame):
    return frame.assign(
        NegSentIntensity=(-frame.SentMean).clip(lower=0),
        NegMinIntensity=(-frame.MinSentiment).clip(lower=0),
    )
