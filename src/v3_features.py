"""As-of aggregation of the frozen, already-scored financial headlines."""

import re

import numpy as np
import pandas as pd

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
MONOTONE = {"SubstantiveImpactSum", "NewsShock", "HasLevel3Event"}
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


def aggregate(frame, news, lexicon, profile="weighted"):
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
        if profile == "pm_material" and row.session == "PM":
            relevant = relevant[impact[relevant] >= 2]
        ages = (row.cut - asof.iloc[lo:hi]).dt.total_seconds().to_numpy() / 3600
        decay = np.exp(-DECAY * ages)
        active = indices[tier1[indices]]
        mean = std = negative = cumulative = tail = 0.0
        if len(relevant):
            weight = confidence[relevant] * decay[relevant - lo]
            if profile != "filtered":
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
