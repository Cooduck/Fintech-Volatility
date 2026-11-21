"""Filesystem layout and frozen study calendar."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = ROOT / "data"
ARCHIVE_DATA = BASE_DATA / "v4"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
BENCHMARK_RESULTS = RESULTS / "v4"
PORTFOLIO_ARCHIVE = BENCHMARK_RESULTS / "phase_b"
FINAL_RESULTS = RESULTS / "v5"
PORTFOLIO_RESULTS = FINAL_RESULTS / "phase_b"
NEWS_INDEX = RESULTS / "improved" / "finance_news_index.csv.gz"
DATES = sorted(pd.read_csv(RESULTS / "aligned_daily_decay.csv", usecols=["date"]).date.unique())

START = "2024-11-21"
END = "2026-11-20"
FREEZE = "2026-11-19"


NAMES = dict(
    NVDA="Nvidia",
    MSFT="Microsoft",
    TSLA="Tesla",
    AAPL="Apple",
    AMZN="Amazon",
    META="Meta Platforms",
    GOOGL="Alphabet",
    AMD="Advanced Micro Devices",
    AVGO="Broadcom",
)
CIKS = dict(
    zip(
        NAMES,
        (1045810, 789019, 1318605, 320193, 1018724, 1326801, 1652044, 2488, 1730168),
    )
)
CORE = list(NAMES)[:3]
