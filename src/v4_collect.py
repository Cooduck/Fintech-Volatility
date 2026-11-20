"""Fetch V4 public inputs into a separate research snapshot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import re
from urllib.parse import urlencode

from bs4 import BeautifulSoup
import pandas as pd

from collect_data import ROOT, fetch
from collect_finance import publisher

DATA = ROOT / "data" / "v4"
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
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
DOMAINS = [
    "https://247wallst.com",
    "https://finbold.com",
    "https://siliconangle.com",
    "https://financefeeds.com",
    "https://investorplace.com",
]


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def vix_download(until):
    data = pd.read_csv(BytesIO(fetch(VIX_URL)), usecols=["DATE", "CLOSE"])
    data.DATE = pd.to_datetime(data.DATE).dt.strftime("%Y-%m-%d")
    return data[data.DATE.between("2024-08-01", until)].sort_values("DATE")


def yahoo_frame(symbol, interval, start, end):
    query = urlencode(
        dict(
            period1=int(pd.Timestamp(start, tz="UTC").timestamp()),
            period2=int(pd.Timestamp(end, tz="UTC").timestamp()),
            interval=interval,
            includePrePost="false",
        )
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    body = json.loads(fetch(url))["chart"]["result"][0]
    data = pd.DataFrame(body["indicators"]["quote"][0])
    data.insert(0, "utc", pd.to_datetime(body["timestamp"], unit="s", utc=True))
    return data.dropna(subset=["open", "high", "low", "close"])


def filings(symbol, until):
    cik = CIKS[symbol]
    index = json.loads(fetch(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))[
        "filings"
    ]
    recent = [index["recent"]]
    for part in index["files"]:
        if part["filingTo"] >= START and part["filingFrom"] < until:
            recent.append(
                json.loads(fetch("https://data.sec.gov/submissions/" + part["name"]))
            )
    table = pd.concat([pd.DataFrame(item) for item in recent], ignore_index=True)
    table["known_at"] = pd.to_datetime(
        table.acceptanceDateTime, utc=True, format="mixed"
    )
    table["url"] = [
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/{p}"
        for a, p in zip(table.accessionNumber, table.primaryDocument)
    ]
    return table[
        table.known_at.ge(pd.Timestamp(START, tz="UTC"))
        & table.known_at.lt(pd.Timestamp(until, tz="UTC"))
    ].drop_duplicates("accessionNumber")


def quarterly_releases(table):
    from collect_finance import excerpt

    candidates = table[
        table.form.eq("8-K")
        & table["items"].str.contains(r"(?<!\d)2\.02(?!\d)", regex=True, na=False)
    ]
    pattern = re.compile(
        r"(?:financial|earnings|operating) results.{0,120}(?:quarter|months ended|year ended)|"
        r"(?:quarter|year ended).{0,100}(?:financial|earnings) results|announcing its results for the (?:fiscal )?(?:quarter|year)",
        re.I,
    )

    def confirmed(row):
        try:
            content = excerpt(row.url, "8-K")
            return bool(pattern.search(content)) and not re.search(
                r"will (?:post|publish|release) its financial results", content, re.I
            )
        except (ValueError, OSError):
            return False

    with ThreadPoolExecutor(3) as pool:
        flags = list(pool.map(confirmed, candidates.itertuples()))
    return candidates.loc[flags, ["known_at", "url", "items"]].assign(
        verified_quarterly=True
    )


def collect_earnings(symbol):
    path = DATA / f"{symbol}_earnings.csv"
    if path.exists() and "verified_quarterly" in pd.read_csv(path, nrows=0).columns:
        return
    save_csv(quarterly_releases(filings(symbol, END)).assign(symbol=symbol), path)


def first_paragraph(title, url, *, known_at):
    if "SEC filing" in title:
        raise ValueError("use the SEC snapshot excerpt")
    soup = BeautifulSoup(fetch(url), "lxml")
    meta = {m.get("property"): m.get("content") for m in soup.select("meta[property]")}
    modified = meta.get("article:modified_time")
    if not modified or pd.Timestamp(modified).tz_convert("UTC") > pd.Timestamp(
        known_at
    ):
        raise ValueError(
            "no lead with a verified timestamp at or before the archived news cutoff"
        )
    for selector in (".group p", "article p", "main p", "div[class*=body] p"):
        for tag in soup.select(selector):
            line = tag.get_text(" ", strip=True)
            if len(line.split()) >= 18 and not re.match(
                r"(?i)^(subscribe|sign up|this article|click here)", line
            ):
                return " ".join(line.split()[:150])
    raise ValueError("No accessible article lead")


def collect_research_articles(symbol):
    for domain in DOMAINS:
        path = DATA / "publishers" / f"{symbol}_{domain.split('//')[1]}.csv"
        if path.exists():
            continue
        records = publisher(symbol, domain, company=NAMES[symbol])
        save_csv(
            pd.DataFrame(
                records, columns=["utc", "published", "title", "url", "source"]
            ),
            path,
        )
        print(symbol, domain, len(records), flush=True)


def collect_filing_news(symbol):
    from collect_finance import excerpt

    path = DATA / "publishers" / f"{symbol}_edgar.csv"
    if path.exists():
        return
    table = filings(symbol, END)
    major = table["items"].str.contains(
        r"(?<!\d)(?:2\.02|5\.02|1\.01|8\.01)(?!\d)", na=False
    )
    selected = table[(table.form.eq("8-K") & major) | table.form.isin(["10-K", "10-Q"])]
    rows = []
    for row in selected.itertuples():
        try:
            detail = excerpt(row.url, row.form)
        except (OSError, ValueError):
            continue
        rows.append(
            dict(
                utc=row.known_at,
                published=pd.NaT,
                title=f"{symbol} {row.form}. {detail}",
                source="SEC filing",
                url=row.url,
            )
        )
    save_csv(
        pd.DataFrame(rows, columns=["utc", "published", "title", "source", "url"]), path
    )
    print(symbol, "SEC", len(rows), flush=True)


def collect_market(symbol):
    for interval, start in [("1d", "2024-08-01"), ("1h", START)]:
        path = DATA / f"{symbol}_{interval}.csv"
        if not path.exists():
            save_csv(yahoo_frame(symbol, interval, start, END), path)


def run(phase):
    path = DATA / "VIX.csv"
    if not path.exists():
        save_csv(vix_download(FREEZE), path)
    symbols = CORE if phase == "a" else list(NAMES)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(collect_earnings, symbols))
    if phase == "b":
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(collect_market, list(NAMES)[3:]))
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(collect_research_articles, list(NAMES)[3:] + ["MSFT"]))
            list(pool.map(collect_filing_news, list(NAMES)[3:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["a", "b"], default="a")
    run(parser.parse_args().phase)
