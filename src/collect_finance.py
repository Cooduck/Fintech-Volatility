"""Build point-in-time corpora using publisher release times and SEC acceptance times."""

import html
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd
from collect_data import DATA, fetch, events, yahoo

START = pd.Timestamp("2024-11-19", tz="UTC")
END = pd.Timestamp("2026-11-20", tz="UTC")
SYMBOLS = {"NVDA": "1045810", "MSFT": "789019", "TSLA": "1318605"}
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def publisher(symbol, domain, *, company=None, start=START, end=END):
    import urllib.request as u

    company = company or {"NVDA": "Nvidia", "TSLA": "Tesla", "MSFT": "Microsoft"}[symbol]
    pattern = re.compile(r"\b(?:" + company + "|" + symbol + r")\b", re.I)
    rows = []
    for page in range(1, 501):
        params = {
            "search": company,
            "after": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "before": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "per_page": 100,
            "page": page,
            "_fields": "date_gmt,modified_gmt,title,link",
        }
        url = domain + "/wp-json/wp/v2/posts?" + urlencode(params)
        for attempt in range(5):
            try:
                with u.urlopen(
                    u.Request(
                        url,
                        headers={
                            "User-Agent": "Mozilla/5.0 university financial research"
                        },
                    ),
                    timeout=60,
                ) as response:
                    pages = int(response.headers["X-WP-TotalPages"])
                    records = json.load(response)
                break
            except u.HTTPError as exc:
                if exc.code != 429 or attempt == 4:
                    raise
                import time

                time.sleep(
                    max(5 * 2**attempt, int(exc.headers.get("Retry-After", "0")))
                )
            except (OSError, TimeoutError):
                if attempt == 4:
                    raise
                import time

                time.sleep(2**attempt)
        for article in records:
            title = html.unescape(article["title"]["rendered"])
            if pattern.search(title):
                published = pd.to_datetime(article["date_gmt"], utc=True)
                available = max(
                    published, pd.to_datetime(article["modified_gmt"], utc=True)
                )
                if start <= available < end:
                    rows.append(
                        dict(
                            utc=available,
                            published=published,
                            title=title,
                            url=article["link"],
                            source=domain.split("//")[1],
                        )
                    )
        if page >= pages:
            break
    return rows


def finnhub(symbol, start, end, token):
    # Optional alternative when the user has licensed historical access.
    rows = []
    for lo in pd.date_range(start, end, freq="7D", inclusive="left"):
        hi = min(lo + pd.Timedelta(days=7), end)
        params = {
            "symbol": symbol,
            "from": str(lo.date()),
            "to": str(hi.date()),
            "token": token,
        }
        items = json.loads(
            fetch("https://finnhub.io/api/v1/company-news?" + urlencode(params))
        )
        for i in items:
            at = pd.Timestamp(i["datetime"], unit="s", tz="UTC")
            if lo <= at < hi:
                rows.append(
                    dict(utc=at, title=i["headline"], url=i["url"], source=i["source"])
                )
    return rows


def excerpt(url, form):
    soup = BeautifulSoup(fetch(url), "lxml")
    if form != "8-K":
        facts = []
        contexts = {}
        for ctx in soup.find_all(re.compile(r"(?:^|:)context$")):
            if ctx.find(re.compile(r"(?:explicitmember|typedmember)$", re.I)):
                continue
            first = ctx.find(re.compile("startdate$", re.I))
            last = ctx.find(re.compile("enddate$", re.I))
            if not first or not last:
                continue
            start = pd.Timestamp(first.get_text(strip=True))
            end = pd.Timestamp(last.get_text(strip=True))
            duration = (end - start).days + 1
            if (form == "10-Q" and 75 <= duration <= 105) or (
                form == "10-K" and 345 <= duration <= 385
            ):
                contexts[ctx.get("id", "")] = (end, duration)
        for concept in [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "NetIncomeLoss",
            "EarningsPerShareDiluted",
        ]:
            if concept.startswith("RevenueFrom") and any(
                f.startswith("Revenues ") for f in facts
            ):
                continue
            cells = [
                t
                for t in soup.find_all(
                    "ix:nonfraction",
                    attrs={"name": re.compile(":" + concept + "$", re.I)},
                )
                if t.get("contextref") in contexts
                and t.get_text(strip=True) not in ("", "—", "-")
            ]
            if cells:
                cell = max(cells, key=lambda t: contexts[t["contextref"]][0])
                end, duration = contexts[cell["contextref"]]
                facts.append(
                    f"{concept} {cell.get('sign', '')}{cell.get_text(strip=True)} x10^{cell.get('scale', '0')} {cell.get('unitref', '')} over {duration} days ending {end.date()}"
                )
        if facts:
            return "Financial statements: " + ", ".join(facts[:3])
    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)
    matches = list(re.finditer(r"\bItem\s+(?:[12578]\.\d{2})[.\s:–-]+", text, re.I))
    for m in matches:
        part = re.split(
            r"\bItem\s+\d\.\d{2}|SIGNATURES", text[m.end() :], maxsplit=1, flags=re.I
        )[0]
        if len(part) > 85:
            return text[m.start() : m.end()] + part[:650]
    raise ValueError("no substantive extract")


def sec(symbol):
    cik = SYMBOLS[symbol]
    bundle = json.loads(
        fetch(f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json")
    )
    chunks = [bundle["filings"]["recent"]]
    for entry in bundle["filings"]["files"]:
        if entry["filingTo"] >= str(START.date()) and entry["filingFrom"] < str(
            END.date()
        ):
            chunks.append(
                json.loads(fetch("https://data.sec.gov/submissions/" + entry["name"]))
            )
    out = []
    skips = []
    for raw in chunks:
        table = pd.DataFrame(raw)
        for r in table[table.form.isin(["8-K", "10-Q", "10-K"])].itertuples():
            at = pd.to_datetime(r.acceptanceDateTime, utc=True)
            if not START <= at < END:
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{r.accessionNumber.replace('-', '')}/{r.primaryDocument}"
            try:
                detail = excerpt(url, r.form)
                out.append(
                    dict(
                        utc=at,
                        title=f"{symbol} {r.form}. " + detail,
                        url=url,
                        source="SEC filing",
                    )
                )
            except Exception as e:
                skips.append(dict(utc=at, url=url, reason=type(e).__name__))
    pd.DataFrame(skips, columns=["utc", "url", "reason"]).to_csv(
        DATA / f"{symbol}_sec_skipped.csv", index=False
    )
    return out


def build(symbol, finnhub_key=None):
    rows = []
    for domain in [
        "https://247wallst.com",
        "https://investorplace.com",
        "https://financefeeds.com",
        "https://finbold.com",
        "https://siliconangle.com",
    ]:
        path = DATA / f"{symbol}_{domain.split('//')[1]}.csv"
        if not path.exists() or "published" not in pd.read_csv(path, nrows=0).columns:
            items = publisher(symbol, domain)
            pd.DataFrame(
                items, columns=["utc", "published", "title", "url", "source"]
            ).to_csv(path, index=False)
        rows += pd.read_csv(path).to_dict("records")
        print(symbol, domain, len(rows), flush=True)
    if finnhub_key:
        rows += finnhub(symbol, START, END, finnhub_key)
    rows += sec(symbol)
    for event in events():
        if symbol == "NVDA" or event["source"] == "AP race call":
            rows.append(event)
    df = pd.DataFrame(rows)
    df.utc = pd.to_datetime(df.utc, utc=True, format="mixed")
    df = df.sort_values("utc").drop_duplicates(["utc", "title"])
    df.to_csv(DATA / f"{symbol}_finance.csv", index=False)
    print(symbol, len(df), df.source.value_counts().to_dict(), flush=True)
    if symbol != "NVDA":
        yahoo(symbol, "1d", "2024-08-01", "2026-11-20")
        yahoo(symbol, "1h", "2024-11-21", "2026-11-20")


def collect(include_finnhub=False):
    from dotenv import dotenv_values

    token = (
        dotenv_values(DATA.parent / ".env").get("FINNHUB_API_KEY")
        if include_finnhub
        else None
    )
    if include_finnhub and not token:
        raise ValueError("请在项目 .env 中设置 FINNHUB_API_KEY")
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda s: build(s, token), SYMBOLS))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--finnhub", action="store_true")
    collect(parser.parse_args().finnhub)
