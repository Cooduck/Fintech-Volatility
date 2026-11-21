"""Supplement V4 headlines using individually timestamped public sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from urllib.parse import urljoin, urlsplit

import pandas as pd
import requests
from dotenv import dotenv_values

from cnbc_archive import ARCH as PREVIOUS_ARCH, day_links, story
from v4_collect import DATA, END, NAMES, ROOT, START, collect_filing_news, save_csv

TAGS = {
    "NVDA": r"Nvidia|NVDA",
    "MSFT": r"Microsoft|MSFT|Azure",
    "TSLA": r"Tesla|TSLA",
    "AAPL": r"Apple|AAPL|iPhone",
    "AMZN": r"Amazon|AMZN|AWS",
    "META": r"Meta Platforms|META|Facebook|Instagram",
    "GOOGL": r"Alphabet|GOOGL|Google",
    "AMD": r"AMD|Advanced Micro Devices",
    "AVGO": r"AVGO|Broadcom",
}
DOMAINS = {
    "reuters.com": "Reuters",
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "wsj.com": "WSJ",
    "barrons.com": "Barron's",
    "finance.yahoo.com": "Yahoo Finance",
    "bloomberg.com": "Bloomberg",
}
COLS = ["utc", "published", "title", "url", "source", "symbol"]
ARCH = DATA / "cnbc_expansion_cache"


def labels(title):
    return [
        symbol
        for symbol, pattern in TAGS.items()
        if re.search(r"\b(?:" + pattern + r")\b", title, re.I)
    ]


def cnbc():
    import hashlib

    ARCH.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START, END, inclusive="left", tz="UTC")
    with ThreadPoolExecutor(8) as pool:
        content = [
            item
            for chunk in pool.map(lambda d: day_links(d, ARCH, labels), dates)
            for item in chunk
        ]
    links = sorted(set(x for x in content if isinstance(x, str)))
    missing = [x for x in content if isinstance(x, dict)]
    print(
        f"CNBC archive located {len(links)} links; {len(missing)} sitemap errors",
        flush=True,
    )

    def fetch(link):
        cached = PREVIOUS_ARCH / (hashlib.sha256(link.encode()).hexdigest() + ".json")
        if cached.exists():
            rows = json.loads(cached.read_text(encoding="utf8"))
            if rows:
                return [dict(rows[0], symbol=s) for s in labels(rows[0]["title"])]
        return story(link, ARCH, labels)

    rows = []
    with ThreadPoolExecutor(10) as pool:
        for k, block in enumerate(pool.map(fetch, links), 1):
            for row in block:
                (missing if "error" in row else rows).append(row)
            if k % 500 == 0:
                print(f"CNBC {k}/{len(links)}", flush=True)
    save_csv(pd.DataFrame(rows, columns=COLS), DATA / "cnbc_expansion.csv")
    save_csv(
        pd.DataFrame(missing, columns=["url", "error"]),
        DATA / "cnbc_expansion_errors.csv",
    )
    print("CNBC articles", len(rows), "errors", len(missing), flush=True)


def fool(symbols=None):
    """Download publicly available investment articles from their monthly index."""
    from xml.etree import ElementTree
    from bs4 import BeautifulSoup, SoupStrainer
    from cnbc_archive import get
    from hashlib import sha256

    symbols = set(NAMES) if symbols is None else set(symbols)
    store = DATA / "fool_expansion_cache"
    store.mkdir(parents=True, exist_ok=True)

    def month(day):
        objects = ElementTree.fromstring(
            get(f"https://www.fool.com/sitemap/{day:%Y/%m}")
        )
        links = []
        for item in objects:
            link = item.findtext("{*}loc") or ""
            stamp = item.findtext("{*}lastmod")
            if "/investing/" not in link or not stamp:
                continue
            slug = urlsplit(link).path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
            if symbols.intersection(labels(slug)) and stamp[:10] >= START and stamp[:10] < END:
                links.append((link, stamp))
        return links

    dates = pd.date_range("2024-11-01", END, freq="MS")
    with ThreadPoolExecutor(5) as pool:
        groups = list(pool.map(month, dates))
    links = dict(pair for group in groups for pair in group)
    print("Fool investment article links", len(links), flush=True)

    def article(job):
        link, stamp = job
        path = store / f"{sha256(link.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf8"))
        try:
            soup = BeautifulSoup(get(link), "lxml", parse_only=SoupStrainer("meta"))
            meta = {x.get("property"): x.get("content") for x in soup.find_all("meta")}
            published = pd.to_datetime(meta["article:published_time"], utc=True)
            modified = pd.to_datetime(meta.get("article:modified_time"), utc=True)
            available = max(
                published,
                pd.Timestamp(stamp),
                modified if pd.notna(modified) else published,
            )
            title = meta["og:title"].split(" | The Motley Fool", 1)[0]
            rows = [
                dict(
                    utc=available.isoformat(),
                    published=published.isoformat(),
                    title=title,
                    url=link,
                    source="The Motley Fool",
                    symbol=s,
                )
                for s in labels(title)
                if START <= available.date().isoformat() < END
            ]
            path.write_text(json.dumps(rows), encoding="utf8")
            return rows
        except (OSError, KeyError, ValueError):
            return []

    rows = []
    with ThreadPoolExecutor(10) as pool:
        for i, block in enumerate(pool.map(article, links.items()), 1):
            rows.extend(block)
            if i % 500 == 0:
                print(f"Fool {i}/{len(links)}", flush=True)
    save_csv(pd.DataFrame(rows, columns=COLS), DATA / "fool_expansion.csv")
    print("Fool records", len(rows), flush=True)


def gdelt(provider=requests):
    # DOC searches a rolling window. Each catalogue timestamp is GDELT's
    # first-seen time, with publication left unknown.
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"
    end = pd.Timestamp(END, tz="UTC")
    start = max(pd.Timestamp(START, tz="UTC"), end - pd.Timedelta(days=90))
    rows = []
    for symbol, company in NAMES.items():
        terms = f'("{company}" OR "{symbol}")'
        whitelist = "(" + " OR ".join(f"domainis:{domain}" for domain in DOMAINS) + ")"
        for left in pd.date_range(start, end, freq="7D", inclusive="left"):
            right = min(left + pd.Timedelta(days=7), end)
            args = dict(
                query=terms + " " + whitelist,
                mode="ArtList",
                format="json",
                maxrecords=250,
                startdatetime=left.strftime("%Y%m%d%H%M%S"),
                enddatetime=right.strftime("%Y%m%d%H%M%S"),
            )
            response = provider.get(endpoint, params=args, timeout=15)
            response.raise_for_status()
            body = response.json()
            articles = body.get("articles", [])
            if len(articles) == 250:
                print(f"GDELT capped batch {symbol} {left.date()}", flush=True)
            for item in articles:
                host = (urlsplit(item["url"]).hostname or "").removeprefix("www.")
                at = pd.to_datetime(
                    item.get("seendate"),
                    format="%Y%m%dT%H%M%SZ",
                    utc=True,
                    errors="coerce",
                )
                if (
                    host in DOMAINS
                    and pd.notna(at)
                    and left <= at < right
                    and symbol in labels(item["title"])
                ):
                    rows.append(
                        dict(
                            utc=at,
                            published=pd.NaT,
                            title=item["title"],
                            url=item["url"],
                            source=DOMAINS[host],
                            symbol=symbol,
                        )
                    )
            time.sleep(5)
        save_csv(
            pd.DataFrame(rows, columns=COLS).drop_duplicates(
                ["symbol", "url", "title"]
            ),
            DATA / "gdelt_expansion.csv",
        )
    return rows


def finnhub(provider=requests):
    token = dotenv_values(ROOT / ".env").get("FINNHUB_API_KEY")
    if not token:
        print("FINNHUB_API_KEY missing from project .env", flush=True)
        return
    endpoint = "https://finnhub.io/api/v1/company-news"
    rows = []
    end = pd.Timestamp(END, tz="UTC")
    for symbol in NAMES:
        for day in pd.date_range(START, END, freq="7D", inclusive="left", tz="UTC"):
            right = min(day + pd.Timedelta(days=7), end)
            response = provider.get(
                endpoint,
                headers={"X-Finnhub-Token": token},
                params={
                    "symbol": symbol,
                    "from": str(day.date()),
                    "to": str(right.date()),
                },
                timeout=15,
            )
            response.raise_for_status()
            for item in response.json():
                ts = pd.Timestamp(item["datetime"], unit="s", tz="UTC")
                if day <= ts < right and item.get("headline") and item.get("url"):
                    rows.append(
                        dict(
                            utc=ts,
                            published=ts,
                            title=item["headline"],
                            url=item["url"],
                            source="Finnhub",
                            symbol=symbol,
                            publisher_reported=item.get("source", ""),
                            lead_summary=" ".join(
                                item.get("summary", "").split()[:150]
                            ),
                        )
                    )
            time.sleep(1.1)
        save_csv(
            pd.DataFrame(
                rows, columns=COLS + ["publisher_reported", "lead_summary"]
            ).drop_duplicates(["symbol", "url", "title"]),
            DATA / "finnhub_expansion.csv",
        )


def specialist():
    import ssl
    import certifi
    from collect_finance import publisher

    queries = [
        ("AMD", "AMD", "https://wccftech.com"),
        ("AVGO", "Broadcom", "https://www.networkworld.com"),
        ("AVGO", "VMware", "https://www.networkworld.com"),
    ]
    context = ssl.create_default_context(cafile=certifi.where())

    def collect(query):
        symbol, company, domain = query
        rows = publisher(
            symbol, domain, company=company, ssl_context=context,
            start=pd.Timestamp(START, tz="UTC"), end=pd.Timestamp(END, tz="UTC"),
        )
        columns = ["utc", "published", "title", "url", "source"]
        name = f"{symbol}_{urlsplit(domain).netloc}_{company.lower()}.csv"
        save_csv(pd.DataFrame(rows, columns=columns), DATA / "publishers" / name)
        print(name, len(rows), flush=True)

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(collect, queries))


def toms():
    from bs4 import BeautifulSoup, SoupStrainer

    site = "https://www.tomshardware.com"
    pattern = re.compile(r"\b(?:" + TAGS["AMD"] + r")\b", re.I)

    def archive(month):
        url = f"{site}/news/archive/{month:%Y/%m}"
        try:
            response = requests.get(url, timeout=(5, 15))
            response.raise_for_status()
        except requests.RequestException as err:
            print(f"Archive unavailable {month:%Y/%m}: {err}", flush=True)
            return []
        page = BeautifulSoup(response.content, "lxml")
        return [
            urljoin(url, anchor["href"])
            for anchor in page.select(".day-article a[href]")
            if pattern.search(anchor.get_text(" ", strip=True))
        ]

    def article(url):
        if urlsplit(url).netloc != urlsplit(site).netloc:
            return None
        try:
            page = requests.get(url, timeout=(5, 15))
            page.raise_for_status()
            doc = BeautifulSoup(page.content, "lxml", parse_only=SoupStrainer(["meta", "script"]))
            meta = {tag.get("property"): tag.get("content") for tag in doc.find_all("meta")}
            stamps = {}
            for schema in doc.find_all("script", type="application/ld+json"):
                pairs = re.findall(
                    r'"(datePublished|dateModified)"\s*:\s*"([^"]+)"', schema.string or ""
                )
                stamps.update(pairs)
            published = pd.to_datetime(
                meta.get("article:published_time") or stamps["datePublished"], utc=True
            )
            modified = pd.to_datetime(
                meta.get("article:modified_time") or stamps["dateModified"], utc=True
            )
            title = meta["og:title"]
        except (KeyError, ValueError, requests.RequestException):
            return None
        available = max(published, modified)
        if not START <= available.date().isoformat() < END or not pattern.search(title):
            return None
        return dict(
            utc=available, published=published, title=title, url=url, source="Tom's Hardware"
        )

    months = pd.date_range(pd.Timestamp(START).replace(day=1), END, freq="MS")
    with ThreadPoolExecutor(6) as pool:
        links = sorted({url for group in pool.map(archive, months) for url in group})
        print("Tom's Hardware archive links", len(links), flush=True)
        rows = [row for row in pool.map(article, links) if row is not None]
    columns = ["utc", "published", "title", "url", "source"]
    save_csv(pd.DataFrame(rows, columns=columns), DATA / "publishers" / "AMD_toms.csv")
    print("Tom's Hardware records", len(rows), flush=True)


def sec_earnings():
    """Cache the opening paragraphs of the earnings releases filed with EDGAR."""
    from bs4 import BeautifulSoup
    from collect_data import fetch
    from v3_pipeline import INDEX
    from v4_scoring import lead_identity, lead_snapshots, prepare

    def retrieve(job):
        (symbol, filing, stamp), news = job
        release = None
        status = "no_exhibit"
        try:
            doc = BeautifulSoup(fetch(filing), "lxml")
            filing_path = urlsplit(filing)
            accession = filing_path.path.rsplit("/", 1)[0]
            for link in doc.select("tr a[href]"):
                name = link.get_text(" ", strip=True)
                label = link.find_parent("tr").get_text(" ", strip=True)
                if not re.match(r"^99\.0?1\b", label) and not re.search(
                    r"\bpress release\b", name, re.I
                ):
                    continue
                address = urljoin(filing, link["href"])
                parts = urlsplit(address)
                if (
                    parts.netloc == filing_path.netloc
                    and parts.path.rsplit("/", 1)[0] == accession
                    and parts.path != filing_path.path
                ):
                    release = address
                    break
            if release:
                exhibit = BeautifulSoup(fetch(release), "lxml")
                status = "no_earnings_opening_paragraph"
                for para in exhibit.find_all(["p", "div"]):
                    if para.find(["p", "div", "table"]):
                        continue
                    words = para.get_text(" ", strip=True).split()
                    text = " ".join(words)
                    if (
                        len(words) >= 16
                        and re.search(r"\b(?:quarter|quarterly|fiscal)\b", text, re.I)
                        and re.search(
                            r"\b(?:reported|reports|announced|announces|released|releases)\b",
                            text, re.I,
                        )
                    ):
                        return (
                            dict(symbol=symbol, filing_url=filing, exhibit_url=release,
                                 known_at=stamp, status="verified"),
                            {lead_identity(row): " ".join(words[:150]) for row in news.itertuples()},
                        )
        except (OSError, ValueError) as err:
            status = type(err).__name__
        return (
            dict(symbol=symbol, filing_url=filing, exhibit_url=release, known_at=stamp,
                 status=status),
            {},
        )

    filings = pd.concat(
        [pd.read_csv(DATA / f"{symbol}_earnings.csv") for symbol in NAMES],
        ignore_index=True,
    )
    filings = filings[filings.verified_quarterly.eq(True)].copy()
    filings["known_at"] = pd.to_datetime(filings.known_at, utc=True, format="mixed")
    news = prepare(pd.read_csv(INDEX))
    news = news[news.source.eq("SEC filing")].merge(
        filings[["symbol", "url", "known_at"]],
        left_on=["symbol", "url", "utc"],
        right_on=["symbol", "url", "known_at"],
        validate="many_to_one",
    )
    reports = []
    originals = {}
    with ThreadPoolExecutor(3) as pool:
        jobs = news.groupby(["symbol", "url", "known_at"])
        for report, entries in pool.map(retrieve, jobs):
            reports.append(report)
            originals.update(entries)
    cached = lead_snapshots()
    cached.update(originals)
    save_csv(
        pd.DataFrame(
            [(*identity, lead) for identity, lead in cached.items()],
            columns=["url", "known_at", "headline", "lead"],
        ),
        DATA / "lead_snapshots.csv",
    )
    resolution = pd.DataFrame(reports, columns=[
        "symbol", "filing_url", "exhibit_url", "known_at", "status",
    ])
    save_csv(resolution, ROOT / "results" / "v5" / "phase_b" / "earnings_lead_sources.csv")
    print(len(filings), "quarterly filings checked against the corpus", flush=True)
    print(resolution.status.value_counts().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "provider",
        choices=["cnbc", "fool", "gdelt", "finnhub", "sec", "sec_earnings", "specialist", "toms"],
    )
    parser.add_argument("--ticker",action="append",choices=list(NAMES))
    opts = parser.parse_args()
    if opts.provider == "sec":
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(lambda s: collect_filing_news(s, expanded=True), NAMES))
    elif opts.provider=="fool":
        fool(opts.ticker)
    else:
        globals()[opts.provider]()


if __name__ == "__main__":
    main()
