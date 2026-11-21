"""Read publisher pages and archived official filings."""

from urllib.parse import urlencode
from urllib.request import Request, urlopen
import gzip
import html
import json
import re
import time
import warnings

from bs4 import BeautifulSoup, SoupStrainer, XMLParsedAsHTMLWarning
import pandas as pd

from config import BASE_DATA, END as STUDY_END, START as STUDY_START
from market_data import fetch

START = pd.Timestamp(STUDY_START, tz="UTC")
END = pd.Timestamp(STUDY_END, tz="UTC")
ARCH = BASE_DATA / "cnbc_archive"
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def publisher(
    symbol, domain, *, company=None, start=START, end=END, ssl_context=None
):
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
                    context=ssl_context,
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


def excerpt(url, form, *, item=None):
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
    item_pattern = re.escape(item) if item else r"[12578]\.\d{2}"
    heading = rf"\bItem\s+(?:{item_pattern})[.\s:–-]+"
    for m in re.finditer(heading, text, re.I):
        part = re.split(
            r"\bItem\s+\d\.\d{2}|SIGNATURES", text[m.end() :], maxsplit=1, flags=re.I
        )[0]
        if len(part) > 85:
            return text[m.start() : m.end()] + part[:650]
    raise ValueError("no substantive extract")


TAGS = {
    "NVDA": r"\b(?:nvidia|nvda)\b",
    "MSFT": r"\b(?:microsoft|msft|azure)\b",
    "TSLA": r"\b(?:tesla|tsla)\b",
}


def names(title):
    return [s for s, pat in TAGS.items() if re.search(pat, title, re.I)]


def fetch_cnbc(url):
    for attempt in range(3):
        try:
            with urlopen(
                Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 university financial research",
                        "Accept-Encoding": "gzip",
                    },
                ),
                timeout=18,
            ) as r:
                data = r.read()
                return (
                    gzip.decompress(data)
                    if r.headers.get("Content-Encoding") == "gzip"
                    else data
                ).decode("utf8", errors="replace")
        except Exception as exc:
            if getattr(exc, "code", 0) in (403, 404) or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def day_links(day, archive=ARCH, identify=names):
    path = archive / (str(day.date()) + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    url = f"https://www.cnbc.com/site-map/articles/{day.year}/{day:%B}/{day.day}/"
    try:
        raw = fetch_cnbc(url)
        pairs = []
        for title, link in re.findall(
            r'"__typename":"cnbcnewsstory","title":("(?:[^"\\]|\\.)*"),"url":("(?:[^"\\]|\\.)*")',
            raw,
        ):
            headline = json.loads(title)
            link = json.loads(link)
            if identify(headline) and "/pro/" not in link:
                pairs.append(link)
        if not pairs and "site-map" not in raw:
            raise ValueError("Unrecognized sitemap")
        path.write_text(json.dumps(pairs), encoding="utf8")
        return pairs
    except Exception as e:
        return [
            dict(url=url, error=str(type(e).__name__) + str(getattr(e, "code", "")))
        ]


def story(link, archive=ARCH, identify=names):
    import hashlib

    path = archive / (hashlib.sha256(link.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    try:
        b = BeautifulSoup(fetch_cnbc(link), "lxml", parse_only=SoupStrainer("meta"))
        meta = {m.get("property"): m.get("content") for m in b.find_all("meta")}
        pub = pd.to_datetime(meta["article:published_time"], utc=True)
        mod = pd.to_datetime(meta.get("article:modified_time"), utc=True)
        at = max(pub, mod) if pd.notna(mod) else pub
        title = html.unescape(meta["og:title"])
        rows = [
            dict(
                utc=at.isoformat(),
                published=pub.isoformat(),
                title=title,
                url=link,
                source="CNBC",
                symbol=s,
            )
            for s in identify(title)
            if START <= at < END
        ]
        path.write_text(json.dumps(rows), encoding="utf8")
        return rows
    except Exception as e:
        return [
            dict(url=link, error=str(type(e).__name__) + str(getattr(e, "code", "")))
        ]
