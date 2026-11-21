"""Collect public CNBC financial headlines. Use the later of publication and modification."""

import gzip, html, json, re, time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
from collect_finance import DATA, START, END

ARCH = DATA / "cnbc_archive"
ARCH.mkdir(exist_ok=True)
TAGS = {
    "NVDA": r"\b(?:nvidia|nvda)\b",
    "MSFT": r"\b(?:microsoft|msft|azure)\b",
    "TSLA": r"\b(?:tesla|tsla)\b",
}


def names(title):
    return [s for s, pat in TAGS.items() if re.search(pat, title, re.I)]


def get(url):
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
        raw = get(url)
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
        b = BeautifulSoup(get(link), "lxml", parse_only=SoupStrainer("meta"))
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


def main():
    days = pd.date_range(START, END, inclusive="left")
    with ThreadPoolExecutor(6) as pool:
        parts = sum(pool.map(day_links, days), [])
    errors = [x for x in parts if isinstance(x, dict)]
    links = sorted(set(x for x in parts if isinstance(x, str)))
    print(
        "archive days",
        len(days),
        "failed",
        len(errors),
        "links",
        len(links),
        flush=True,
    )
    rows = []
    with ThreadPoolExecutor(6) as pool:
        for block in pool.map(story, links):
            for row in block:
                (errors if "error" in row else rows).append(row)
    pd.DataFrame(
        rows, columns=["utc", "published", "title", "url", "source", "symbol"]
    ).to_csv(DATA / "cnbc_finance.csv", index=False)
    pd.DataFrame(errors, columns=["url", "error"]).to_csv(
        DATA / "cnbc_unavailable.csv", index=False
    )
    print("rows", len(rows), "total unavailable", len(errors), flush=True)


if __name__ == "__main__":
    main()
