"""Capture timestamped headlines before 09:25 ET and settle out-of-time labels."""

import argparse
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from urllib.parse import urlencode

import exchange_calendars as xc
import joblib
import pandas as pd
import requests

from collect_data import ROOT
from panels import price_panel
from semantic import ANCHORED, key, score
from v3_features import timestamp_news
from v3_pipeline import INDEX
from v4_collect import DOMAINS, NAMES, filings, vix_download, yahoo_frame
from v4_features import build_features, inputs, market_frame
from v4_models import route_scores
from v4_pipeline import digest, output_dir

OUT = ROOT / "results" / "v4_forward"
FIRST_DAY = "2026-11-20"


def write_event(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("a", encoding="utf8") as file:
        file.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def events(name):
    path = OUT / name
    return (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf8").splitlines()
            if line
        ]
        if path.exists()
        else []
    )


def now():
    return pd.Timestamp.now(tz="UTC")


def current_calendar(day):
    return xc.get_calendar(
        "XNYS", start=day - pd.Timedelta(days=160), end=day + pd.Timedelta(days=14)
    ).schedule


def load_freeze(phase):
    folder = output_dir(phase)
    manifest = json.loads((folder / "frozen_manifest.json").read_text(encoding="utf8"))
    file = folder / "frozen_models.joblib"
    if digest(file) != manifest["model_sha256"]:
        raise ValueError("Frozen model digest changed")
    for path, expected in manifest["data_and_code"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return manifest, joblib.load(file)


def latest_news(symbol, domain, start, cutoff):
    query = urlencode(
        dict(
            search=NAMES[symbol],
            per_page=50,
            after=start.strftime("%Y-%m-%dT%H:%M:%S"),
            before=cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
            _fields="date_gmt,modified_gmt,title,link",
        )
    )
    url = f"{domain}/wp-json/wp/v2/posts?{query}"
    response = requests.get(url, timeout=(3, 5))
    observed = now()
    response.raise_for_status()
    if observed > cutoff:
        raise ValueError("publisher response missed the feature cutoff")
    pattern = re.compile(
        r"\b(?:" + re.escape(NAMES[symbol]) + "|" + symbol + r")\b", re.I
    )
    article_rows = []
    for item in response.json():
        title = html.unescape(item["title"]["rendered"])
        available = max(
            pd.Timestamp(item["date_gmt"], tz="UTC"),
            pd.Timestamp(item["modified_gmt"], tz="UTC"),
        )
        if start < available <= observed <= cutoff and pattern.search(title):
            article_rows.append(
                dict(
                    symbol=symbol,
                    title=symbol + " | " + title,
                    url=item["link"],
                    source=domain.split("//")[1],
                    published=pd.Timestamp(item["date_gmt"], tz="UTC").isoformat(),
                    first_seen_utc=observed.isoformat(),
                    utc=observed.isoformat(),
                    publication_available=available.isoformat(),
                )
            )
    return article_rows


def market_reader(day):
    @lru_cache(None)
    def read(ticker, interval):
        start = (day - pd.Timedelta(days=125)).strftime("%Y-%m-%d")
        end = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yahoo_frame(ticker, interval, start, end).set_index("utc")
        return data

    return read


def collect_scores(symbols, start, cutoff):
    jobs = [(ticker, domain) for ticker in symbols for domain in DOMAINS]

    def collect(job):
        ticker, domain = job
        try:
            return job, latest_news(ticker, domain, start, cutoff), None
        except (requests.RequestException, ValueError, KeyError) as error:
            return job, [], type(error).__name__

    with ThreadPoolExecutor(15) as pool:
        responses = list(pool.map(collect, jobs))
    failures = [
        dict(symbol=s, source=u, error=err) for (s, u), _, err in responses if err
    ]
    seen = {(r["symbol"], r["url"]) for r in events("first_seen.jsonl")}
    rows = []
    for _, articles, _ in responses:
        for record in articles:
            identity = (record["symbol"], record["url"])
            if identity not in seen:
                write_event("first_seen.jsonl", record)
                seen.add(identity)
                rows.append(record)
    cache = score([r["title"] for r in rows], system=ANCHORED)
    for row in rows:
        parsed = cache[key(row["title"], ANCHORED)]
        write_event(
            "scored_first_seen.jsonl",
            dict(
                row, **{k: v for k, v in parsed.items() if k not in ("title", "text")}
            ),
        )
    return failures


def predict(day, schedule, phase, request_time):
    date = str(day.date())
    if date < FIRST_DAY or day not in schedule.index:
        return "outside trading calendar"
    session = schedule.loc[day]
    fixed_cut = session.open - pd.Timedelta(minutes=5)
    if request_time < fixed_cut - pd.Timedelta(minutes=5) or request_time >= fixed_cut:
        return "outside the 09:20-09:25 ET capture window"
    if any(row.get("date") == date for row in events("daily_log.jsonl")):
        return "capture already attempted"
    manifest, routes = load_freeze(phase)
    if pd.Timestamp(manifest["created_utc"]) >= fixed_cut:
        return "model freeze occurred after this cutoff"
    if pd.Timestamp(manifest["training_dates"][-1]) >= day:
        raise ValueError("training dates overlap the forward decision")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        with (OUT / f"attempt_{date}.json").open("x", encoding="utf8") as token:
            token.write(
                json.dumps(dict(date=date, phase=phase, utc=request_time), default=str)
            )
    except FileExistsError:
        return "another capture already claimed this trading day"
    write_event(
        "daily_log.jsonl",
        dict(date=date, phase=phase, status="started", utc=request_time),
    )
    start = schedule.iloc[schedule.index.get_loc(day) - 1].open + pd.Timedelta(hours=3)
    try:
        failures = collect_scores(manifest["symbols"], start, fixed_cut)
        if any(
            sum(f["symbol"] == symbol for f in failures) == len(DOMAINS)
            for symbol in manifest["symbols"]
        ):
            raise ValueError("a ticker has no successful live news source")
        import time

        remaining = (fixed_cut - now()).total_seconds()
        if remaining > 0:
            time.sleep(remaining)
        baseline = pd.read_csv(
            INDEX if phase == "a" else output_dir(phase) / "scored_news.csv.gz"
        )
        combined = pd.concat(
            [baseline, pd.DataFrame(events("scored_first_seen.jsonl"))],
            ignore_index=True,
        )
        reader = market_reader(day)
        vix = vix_download(str((day - pd.Timedelta(days=1)).date()))
        qqq = price_panel(
            "halfday",
            "QQQ",
            continuous=True,
            read_bars=reader,
            schedule_table=schedule,
            forecast_date=date,
        )
        pending = []
        for symbol in manifest["symbols"]:
            lex, _, _, notices = inputs(symbol)
            data = market_frame(symbol, reader, schedule, forecast_date=date, qqq=qqq)
            final = data.date.eq(date)
            data.loc[final, "cut"] = fixed_cut
            firm_filings = filings(symbol, str((day + pd.Timedelta(days=1)).date()))
            recent = firm_filings[
                firm_filings.known_at.gt(start) & firm_filings.known_at.le(fixed_cut)
            ]
            accepted = quarterly_releases(recent)
            news = timestamp_news(combined[combined.symbol.eq(symbol)], "conservative")
            frame = build_features(data, news, lex, vix, accepted, notices, schedule)
            point = frame[frame.date.eq(date)]
            for target, route in routes.items():
                raw = route_scores(point, route)[0]
                active = bool(point.HasLevel3Event.iloc[0])
                pending.append(
                    dict(
                        date=date,
                        symbol=symbol,
                        target=target,
                        phase=phase,
                        feature_cut=fixed_cut,
                        score=float(raw),
                        alert=bool(raw >= route["alert_line"]),
                        event_brief=active,
                        statistical_brief=bool(
                            raw >= route["brief_line"]
                            or (target == "B1" and raw >= 3.5 * route["prior"])
                        ),
                        first_seen_article_count=int(point.TotalNews.iloc[0]),
                        model_sha256=manifest["model_sha256"],
                    )
                )
        issued = now()
        if issued >= session.open:
            write_event(
                "daily_log.jsonl",
                dict(
                    date=date,
                    phase=phase,
                    status="late_before_scoring_completed",
                    utc=issued,
                    source_failures=failures,
                ),
            )
            return "late; prediction excluded"
        for row in pending:
            write_event("predictions.jsonl", dict(row, issued_utc=issued))
        write_event(
            "daily_log.jsonl",
            dict(
                date=date,
                phase=phase,
                status="issued",
                utc=issued,
                source_failures=failures,
                count=len(pending),
            ),
        )
        return f"issued {len(pending)} forward rows"
    except Exception as error:
        write_event(
            "daily_log.jsonl",
            dict(date=date, phase=phase, status="failed", error=str(error), utc=now()),
        )
        raise


def settle(day, schedule, actual_now):
    predictions = events("predictions.jsonl")
    seen = {(r["date"], r["symbol"], r["target"]) for r in events("outcomes.jsonl")}
    reader = market_reader(day)
    for entry in predictions:
        date = pd.Timestamp(entry["date"])
        if date not in schedule.index:
            continue
        next_date = schedule.index.get_loc(date) + 1
        if next_date >= len(schedule) or schedule.close.iloc[next_date] > actual_now:
            continue
        identifier = tuple(entry[k] for k in ["date", "symbol", "target"])
        if identifier in seen:
            continue
        frame = price_panel(
            "halfday",
            entry["symbol"],
            continuous=True,
            read_bars=reader,
            schedule_table=schedule,
        )
        row = frame[frame.date.eq(entry["date"]) & frame.session.eq("AM")]
        if row.empty:
            continue
        point = row.iloc[0]
        z, sigma = (
            (point.z, point.sigma)
            if entry["target"] == "B1"
            else (point.z_intraday, point.sigma_intraday)
        )
        if pd.isna(z):
            continue
        write_event(
            "outcomes.jsonl",
            dict(
                date=identifier[0],
                symbol=identifier[1],
                target=identifier[2],
                sigma=float(sigma),
                z=float(z),
                y=int(z > 1.5),
                settled_utc=actual_now,
            ),
        )
        seen.add(identifier)


def evaluate_forward():
    """Report only matched, prospectively issued observations after ten settled days."""
    predictions = pd.DataFrame(events("predictions.jsonl"))
    outcomes = pd.DataFrame(events("outcomes.jsonl"))
    if predictions.empty or outcomes.empty:
        return
    from sklearn.metrics import average_precision_score, roc_auc_score

    matched = predictions.merge(
        outcomes, on=["date", "symbol", "target"], validate="one_to_one"
    )
    rows = []
    for (phase, model, target), batch in matched.groupby(
        ["phase", "model_sha256", "target"]
    ):
        sessions = batch.date.nunique()
        if sessions < 10 or batch.y.nunique() < 2:
            continue
        rows.append(
            dict(
                phase=phase,
                model_sha256=model,
                target=target,
                sessions=sessions,
                n=len(batch),
                positive=int(batch.y.sum()),
                ROC_AUC=roc_auc_score(batch.y, batch.score),
                PR_AUC=average_precision_score(batch.y, batch.score),
                recall=float(batch.loc[batch.y.eq(1), "alert"].mean()),
            )
        )
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "evaluation.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--run-once", action="store_true")
    commands.add_argument("--settle", action="store_true")
    commands.add_argument("--status", action="store_true")
    parser.add_argument("--phase", choices=["a", "b"], default="b")
    args = parser.parse_args()
    instant = now()
    day = instant.tz_convert("America/New_York").tz_localize(None).normalize()
    schedule = current_calendar(day)
    if args.status:
        print(
            json.dumps(
                dict(
                    utc=instant,
                    day=str(day.date()),
                    predictions=len(events("predictions.jsonl")),
                    settled=len(events("outcomes.jsonl")),
                    phase_ready=(
                        output_dir(args.phase) / "frozen_manifest.json"
                    ).is_file(),
                ),
                default=str,
            )
        )
    elif args.settle:
        settle(day, schedule, instant)
        evaluate_forward()
    else:
        print(predict(day, schedule, args.phase, instant))


if __name__ == "__main__":
    main()
