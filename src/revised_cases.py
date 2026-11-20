"""Explain dynamic alerts and replay the November release before the open."""

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from panels import attach
from collect_data import ROOT
from semantic import key, read, append, call, client

OUT = ROOT / "results" / "improved"
PROMPT = "仅根据输入的当时新闻和模型特征，写一份简明风险简报。仅返回json对象，brief字段最多150个汉字。概率为回测风险。不可推断之后走势、事件因果或给出交易建议。"


def capture(
    phase, target, fold, model, cols, train, test, p, rule, news, cache, system, market
):
    triggered = (p >= rule["brief_line"]) | (p / rule["prior"] >= 3.5)
    selected = triggered | (
        (test.date.isin(["2026-11-09", "2026-11-12"])) & (test.session == "AM")
    )
    rows = test[selected].copy()
    rows["kind"] = "scheduled"
    if target in ["B1", "B2"] and (test.date == "2026-11-12").any():
        raw = market("halfday", "NVDA")
        raw = raw[raw.cut <= pd.Timestamp("2026-11-12T14:30Z")].copy()
        for t in ["13:47", "13:49", "14:00", "14:15"]:
            cutoff = pd.Timestamp("2026-11-12T" + t + "Z")
            # Last row is AM. Predictors from market_panel are all lagged and
            # identical to the prior close state. Only as-of news may advance.
            raw.loc[raw.index[-1], "cut"] = cutoff
            row = attach(raw, news, cache, 0.25, system=system).iloc[[-1]].copy()
            row["kind"] = "event_replay"
            rows = pd.concat([rows, row], ignore_index=True)
    if rows.empty:
        return []
    explainer = shap.TreeExplainer(
        model,
        data=train[cols].sample(min(64, len(train)), random_state=24),
        model_output="probability",
    )
    vals = explainer(rows[cols], check_additivity=False)
    ps = model.predict_proba(rows[cols])[:, 1]
    err = np.abs(vals.values.sum(axis=1) + vals.base_values - ps)
    if max(err) > 1e-5:
        raise ValueError(f"SHAP reconstruction error {max(err)}")
    out = []
    for j, r in enumerate(rows.itertuples()):
        phi = pd.Series(vals.values[j], index=cols)
        top = phi[phi > 0].nlargest(3).round(7).to_dict()
        n = news[(news.utc > r.start) & (news.utc <= r.cut)]
        assert r.market_known_at <= r.cut and (n.empty or n.utc.max() <= r.cut)
        stories = []
        for rec in n.itertuples():
            s = cache[key(rec.title, system)]
            if s["impact_level"] == 3:
                stories.append(s["rationale"])
        risk = float(ps[j])
        lift = risk / rule["prior"]
        out.append(
            dict(
                fold=fold,
                date=r.date,
                session=r.session,
                cut=str(r.cut),
                target=target,
                kind=r.kind,
                p=risk,
                prior=rule["prior"],
                lift=lift,
                alert_line=rule["alert_line"],
                brief_line=rule["brief_line"],
                alert=risk >= rule["alert_line"],
                brief_trigger=risk >= rule["brief_line"] or lift >= 3.5,
                news_count=len(n),
                latest_news=str(n.utc.max()) if len(n) else None,
                market_known_at=str(r.market_known_at),
                top=json.dumps(top, ensure_ascii=False),
                reasons=json.dumps(stories, ensure_ascii=False),
                shap_error=err[j],
            )
        )
    return out


def finish_cases(phase, records, offline):
    tab = pd.DataFrame(records)
    tab.to_csv(OUT / f"{phase}_case_scores.csv", index=False)
    cache = {r["key"]: r for r in read("dynamic_briefs.jsonl")}
    briefs = []
    for r in tab[tab.brief_trigger].itertuples():
        inp = dict(
            cut=r.cut,
            target=r.target,
            risk=round(r.p, 6),
            lift=round(r.lift, 3),
            contributions=json.loads(r.top),
            news=json.loads(r.reasons),
        )
        k = key(json.dumps(inp, ensure_ascii=False), PROMPT)
        if phase == "cached":
            b = "；".join(f"{c} +{v:.3f}" for c, v in inp["contributions"].items())
        else:
            if k not in cache:
                if offline:
                    raise ValueError("动态简报尚未缓存")
                for attempt in range(3):
                    b = call(client(), PROMPT, inp, True)["brief"]
                    if isinstance(b, str) and 0 < len(b) <= 150:
                        break
                else:
                    raise ValueError("归因简报长度无效")
                cache[k] = dict(key=k, brief=b)
                append("dynamic_briefs.jsonl", cache[k])
            b = cache[k]["brief"]
        briefs.append(dict(**inp, kind=r.kind, brief=b))
    pd.DataFrame(briefs).to_csv(OUT / f"{phase}_briefs.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for target, g in tab[(tab.date == "2026-11-12") & (tab.session == "AM")].groupby(
        "target"
    ):
        g = g.sort_values("cut")
        t = pd.to_datetime(g.cut, utc=True).dt.tz_convert("America/New_York")
        ax.plot(t, g.p, "o-", label=target)
        ax.axhline(
            min(g.brief_line.iloc[0], 3.5 * g.prior.iloc[0]),
            ls=":",
            color=ax.lines[-1].get_color(),
            label=target + " brief threshold",
        )
    from matplotlib.dates import DateFormatter

    ax.axvline(
        pd.Timestamp("2026-11-12T13:48Z"), ls="--", color="grey", label="Release"
    )
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz="America/New_York"))
    ax.set(
        xlabel="New York time | 12 November 2026",
        ylabel="Frozen model score",
        title="Event replay using information available at each cutoff",
    )
    ax.legend(fontsize=8)
    fig.savefig(OUT / f"{phase}_event_replay.png", dpi=160)
    plt.close(fig)
