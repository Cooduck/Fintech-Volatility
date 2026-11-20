"""Matched-calendar inference, extreme news-volume diagnostics and reproducibility checks."""
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from analysis_models import OUT
from collect_data import DATA
from panels import bars
from semantic import read
def main():
    p=pd.read_csv(OUT/"oos_predictions.csv")
    assert not p.duplicated(["dataset","grain","date","session","algorithm","group"]).any()
    assert (pd.to_datetime(p.last_train_end,utc=True)<=pd.to_datetime(p.cut,utc=True)).all()
    a=pd.read_csv(OUT/"aligned_session_am_pm.csv");d=pd.read_csv(OUT/"aligned_daily_decay.csv");h=pd.read_csv(OUT/"aligned_daily_decay_history.csv")
    assert set(a.date)==set(d.date)
    assert a.groupby("date").session.nunique().eq(2).all()
    tails=[];comparisons=[]
    for (ds,gr,alg),rows in p.query("group=='D'").groupby(["dataset","grain","algorithm"]):
        panel=h if ds=="history" else a if gr=="halfday" else d
        for fold,s in rows.groupby("fold"):
            training=panel[(panel.date<s.date.min())&(panel.end<=s.cut.min())]
            test=s.merge(panel[["date","session","N","NewsShock"]],on=["date","session"],validate="one_to_one")
            cutoff=training.N.quantile(.95)
            for cat,g in test.groupby(np.where(test.N>cutoff,"above_training_p95","typical")):
                tails.append(dict(dataset=ds,grain=gr,algorithm=alg,fold=fold,slice=cat,n=len(g),training_p95=cutoff,max_N=g.N.max(),p_min=g.p.min(),p_mean=g.p.mean(),p_max=g.p.max(),abnormal_rate=g.y.mean()))
    pd.DataFrame(tails).to_csv(OUT/"news_volume_tails.csv",index=False)
    def auc(y,prob):
        sel=y==1;cnt=sel.sum()
        if cnt==0 or cnt==len(y):return np.nan
        return (rankdata(prob)[sel].sum()-cnt*(cnt+1)/2)/(cnt*(len(y)-cnt))
    for alg,g in p.query("dataset=='recent'").groupby("algorithm"):
        metrics=[]
        for grain in ["halfday","daily"]:
            v=g.query("grain==@grain").pivot(index=["date","session","y"],columns="group",values="p").reset_index()
            days=[np.asarray(ix) for ix in v.groupby("date").indices.values()]
            metrics.append((v.y.to_numpy(),v.A.to_numpy(),v.D.to_numpy(),days))
        n=len(metrics[0][3]);assert n==len(metrics[1][3])
        def contrast(ids):
            delta=[]
            for y,base,full,days in metrics:
                ix=np.concatenate([days[i] for i in ids]);delta.append(auc(y[ix],full[ix])-auc(y[ix],base[ix]))
            return delta[0]-delta[1]
        rng=np.random.default_rng(400)
        diff=[contrast(((rng.integers(n,size=(n+4)//5)[:,None]+np.arange(5))%n).ravel()[:n]) for _ in range(1000)]
        lo,hi=np.nanpercentile(diff,[2.5,97.5]);comparisons.append(dict(algorithm=alg,halfday_minus_daily_delta_auc=contrast(range(n)),lo95=lo,hi95=hi))
    pd.DataFrame(comparisons).to_csv(OUT/"granularity_contrast.csv",index=False)
    sha=pd.read_csv(OUT/"shap_oos.csv");phi=sha.filter(like="phi_")
    err=(sha.base+phi.sum(axis=1)-sha.p).abs().max()
    assert err<1e-5
    assert len(pd.read_csv(OUT/"briefs.csv"))==sum(sha.p>.7)
    news=pd.read_csv(DATA/"recent.csv")
    news.utc=pd.to_datetime(news.utc,format="mixed",utc=True)
    for df in [a,d]:
        expected=[news.utc.between(l,r,inclusive="right").sum() for l,r in zip(pd.to_datetime(df.start,utc=True),pd.to_datetime(df.cut,utc=True))]
        assert np.array_equal(df.N.to_numpy(),expected)
    df=bars("NVDA","15m");release=pd.Timestamp("2026-11-12 13:48Z")
    complete=df[df.index+pd.Timedelta(minutes=15)<=release].iloc[-1].close
    opening=df.loc["2026-11-12 14:30:00+00:00","open"]
    case=a.query("date=='2026-11-12' and session=='AM'").iloc[0]
    u=read("usage.jsonl")
    out=dict(sha_max_error=float(err),scored_texts=len(read("deepseek_cache.jsonl")),calls=len(u),estimated_usd=sum(i["estimated_usd"] for i in u),ceo_last_pre_close=float(complete),ceo_regular_open=float(opening),ceo_pre_to_open_pct=float((opening/complete-1)*100),ceo_previous_daily_close=float(case.previous_close))
    (OUT/"audit.json").write_text(json.dumps(out,indent=2),encoding="utf8")
if __name__=="__main__":main()

