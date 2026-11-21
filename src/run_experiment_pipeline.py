"""Replicate study from frozen inputs and per-headline cache."""
import argparse,json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from panels import price_panel,load_news,attach,mask
from semantic import score,read,key
from case_study import plots,events_plot
from analysis_models import OUT,folds,fit,summarize,paired_auc,granger

def main():
    import sys
    route=argparse.ArgumentParser(add_help=False)
    route.add_argument("--study",choices=["original","improved","v3","v4","v5"])
    chosen,remaining=route.parse_known_args()
    legacy_flags={"--ticker","--granularity","--time-decay","--intraday-case","--mask-dates","--model"}
    legacy_request=any(token.split("=",1)[0] in legacy_flags for token in remaining)
    if chosen.study=="v5":
        from v5_pipeline import main as v5
        return v5(remaining)
    if chosen.study=="v4":
        from v4_pipeline import main as v4
        return v4(remaining)
    if chosen.study=="v3":
        from v3_pipeline import main as v3
        return v3(remaining)
    if chosen.study=="improved" or (chosen.study is None and not legacy_request):
        from improved_pipeline import main as revised
        return revised(remaining)
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["demo","deepseek"],default="deepseek")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--granularity",choices=["halfday","daily","both"],default="both")
    parser.add_argument("--time-decay",type=float,default=.15)
    parser.add_argument("--model",choices=["deepseek-flash"],default="deepseek-flash")
    parser.add_argument("--ticker",choices=["NVDA"],default="NVDA")
    parser.add_argument("--intraday-case",choices=["15m"],default="15m")
    parser.add_argument("--mask-dates",action="store_true",help="also apply date masking to primary specification")
    args=parser.parse_args(remaining)
    news={name:load_news(name) for name in ["history","recent"]}
    titles=[t for n in news.values() for t in n.title]
    inputs=titles+[mask(t) for t in titles]
    estimate=score(inputs,True)
    estimate["usd_offpeak"]=(estimate["input_tokens_est"]*.15+estimate["output_tokens_est"]*.60)/1e6
    print(estimate,flush=True)
    if args.dry_run:return
    if args.mode=="demo" and estimate["uncached"]:raise SystemExit("离线复现缺少评分缓存，请先运行真实模式")
    cache=score(inputs)
    dense=price_panel("daily");half=price_panel("halfday")
    specs=[("recent","halfday",half),("recent","daily",dense),("history","daily",dense)]
    models=[];shaps=[];rob=[];stats=[]
    recent_start=half.date.min()
    for name,grain,df in specs:
        if args.granularity not in ["both",grain]:continue
        df=df[df.date.between(recent_start if name=="recent" else "2019-01-02","2026-11-19" if name=="recent" else "2023-12-29")].copy()
        # Align both recent designs to identical target dates after 20-session warmup.
        x=attach(df,news[name],cache,args.time_decay,args.mask_dates)
        if name=="recent":
            valid=set(half.dropna(subset=["z","vol20"]).date)&set(half[half.session.eq("PM")].date)&set(half[half.session.eq("AM")].date)
            # The hourly panel begins on Nov 21. All same-type histories are available by Jan 2.
            x=x[x.date.isin(valid)&x.date.ge("2025-01-02")].reset_index(drop=True)
        tests=folds(x.date,name=="recent")
        filename=("aligned_session_am_pm" if grain=="halfday" else "aligned_daily_decay")+("_history" if name=="history" else "")
        x.to_csv(OUT/(filename+".csv"),index=False)
        p,s=fit(x,grain,name,tests,explain=True)
        models.append(p)
        if s is not None:
            s=s.merge(x[["date","session","start"]],on=["date","session"],validate="many_to_one")
            shaps.append(s)
        print(name,grain,len(x),"main fitted",flush=True)
        stats.extend(granger(x,name,grain))
        for tau,decay,masked in [(1.5,args.time_decay,args.mask_dates),(1.2,args.time_decay,args.mask_dates),(1.8,args.time_decay,args.mask_dates),(1.5,0.,args.mask_dates),(1.5,.30,args.mask_dates),(1.5,args.time_decay,True)]:
            if tau==1.5 and decay==args.time_decay and masked==args.mask_dates:rep=p.query("group=='D' and algorithm=='LightGBM'")
            else:
                alt=x if decay==args.time_decay and masked==args.mask_dates else attach(df,news[name],cache,decay,masked)
                alt=alt[alt.date.isin(x.date)]
                rep,_=fit(alt,grain,name,tests,tau,full=False)
            row=summarize(rep,["dataset","grain"]).assign(tau=tau,decay=decay,masked=masked);rob.append(row)
    pred=pd.concat(models);pred.to_csv(OUT/"oos_predictions.csv",index=False)
    by=["dataset","grain","algorithm","group"]
    summarize(pred,by).to_csv(OUT/"ablation.csv",index=False)
    summarize(pred,by+["fold"]).to_csv(OUT/"fold_metrics.csv",index=False)
    paired_auc(pred).to_csv(OUT/"paired_auc.csv",index=False)
    pd.concat(rob).to_csv(OUT/"robustness.csv",index=False)
    g=pd.DataFrame(stats);g["q_BH"]=multipletests(g.p,method="fdr_bh")[1];g.to_csv(OUT/"granger.csv",index=False)
    sha=pd.concat(shaps);sha.to_csv(OUT/"shap_oos.csv",index=False)
    events_plot()
    if args.granularity=="both":plots(pred,sha,news,cache,args.mode=="deepseek")
    print("completed",flush=True)
if __name__=="__main__":main()

