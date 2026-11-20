import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
from sklearn.metrics import roc_curve
from analysis_models import OUT
from collect_data import events
from panels import bars
from semantic import call,client,key,read,append
def events_plot():
    ev=events();fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    records=[];summ=[]
    for ax,item,label in zip(axes.flat,ev,["美国参议院结果","美国众议院结果","董事会调查声明","CEO 变更公告"]):
        event=pd.Timestamp(item["utc"]);ref="NVDA" if item["source"]!="AP race call" else "QQQ"
        ax.axvline(0,color="black",ls="--",lw=1)
        for sym in ["NVDA","QQQ"]:
            df=bars(sym,"15m")
            df=df[df.index.minute.isin([0,15,30,45]) & (df.index.second==0)]
            block=df[(df.index>=event-pd.Timedelta(hours=2))&(df.index+pd.Timedelta(minutes=15)<=event+pd.Timedelta(hours=4))]
            block=block[(block.index+pd.Timedelta(minutes=15)>event-pd.Timedelta(hours=2))].copy()
            block["bucket"]=np.where(block.index+pd.Timedelta(minutes=15)<=event,"before",np.where(block.index>=event,"after","crosses_release"))
            block["minutes_from_release"]=(block.index-event).total_seconds()/60
            block["symbol"]=sym;block["event"]=item["title"];block["event_utc"]=event
            records.append(block.reset_index())
            pre=block.query("bucket=='before'");post=block.query("bucket=='after'")
            nextbar=df[df.index>=event].head(1)
            prev=df[df.index+pd.Timedelta(minutes=15)<=event].tail(1)
            first_return=100*(nextbar.open.iloc[0]/prev.close.iloc[0]-1) if len(prev) and len(nextbar) else np.nan
            def rv(x):
                if x.empty:return np.nan
                return np.sqrt(np.sum(np.log(x.close/x.open)**2))
            summ.append(dict(event_utc=event,title=item["title"],url=item["url"],symbol=sym,pre_bars=len(pre),post_bars=len(post),rv_pre_open_close=rv(pre),rv_post_open_close=rv(post),next_bar_utc=nextbar.index[0],first_open_vs_last_complete_close_pct=first_return))
            if len(block):
                anchor=prev.close.iloc[0]
                ax.plot(block.minutes_from_release+15,100*(block.close/anchor-1),marker=".",label=sym)
        ax.set(xlim=(-120,240),xlabel="距发布时间（分钟，点位对应柱收盘）",ylabel="相对发布前最近完整柱收盘的变动（%）",title=f"{label} · {event.tz_convert('America/New_York'):%m-%d %H:%M} 美东时间")
        if label=="CEO 变更公告":ax.axvline(42,color="#955a19",ls=":",label="09:30 早盘预警截断")
        if ax.get_legend_handles_labels()[0]:ax.legend();ax.grid(alpha=.2)
        else:ax.set_yticks([]);ax.text(.5,.55,"数据源在此窗口内无行情",transform=ax.transAxes,ha="center")
    fig.savefig(OUT/"intraday_event_study.png",dpi=160);plt.close(fig)
    pd.concat(records).to_csv(OUT/"intraday_15m_case.csv",index=False)
    pd.DataFrame(summ).to_csv(OUT/"event_windows.csv",index=False)
def plots(pred,sha,news,cache,make_briefs=True):
    recent=pred.query("dataset=='recent' and algorithm=='LightGBM'")
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for ax,grain in zip(axes[0],["halfday","daily"]):
        for g,s in recent[recent.grain.eq(grain)].groupby("group"):
            f,t,_=roc_curve(s.y,s.p);ax.plot(f,t,label=g)
        ax.plot([0,1],[0,1],ls=":",color="grey");ax.set(xlabel="False positive rate",ylabel="True positive rate",title=grain+" | walk-forward ROC");ax.legend()
    focus=sha.query("grain=='halfday' and date>='2026-11-02'").copy()
    ax=axes[1,0];ax.plot(focus.cut,focus.p,marker=".",label="D / LightGBM")
    hit=focus[focus.y.eq(1)]
    ax.scatter(hit.cut,hit.p,color="red",label="Observed abnormal")
    ax.axhline(.7,color="black",ls="--",label="Brief threshold");ax.tick_params(axis="x",rotation=25);ax.set(title="November holdout | NVDA",ylabel="Predicted probability",ylim=(-.01,1));ax.legend(fontsize=8)
    ax=axes[1,1];cols=[c for c in sha if c.startswith("phi_")]
    important=sha.query("grain=='halfday'")[cols].abs().mean().nlargest(8).sort_values()
    ax.barh([c[4:] for c in important.index],important.values);ax.set(title="Half-day out-of-sample importance",xlabel="Mean absolute SHAP (probability)")
    dashboard=fig
    # Preselected CEO case; include all signed contributions in a true waterfall.
    r=focus.query("date=='2026-11-12' and session=='AM'").iloc[0]
    vals=r[cols].astype(float).dropna().sort_values(key=abs,ascending=False)
    labels=[c[4:] for c in vals.index[:8]]+["Other features"]
    values=list(vals.iloc[:8])+[vals.iloc[8:].sum()]
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    running=r.base
    for i,v in enumerate(values):
        ax.barh(i,abs(v),left=min(running,running+v),color="#bd493b" if v>0 else "#3177a0")
        running+=v
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    ax.axvline(r.base,color="gray",ls=":");ax.axvline(r.p,color="black",ls="--")
    ax.set(title=f"NVDA Nov 12 AM | base {r.base:.3f} to forecast {r.p:.3f}",xlabel="Predicted risk probability")
    fig.savefig(OUT/"shap_waterfall.png",dpi=160);plt.close(fig)
    briefs=[];existing={x["key"]:x for x in read("brief_cache.jsonl")}
    for _,r in sha[sha.p.gt(.7)].iterrows():
        positive=r[cols].dropna().astype(float);positive=positive[positive>0].nlargest(3)
        source_news=news[r.dataset]
        stories=source_news[(source_news.utc>r.start)&(source_news.utc<=r.cut)]
        reasons=[cache[key(h)]["rationale"] for h in stories.title if cache[key(h)]["impact_level"]==3]
        inp=dict(cut=str(r.cut),p=float(r.p),features={c[4:]:round(v,6) for c,v in positive.items()},high_impact=reasons)
        k=key("brief-v1"+json.dumps(inp,ensure_ascii=False))
        if k not in existing and make_briefs:
            sys="只基于输入的当时信息和SHAP概率贡献输出json对象，brief字段为不超过150个汉字的风险简报。空新闻列表请说明无三级事件。禁止补充后续走势或事件。"
            for attempt in range(3):
                b=call(client(),sys,inp,True)["brief"]
                if 0<len(b)<=150:break
            else:raise ValueError("简报格式无效")
            existing[k]=dict(key=k,brief=b);append("brief_cache.jsonl",existing[k])
        if k in existing:briefs.append(dict(**inp,dataset=r.dataset,grain=r.grain,brief=existing[k]["brief"]))
    dashboard.set_layout_engine(None)
    dashboard.subplots_adjust(left=.10,right=.98,top=.95,bottom=.28,wspace=.30,hspace=.42)
    dashboard.text(.08,.16,"跨日高风险简报 · 美东收盘时刻",fontsize=12,weight="bold")
    for idx,brief in enumerate(briefs[-2:]):
        cut=pd.Timestamp(brief["cut"]).tz_convert("America/New_York")
        dashboard.text(.08,.105-idx*.052,f"{cut:%m-%d}  {brief['brief']}",fontsize=9)
    dashboard.savefig(OUT/"volatility_dashboard.png",dpi=160)
    plt.close(dashboard)
    pd.DataFrame(briefs,columns=["cut","dataset","grain","p","features","high_impact","brief"]).to_csv(OUT/"briefs.csv",index=False)

