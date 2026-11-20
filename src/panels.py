from pathlib import Path
import re
import numpy as np
import pandas as pd
import exchange_calendars as xc
from collect_data import DATA
from semantic import key,mask,SYSTEM

TECH=["ret1","ret2","vol5","vol20","volume_ratio","rsi14","spread"]
SEM=["SentMean","SentStd","Momentum","NegRatio","NewsShock","Interaction","ImpactMean","HighImpactRatio"]
SCHEDULE=xc.get_calendar("XNYS",start="2015-01-01",end="2026-11-21").schedule
def bars(symbol,interval):
    df=pd.read_csv(DATA/f"{symbol}_{interval}.csv",index_col="utc")
    df.index=pd.to_datetime(df.index,utc=True)
    return df.sort_index()
def price_panel(grain,symbol="NVDA",continuous=False):
    day=bars(symbol,"1d")
    day.index=day.index.tz_convert("America/New_York").tz_localize(None).normalize()
    hour=bars(symbol,"1h") if grain=="halfday" else None
    rows=[]
    for i,date in enumerate(SCHEDULE.index):
        if i<2 or date not in day.index or SCHEDULE.index[i-1] not in day.index:continue
        schedule=SCHEDULE.iloc[i]; prev=SCHEDULE.iloc[i-1]; pc=day.loc[SCHEDULE.index[i-1],"close"]
        if grain=="daily":
            d=day.loc[date]
            windows=[("D",SCHEDULE.iloc[i-2]["close"],prev["close"],schedule["close"],d,abs(np.log(d["close"]/pc)),pc)]
        else:
            if schedule.open<hour.index.min():continue
            split=schedule.open+pd.Timedelta(hours=3);windows=[]
            for session,start,end,news_start in [("AM",schedule.open,split,prev.open+pd.Timedelta(hours=3) if continuous else prev["close"]),("PM",split,schedule["close"],schedule.open)]:
                sub=hour.reindex(pd.date_range(start,end,freq="1h",inclusive="left"))
                if sub.empty or sub.isna().any().any():continue
                initial=sub.open.iloc[0]
                prior=pc if session=="AM" else initial
                changes=np.diff(np.log(np.r_[initial,sub.close]))
                rv=np.sqrt(np.sum(changes**2)+(np.log(initial/pc)**2 if session=="AM" else 0))
                d=dict(intraday_rv=np.sqrt(np.sum(changes**2)),open=initial,close=sub.close.iloc[-1],high=sub.high.max(),low=sub.low.min(),volume=sub.volume.sum())
                windows.append((session,news_start,start,end,d,rv,prior))
        for session,start,cut,end,d,rv,prior in windows:
            spread=np.log(d["high"]/d["low"])
            rows.append(dict(date=str(date.date()),session=session,start=start,cut=cut,end=end,sigma_intraday=max(d.get("intraday_rv",rv),spread/np.sqrt(4*np.log(2))),sigma=max(rv,spread/np.sqrt(4*np.log(2))),ret=np.log(d["close"]/prior),volume=d["volume"],range_now=spread,previous_close=pc))
    df=pd.DataFrame(rows)
    # All predictors use only outcomes whose bar has closed by the decision time.
    df["ret1"]=df.ret.shift();df["ret2"]=df.ret.shift(2);df["spread"]=df.range_now.shift()
    for lag in [5,20]:
        df[f"vol{lag}"]=df.groupby("session").ret.transform(lambda s:s.shift().rolling(lag).std())
    df["volume_ratio"]=df.volume.shift()/df.groupby("session").volume.transform(lambda s:s.shift(2).rolling(20).mean())
    if grain=="halfday":
        # Compare the last completed session to its own preceding 20 sessions.
        df["volume_ratio"]=(df.volume/df.groupby("session").volume.transform(lambda s:s.shift().rolling(20).mean())).shift()
    else:df["volume_ratio"]=df.volume.shift()/df.volume.shift(2).rolling(20).mean()
    gains=df.ret.clip(lower=0).shift().rolling(14).mean()
    losses=(-df.ret.clip(upper=0)).shift().rolling(14).mean()
    df["rsi14"]=100*gains/(gains+losses).replace(0,np.nan)
    df["past_mean"]=df.groupby("session").sigma.transform(lambda s:s.shift().rolling(20).mean())
    sd=df.groupby("session").sigma.transform(lambda s:s.shift().rolling(20).std())
    df["z"]=(df.sigma-df.past_mean)/sd.replace(0,np.nan)
    df["is_afternoon_session"]=(df.session=="PM").astype(int)
    history=df.groupby("session").sigma_intraday
    mu=history.transform(lambda s:s.shift().rolling(20).mean())
    std=history.transform(lambda s:s.shift().rolling(20).std())
    df["z_intraday"]=(df.sigma_intraday-mu)/std.replace(0,np.nan)
    df["sigma_lag1"]=df.sigma.shift()
    df["market_known_at"]=df.end.shift()
    return df
def load_news(name):
    files=["recent"] if name=="recent" else ["history","history_nasdaq","historical_releases"]
    df=pd.concat([pd.read_csv(DATA/f"{file}.csv") for file in files],ignore_index=True)
    df.utc=pd.to_datetime(df.utc,utc=True,format="mixed")
    return df.drop_duplicates(["utc","title"]).sort_values("utc").reset_index(drop=True)
def attach(df,news,cache,decay=.15,masked=False,system=SYSTEM):
    lex=pd.read_csv(DATA/"lexicon.csv").set_index("Word")
    texts=news.title.map(mask) if masked else news.title
    scored=pd.DataFrame([cache[key(t,system)] for t in texts])
    pol=scored.sentiment_score.to_numpy();imp=scored.impact_level.to_numpy()
    risk=(scored.impact_level.eq(3)|scored.event_type.isin(["regulatory","governance","macro_policy"])).to_numpy()
    def lexscore(row):
        headline=row.title.partition(" | ")[2] if system!=SYSTEM and " | " in row.title else row.title
        words=re.findall(r"[A-Z]+",headline.upper())
        # Master dictionary records year of admission; use entries known by preceding year.
        pos,neg=dictionaries[row.utc.year]
        return (sum(w in pos for w in words)-sum(w in neg for w in words))/max(1,len(words))
    dictionaries={y:(set(lex.index[(lex.Positive>0)&(lex.Positive<y)]),set(lex.index[(lex.Negative>0)&(lex.Negative<y)])) for y in news.utc.dt.year.unique()}
    lm=news.apply(lexscore,axis=1).to_numpy()
    out=[]
    for start,cut in zip(df.start,df.cut):
        a,b=news.utc.searchsorted([start,cut],side="right")
        if a==b:out.append([0]*8);continue
        ages=(cut-news.utc.iloc[a:b]).dt.total_seconds().to_numpy()/3600
        time_weight=np.exp(-decay*(ages-ages.min()))
        w=time_weight*scored.confidence.to_numpy()[a:b]
        out.append([b-a,np.dot(pol[a:b],w)/max(w.sum(),1e-15),pol[a:b].std(),(pol[a:b]<-.5).mean(),np.dot(imp[a:b],w)/max(w.sum(),1e-15),risk[a:b].mean(),np.average(lm[a:b],weights=time_weight),imp[a:b].max()])
    x=df.copy()
    x[["N","SentMean","SentStd","NegRatio","ImpactMean","HighImpactRatio","Lexicon","max_impact_level"]]=out
    x["has_news"]=(x.N>0).astype(int)
    x["Momentum"]=x.SentMean-x.groupby("session").SentMean.transform(lambda s:s.shift().rolling(3).mean())
    x["NewsShock"]=x.N/(1+x.groupby("session").N.transform(lambda s:s.shift().rolling(20).mean()))
    x["Interaction"]=x.SentMean.abs()*x.NewsShock
    x["y"]=(x.z>1.5).astype(int)
    x[TECH+SEM+["Lexicon"]]=x[TECH+SEM+["Lexicon"]].round(7).astype("float32")
    return x.dropna(subset=TECH+SEM+["z"]).reset_index(drop=True)


