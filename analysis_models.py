import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,brier_score_loss
from statsmodels.tsa.stattools import grangercausalitytests,adfuller
from statsmodels.stats.multitest import multipletests
from panels import TECH,SEM
from collect_data import ROOT
OUT=ROOT/"results"
OUT.mkdir(exist_ok=True)
def classifiers():
    return {"L2":make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)),
            "RF":RandomForestClassifier(n_estimators=140,min_samples_leaf=5,max_features=.8,n_jobs=3,random_state=24),
            "LightGBM":LGBMClassifier(n_estimators=90,learning_rate=.04,num_leaves=9,min_child_samples=18,random_state=24,verbosity=-1,n_jobs=2)}
def metrics(y,p):
    pred=np.array(p)>=.5
    return dict(n=len(y),positive=int(sum(y)),AUC=roc_auc_score(y,p) if len(set(y))>1 else np.nan,F1=f1_score(y,pred,zero_division=0),Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred,zero_division=0),Brier=brier_score_loss(y,p))
def folds(dates,recent):
    dates=sorted(set(dates))
    base=[d for d in dates if not recent or d<"2026-11-02"]
    tests=[list(x) for x in np.array_split(base[len(base)//2:],4)]
    if recent:tests.append([d for d in dates if d>="2026-11-02"])
    return tests
def fit(df,design,name,tests,tau=1.5,full=True,explain=False):
    feats=TECH+(["is_afternoon_session"] if design=="halfday" else [])
    groups={"A":feats,"B":feats+["Lexicon"],"C":feats+SEM[:6],"D":feats+SEM}
    preds=[];attributions=[]
    for fold,dates in enumerate(tests,1):
        test=df[df.date.isin(dates)].copy()
        if test.empty:continue
        train=df[(df.date<min(dates))&(df.end<=test.cut.min())].copy()
        assert train.end.max()<=test.cut.min()
        if train.z.gt(tau).nunique()<2:raise ValueError("Single-class training set")
        for group,cols in groups.items():
            if not full and group!="D":continue
            for alg,model in classifiers().items():
                if not full and alg!="LightGBM":continue
                model.fit(train[cols],train.z.gt(tau))
                p=model.predict_proba(test[cols])[:,1]
                obs=test[["date","session","cut","end","z","sigma"]].copy()
                for k,v in dict(dataset=name,grain=design,fold=fold,group=group,algorithm=alg,y=test.z.gt(tau).astype(int),p=p,train_n=len(train),last_train_end=train.end.max()).items():obs[k]=v
                preds.append(obs)
                if explain and group=="D" and alg=="LightGBM":
                    import shap
                    bg=train[cols].sample(min(len(train),80),random_state=48)
                    explainer=shap.TreeExplainer(model,data=bg,model_output="probability")
                    vals=explainer(test[cols],check_additivity=False)
                    arr=np.asarray(vals.values)
                    base=np.asarray(vals.base_values).reshape(-1)
                    err=np.max(np.abs(base+arr.sum(axis=1)-p))
                    if err>1e-5:raise ValueError(f"SHAP additivity error {err}")
                    sha=obs.copy()
                    sha["base"]=base
                    for j,col in enumerate(cols):sha["phi_"+col]=arr[:,j];sha["value_"+col]=test[col].to_numpy()
                    attributions.append(sha)
    return pd.concat(preds,ignore_index=True),pd.concat(attributions,ignore_index=True) if attributions else None
def summarize(p,by):
    return pd.DataFrame([dict(zip(by,k if isinstance(k,tuple) else [k]),**metrics(d.y,d.p)) for k,d in p.groupby(by)])
def granger(df,name,grain):
    rows=[]
    for session,block in df.groupby("session"):
        for feat in ["NewsShock","SentStd","ImpactMean"]:
            raw=block[["sigma",feat]].copy()
            ps=[adfuller(raw[c],autolag="BIC")[1] for c in raw]
            for c,p in zip(raw,ps):
                if p>=.05:raw[c]=raw[c].diff()
            tests=grangercausalitytests(raw.dropna(),maxlag=3)
            for lag,entry in tests.items():
                f,p,df2,df1=entry[0]["ssr_ftest"]
                rows.append(dict(dataset=name,grain=grain,session=session,feature=feat,lag=lag,F=f,p=p,df_num=df1,df_den=df2,adf_sigma_p=ps[0],adf_feature_p=ps[1]))
    return rows
def paired_auc(p):
    rows=[];rng=np.random.default_rng(4242)
    # Paired moving blocks of five trading dates preserve within-day observations.
    for (ds,gr,al),block in p.groupby(["dataset","grain","algorithm"]):
        piv=block.pivot(index=["date","session","y"],columns="group",values="p").reset_index()
        ids=[np.asarray(v) for v in piv.groupby("date").indices.values()]
        y=piv.y.to_numpy();probs=piv[list("ABCD")].to_numpy()
        def aucs(ix):
            from scipy.stats import rankdata
            ys=y[ix].astype(bool);a=ys.sum();b=len(ix)-a
            if not a*b:return np.full(4,np.nan)
            ranks=rankdata(probs[ix],axis=0)
            return (ranks[ys].sum(axis=0)-a*(a+1)/2)/(a*b)
        deltas=[]
        for _ in range(500):
            starts=rng.integers(len(ids),size=(len(ids)+4)//5)
            sample=((starts[:,None]+np.arange(5))%len(ids)).ravel()[:len(ids)]
            auc=aucs(np.concatenate([ids[i] for i in sample]));deltas.append(auc[3]-auc[:3])
        ci=np.nanpercentile(deltas,[2.5,97.5],axis=0);point=aucs(np.arange(len(y)))
        for j,g in enumerate("ABC"):rows.append(dict(dataset=ds,grain=gr,algorithm=al,contrast="D-"+g,delta_auc=point[3]-point[j],lo95=ci[0,j],hi95=ci[1,j]))
    return pd.DataFrame(rows)

