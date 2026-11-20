"""Download the frozen study inputs from the original providers. All times in raw files are UTC."""
from pathlib import Path
import csv, datetime as dt, html, json, re, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
DATA.mkdir(exist_ok=True)
def fetch(url, headers=None):
    for attempt in range(4):
        try:
            with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0 university financial research",**(headers or {})}),timeout=90) as r:
                return r.read()
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)
def yahoo(symbol,interval,start,end,extended=False):
    params=dict(period1=int(pd.Timestamp(start,tz="UTC").timestamp()),period2=int(pd.Timestamp(end,tz="UTC").timestamp()),interval=interval,includePrePost=str(extended).lower())
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?"+urlencode(params)
    obj=json.loads(fetch(url))["chart"]["result"][0]
    df=pd.DataFrame(obj["indicators"]["quote"][0],index=pd.to_datetime(obj["timestamp"],unit="s",utc=True))
    df.index.name="utc"
    df.dropna(subset=["open","high","low","close"]).to_csv(DATA/f"{symbol}_{interval}.csv")
def market():
    for s in ["NVDA","QQQ"]:
        yahoo(s,"1d","2015-01-01","2026-11-20")
        yahoo(s,"1h","2024-11-21","2026-11-20")
        yahoo(s,"15m","2026-10-28","2026-11-20",True)
def recent():
    rows=[]
    api="https://developer.nvidia.com/blog/wp-json/wp/v2/posts?"
    for page in range(1,100):
        q=urlencode(dict(per_page=100,page=page,after="2024-11-19T00:00:00",before="2026-11-20T00:00:00",_fields="date_gmt,title,link"))
        block=json.loads(fetch(api+q))
        rows.extend(dict(utc=p["date_gmt"]+"Z",title=html.unescape(p["title"]["rendered"]),url=p["link"],source="NVIDIA Technical Blog") for p in block)
        if len(block)<100:break
    sub=json.loads(fetch("https://data.sec.gov/submissions/CIK0001045810.json"))["filings"]["recent"]
    for t,form,items,acc,file in zip(*(sub[k] for k in ["acceptanceDateTime","form","items","accessionNumber","primaryDocument"])):
        if "2024-11-20"<=t<"2026-11-20" and form in ["8-K","10-Q","10-K"]:
            rows.append(dict(utc=t,title=f"NVIDIA filed {form}"+(f", items {items}" if items else ""),url=f"https://www.sec.gov/Archives/edgar/data/1045810/{acc.replace('-','')}/{file}",source="SEC acceptance"))
    evt=events()
    rows.extend(evt)
    pd.DataFrame(rows).drop_duplicates(["utc","title"]).sort_values("utc").to_csv(DATA/"recent.csv",index=False)
def events():
    ap="https://www.ap.org/the-definitive-source/behind-the-news/calling-control-of-congress-in-the-2026-midterm-elections/"
    base="https://www.globenewswire.com/news-release/"
    return [dict(utc=t,title=h,url=u,source=s) for t,h,u,s in [
        ("2026-11-04T04:36:00Z","AP calls Democratic control of the US Senate",ap,"AP race call"),
        ("2026-11-04T06:24:00Z","AP calls Democratic control of the US House",ap,"AP race call"),
        ("2026-11-08T20:00:00Z","NVIDIA board initiates independent investigation into reports of a foreign intelligence effort targeting CEO Jensen Huang",base+"2026/11/08/3398017/0/en/statement-from-nvidia-board-of-directors.html","NVIDIA via GlobeNewswire"),
        ("2026-11-12T13:48:00Z","NVIDIA announces Jensen Huang resigns as CEO and Colette Kress is appointed interim CEO",base+"2026/11/12/3401635/0/en/nvidia-announces-ceo-transition.html","NVIDIA via GlobeNewswire")]]
def history():
    # First alphabetical, descending-date Benzinga block in FNSPID. Locate NVDA
    # using range reads, then collect the entire contiguous NVDA block.
    url="https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"
    def chunk(pos,size):
        return fetch(url+f"?slice={pos}-{size}",{"Range":f"bytes={pos}-{pos+size-1}"})
    pattern=rb"(?:^|\n)(\d{4}-\d\d-\d\d [^\n]{0,2000}?,[A-Z.]{1,8},https?[^\n]*)"
    def first(pos):
        b=chunk(pos,32768)
        return next(csv.reader([re.search(pattern,b).group(1).decode("utf8",errors="replace").strip()]))[2]
    lo,hi=0,286569851
    while hi-lo>32768:
        mid=(lo+hi)//2
        if first(mid)<"NVDA":lo=mid
        else:hi=mid
    rows=[]
    for i in range(30):
        pos=lo+i*524288
        for line in re.findall(pattern,chunk(pos,524288+4096)):
            a=next(csv.reader([line.decode("utf8",errors="replace").strip()]))
            if a[2]=="NVDA" and a[0][11:19]!="00:00:00":
                rows.append(dict(utc=a[0].replace(" UTC","Z"),title=a[1],url=a[3],source="FNSPID Benzinga block"))
        if first(pos+524288)>"NVDA":break
    if not rows:raise RuntimeError("FNSPID NVDA block empty")
    pd.DataFrame(rows).drop_duplicates(["utc","title"]).sort_values("utc").to_csv(DATA/"history.csv",index=False)
def lexicon():
    url="https://docs.google.com/spreadsheets/d/1P3KtWuDyQT2Eu_VwTU3D2sEeQxR9dWCS/export?format=csv"
    import io
    table=pd.read_csv(io.BytesIO(fetch(url)),usecols=["Word","Positive","Negative"])
    table.query("Positive != 0 or Negative != 0").to_csv(DATA/"lexicon.csv",index=False)
if __name__=="__main__":
    with ThreadPoolExecutor(4) as pool:
        jobs={f:pool.submit(f) for f in (recent,market,history,lexicon)}
        for f,j in jobs.items():
            try:j.result();print(f.__name__,"ok",flush=True)
            except Exception as e:print(f.__name__,"FAILED",repr(e),flush=True);raise

