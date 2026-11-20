"""Read NVDA rows from the dated Nasdaq section of FNSPID using HTTP byte ranges."""
from collect_data import DATA,fetch
import re,csv,concurrent.futures,pandas as pd
URL="https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
PAT=re.compile(rb'(?:^|\n)(?:\d+(?:\.0)?),((?:19|20)\d\d-[^\r\n]{0,2000}?,[A-Za-z0-9.^/-]{1,15},https?://[^,\n]+),')
def get(pos,size):
    return fetch(URL+f"?rangestart={pos}&bytes={size}",{"Range":f"bytes={pos}-{pos+size-1}"})
def ticker(pos):
    for size in [262144,1048576,4194304]:
        m=PAT.search(get(pos,size))
        if m:return next(csv.reader([m[1].decode(errors="replace")]))[-2]
    raise ValueError(pos)
def bound(after):
    lo,hi=9293191838,11616489798
    while hi-lo>262144:
        mid=(lo+hi)//2
        sym=ticker(mid)
        if sym<"NVDA" or (after and sym=="NVDA"):lo=mid
        else:hi=mid
    return hi+262144 if after else lo-262144
def main():
    lo=bound(False);hi=bound(True)
    print("byte range",lo,hi,flush=True)
    def extract(p):
        rows=[]
        for m in PAT.finditer(get(p,min(4194304+4096,hi-p))):
            t,h,s,u=next(csv.reader([m[1].decode(errors="replace")]))
            if s=="NVDA" and "2019"<=(t[:4])<="2023" and t[11:19]!="00:00:00":
                rows.append(dict(utc=t.replace(" UTC","Z"),title=h,url=u,source="FNSPID Nasdaq"))
        return rows
    with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=sum(pool.map(extract,range(lo,hi,4194304)),[])
    d=pd.DataFrame(rows).drop_duplicates(["utc","title"]).sort_values("utc")
    d.to_csv(DATA/"history_nasdaq.csv",index=False)
    print(d.groupby(d.utc.str[:4]).size())
if __name__=="__main__":main()

