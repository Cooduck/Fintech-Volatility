"""Supplement explicitly timed FNSPID records with dated primary issuer publications."""
import html,json
from urllib.parse import urlencode
import pandas as pd
from collect_data import DATA,fetch,ThreadPoolExecutor
def main():
    base="https://developer.nvidia.com/blog/wp-json/wp/v2/posts?"
    opts=dict(per_page=100,after="2018-12-31T23:59:59",before="2024-01-01T00:00:00",_fields="date_gmt,title,link")
    def page(p):
        obj=json.loads(fetch(base+urlencode(dict(opts,page=p))))
        return [dict(utc=i["date_gmt"]+"Z",title=html.unescape(i["title"]["rendered"]),url=i["link"],source="NVIDIA Technical Blog") for i in obj]
    with ThreadPoolExecutor(4) as pool:rows=sum(pool.map(page,range(1,25)),[])
    pd.DataFrame(rows).sort_values("utc").to_csv(DATA/"historical_releases.csv",index=False)
if __name__=="__main__":main()

