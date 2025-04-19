from pytrends.request import TrendReq
import time

games = ['The Last of Us', 'Grand Theft Auto V']
year = '2013'
goty_date = '12-07'


pytrends = TrendReq(hl='pt-BR', retries=1, backoff_factor=0.1)
pytrends.build_payload(kw_list=games, cat=41,
                       timeframe=(f'{year}-01-01 {year}-{goty_date}'))

time.sleep(5)
interest_over_time = pytrends.interest_over_time()
interest_over_time.to_csv(f'{year}-GOTY.csv', index=True)
