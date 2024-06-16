# import pandas as pd
# import requests
import yfinance as yf
import os
import datetime
# 方法一(不推荐)
# api_key = 'ESHQSI29UX9X468E'
# proxies = {
#   'http': '127.0.0.1:7890',
#   'https': '127.0.0.1:7890',
# }
# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR&apikey={api_key}'
# r = requests.get(url, proxies=proxies)
# data = r.json()
#
# # 解析数据
# df = pd.DataFrame(data['Time Series (Digital Currency Daily)']).T
# df = df.reset_index()
# df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date', inplace=True)
# df = df.sort_index()
# df.to_csv('btc_data.csv')  # 保存为本地CSV文件


# 设置代理(如果开了代理)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


end_date = datetime.datetime.today().strftime('%Y-%m-%d')
print(f"几天是: {end_date}")
# 使用代理获取比特币数据
btc = yf.download('BTC-USD', start='2015-01-01', end=end_date)
btc.to_csv('btc_data.csv')

