import pandas as pd

data = pd.read_html('/content/台積電 (2330.TW) 過往股價及數據 - Yahoo 財經.html')[0]
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = data[:-1]
data.to_csv('2330.TW.csv')