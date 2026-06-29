import pandas as pd
import yfinance as yf
import datetime

d = datetime.datetime.now()
end = d.date()
start = d.date() - datetime.timedelta(365)


stock_name = "ORCHASP.NS"

data = yf.download(stock_name, start, end)
df = pd.DataFrame(data)

df.columns = df.columns.get_level_values(0)

df['MVA'] = df['Close'].rolling(window = 20).mean()

df['Next_Close'] = df['Close'].shift(-1)
df['Future_return'] = ((df['Next_Close'] - df['Close']) / df['Close'])
df['Trade'] = (df['Future_return'] > 0.02).astype(int)

df = df.dropna()

print(df.tail)