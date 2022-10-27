import pandas as pd
from top_k import topk
from Data_Reader import datamanipulator
df = pd.read_pickle("Data/stocks_prepared.pkl")
features_size = 5
features = df.columns.tolist()
features = list(features[0:features_size])
df = df[features]
df = df.dropna()

lag = 5
datamanipulator = datamanipulator()
df = datamanipulator.detrend(df)
df = df.dropna()
topk = topk(df, lag, features)
top_10 = topk.finding_topk_granger()
print(top_10)