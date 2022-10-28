from top_k import topk
from Data_Reader import datamanipulator
import numpy as np

#data preparation and parameter settings
features_size = 20
lag = 5
start_date = '2021-01-01'
end_date = '2021-07-01'
datamanipulator = datamanipulator()
df = datamanipulator.prepare(features_size, start_date, end_date)
df = datamanipulator.detrend(df)
df = df.dropna()


# calculation of the top 10, save in top_k.csv
features = df.columns.tolist()
topk = topk(df, lag, features)
top_10 = topk.finding_topk_granger()
np.savetxt("top_k.csv",
           top_10,
           delimiter =", ",
           fmt ='% s')