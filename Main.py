from top_k import topk
from Data_Reader import datamanipulator
import numpy as np
import matplotlib.pyplot as plt

def data_analyser(data_string, start_date, end_date, pre, index,  sort_string):

    datamanipulator_function = datamanipulator(data_string)
    df = datamanipulator_function.prepare(features_size, pre, sort_string, index, start_date, end_date)
    df = datamanipulator_function.detrend(df)
    df = df.iloc[1:, :]
    return df

def top_k(df, lags):
    for i in lags:
        # calculation of the top 10, save in top_k.csv
        features = df.columns.tolist()
        topk_stocks = topk(df, i, features)
        top_10 = topk_stocks.finding_topk_granger()
        np.savetxt("top_k" + str(i) + ".csv",
                   top_10,
                   delimiter=", ",
                   fmt='% s')

def plot(features, df):
    df.plot(x=None, y=features, kind="line", subplots=True)
    plt.show()

#data preparation and parameter settings
lags = [1, 5, 10]
features_size = 200
pre = True
start_date = '2021-01-01'
end_date = '2021-07-01'
sort_string = "D"
index = "Date"
data_place = "Data/sp500_stocks.csv"
df = data_analyser(data_place, start_date, end_date, pre, index, sort_string)
#features = ["AAL", "CLX", "FIS"]
#plot(features, df)

top_k(df, lags)





lags_2 = [30]
features_size_2 = 5
start_date_2 = '2009-01-01'
end_date_2 = '2010-01-01'
pre_2 = False
date_place_2 = "Data/train_series_datetime.csv"
sort_string_2 = "H"
index_2 = "Date Time"
#data_analyser(date_place_2, start_date_2, end_date_2, pre_2, index_2, sort_string_2, lags_2)


