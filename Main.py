from top_k import TopK
from Data_Reader import DataManipulator
import numpy as np
import matplotlib.pyplot as plt


# prepares the data to a workable time-series, applies detrending
def data_analyser(data_string, index,  sort_string):

    datamanipulator_function = DataManipulator(data_string)
    df = datamanipulator_function.prepare(sort_string, index)
    df = datamanipulator_function.prep_sp500(df)
    # the next line can be uncommented to create a csv of the prepared dataset
    # df.to_csv('Data/sp500_nodetrending.csv')
    df = datamanipulator_function.detrend(df)
    df = df.iloc[1:, :]
    return df


# calculates the top k differences of the GC value of bivariate and multivariate GC
def top_k(df, lags):
    for i in lags:
        features = df.columns.tolist()
        topk_stocks = TopK(df, i, features)
        # parameters define how many variables you put in the casual relationships
        top_k= topk_stocks.finding_topk_granger(3)
        np.savetxt("top_k" + str(i) + ".csv",
                   top_k,
                   delimiter=", ",
                   fmt='% s')


# use this method to plot desired features
def plot(features, df):
    df.plot(x=None, y=features, kind="line", subplots=True)
    plt.savefig("Data/plot.png")
    plt.show()


# applies the top_k method on the desired parameters
def top_30_sp500():
    # desired parameters
    lags = [30]
    start_date = '2016-01-01'
    end_date = '2016-07-01'
    sort_string = "D"
    index = "Date"
    data_place = "Data/sp500_stocks.csv"
    # applies the desired date to the prepared dataset
    df = data_analyser(data_place, index, sort_string)
    df = df.loc[start_date:end_date]
    # the following lines can be uncommented to prune the dataframe. THis is done for testing purposes
    features_size = 5
    features = df.columns.tolist()
    features = list(features[0:features_size])
    df = df[features]
    return top_k(df, lags)

top_30_sp500()