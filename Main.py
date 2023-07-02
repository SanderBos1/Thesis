from datetime import datetime
from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt
from statistics import mean

from src.GC_calculation_distance import Grangercalculator_distance
from src.VAR import Var
from src.Pruning import Pruning
import scipy.stats as stats

from src.support import Investigation
from src.top_k import TopK_improved
from src.Data_Reader import DataManipulator
from sklearn import preprocessing
import pandas as pd
import time


class Granger_investigation():

    def __init__(self):
        pass

    # prepares the data to a workable time-series
    def data_analyser(self, data_string, index,  sort_string, detrend):

        datamanipulator_function = DataManipulator(data_string)
        df = datamanipulator_function.prepare(sort_string, index)
        df = datamanipulator_function.prep_sp500(df)
        if detrend:
            df = datamanipulator_function.detrend(df)
            # removes the first value, since it is Nan
            df = df.iloc[1:, :]
        # the next line can be uncommented to create a csv of the prepared dataset
        # df.to_csv('Data/sp500_nodetrending.csv')
        return df

    def top_30_sp500_improved(self, df, lag_sizes):
        # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
        features_size = 5
        features = df.columns.tolist()
        stock = list(features[0:features_size])
        df = df[features]
        for i in lag_sizes:
            features = df.columns.tolist()
            topk_stocks = TopK_improved(df, i, features)
            # parameters define how many variables you put in the casual relationships
            top_k = topk_stocks.finding_topk_granger(3, stock)
            # saves the results in a csv file
            for i in top_k:
                print(i)
            print("the end")

    def execution(self):
        lag_sizes = [1]
        start_date = '2016/01/01'
        end_date = '2016/06/01'

        # Defines the period of a timestep, in this case a day
        sort_string = "D"
        # the time-index of the dataset
        index = "Date"
        # where the dataset is stored
        data_place = "Data/sp500_stocks.csv"

        #Choose if you want to apply detrending by differencing.
        detrend = True

        df = self.data_analyser(data_place, index, sort_string, detrend)
        # prunes the dataset on the desired time period
        df = df.loc[start_date:end_date]
        df = df.dropna(axis=1)

        # z-normalizes the dataset (mean 0, var 1)
        # df = df.apply(stats.zscore)

        # The following code can be used to prune the dataset for testing purposes
        features_size = 100
        features = df.columns.tolist()
        stock = list(features[0:features_size])
        df = df[stock]


        # Calculates the window_size
        W = len(df["A"])-1


        investigation = Investigation()

        # # Aims to calculate Granger causality of requested list of pairs or tuples of variables
        # variables = list(combinations(stock, 2))
        #answer = investigation.Granger_Causality(df, 1, variables)

        # Creates intervals of the Granger causality in a specific lag size
        investigation.Count_intervals(df, [5], 2)

        #plots different time-series
        # investigation.plot_stocks(df, ["AMD", "PNC", "RJF"])

        # Aims to calculate the percantage or accuracy of the combinations that can be pruned

        amountClusters = [3, 6, 9, 12]
        taus = [0.005, 0.05, 0.5, 1, 5]
        allbipair = len(list(combinations(df.columns.tolist(), 2)))
        pruning = Pruning(df, 6, allbipair)
        pruning.verifyPruning(amountClusters, taus, allbipair)

        answer = 5
        return answer


GC = Granger_investigation()
start = time.time()
local_time_start = time.ctime(start)
print("start", local_time_start)
GC.execution()
end = time.time()
local_time_end = time.ctime(end)
print("end", local_time_end)
