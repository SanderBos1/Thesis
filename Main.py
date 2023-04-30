from datetime import datetime

import numpy as np

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
        end_date = '2016/05/01'

        # Defines the period of a timestep, in this case a day
        sort_string = "D"
        # the time-index of the dataset
        index = "Date"
        # where the dataset is stored
        data_place = "Data/sp500_stocks.csv"
        detrend = False
        df = self.data_analyser(data_place, index, sort_string, detrend)
        # prunes the dataset on the desired time period
        df = df.loc[start_date:end_date]
        df = df.dropna(axis=1)
        # z-normalizes the dataset (mean 0, var 1)
        df = df.apply(stats.zscore)

        ## The following code can be used to prune the dataset for testing purposes
        features_size = 10
        features = df.columns.tolist()
        stock = list(features[0:features_size])
        df = df[stock]

        # # Helps with exploring the behaviour of distances between created models
        # # Stocks are of the form (X, Z), (Y, Z),(X, Y)
        # stocks = [[["MSFT", "AMGN"],["NEE", "AMGN"],["MSFT", "NEE"]], [["WY", "FCX"],["VRSK", "FCX"],["WY", "VRSK"]],
        #          [["LYB", "CFG"],["MCK", "CFG"],["LYB", "MCK"]],[["MTCH", "CTLT"],["GNRC", "CTLT"],
        #          ["MTCH", "GNRC"]], [["MPC", "LIN"],["LYV", "LIN"],["MPC", "LYV"]],[["NLOK", "CME"],["CVS", "CME"],
        #          ["NLOK", "CVS"]], [["PEG", "AKAM"],["EXPE", "AKAM"],["PEG", "EXPE"]],
        #         [["EW", "ANSS"],["ESS", "ANSS"],["EW", "ESS"]],
        #         [["VLO", "ABMD"], ["REGN", "ABMD"], ["VLO", "REGN"]], [["ALK", "AEP"],["CB", "AEP"],["ALK", "CB"]]]
        #
        # # Calculates the window_size
        # W = len(df["A"])-1
        # print(W)
        # #
        # # calculates the lower, upper and threshold bounds.
        # GC = Grangercalculator_distance(df, 1)
        # answer = GC.GC_calculator(stocks, W, 0.005)

        # Aims to calculate Granger causality of requested pair or tuple of variables
        # investigation = Investigation()
        # answer = investigation.Granger_Causality(df, lag_sizes)

        # calculates univariate variance

        # Aims to calculate the pruning
        pruning = Pruning(df)


        clusters = pruning.clustering(3, 4)
        calculated_clusters = []
        best_univariate = []

        for i in clusters:
            answer = pruning.find_largest_univariate(i)
            best_univariate.append(answer)
            cluster_calculated = pruning.cluster_calculations(i)
            calculated_clusters.append(cluster_calculated)
        answer = pruning.pruning(calculated_clusters, best_univariate, clusters)
        return answer


GC = Granger_investigation()
start = time.time()
local_time_start = time.ctime(start)
print("start", local_time_start)
GC.execution()
end = time.time()
local_time_end = time.ctime(end)
print("end", local_time_end)
