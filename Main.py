from itertools import combinations

from src.GC_calculation import Grangercalculator
from src.GC_calculation_distance import Grangercalculator_distance
from src.top_k_improvement import TopK_improved
from src.Data_Reader import DataManipulator
from src.support import Investigation
from sklearn.neighbors import NearestNeighbors
import scipy as sp
from sklearn import preprocessing
import numpy as np
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

    def znormalization(self, df):
        array = df.to_numpy()
        standard_scaler = preprocessing.StandardScaler()
        x_scaled = standard_scaler.fit_transform(array)
        df = pd.DataFrame(x_scaled, columns=df.columns)
        return df

    def pruning_check(self, df):

        df = self.znormalization(df)
        print(df)
        features_size = 5
        features = df.columns.tolist()
        stocks = list(features[0:features_size])
        stocks = list(combinations(stocks, 2))
        print(stocks)
        Granger_calculator = Grangercalculator_distance(df, 1, features)
        # parameters define how many variables you put in the casual relationships
        GC_Values = Granger_calculator.finding_granger(stocks)
        for j in GC_Values:
            print(j[0])


    def execution(self):
        lag_sizes = [1]
        start_date = '2016-01-05'
        end_date = '2016-01-15'

        # Defines the period of a timestep, in this case a day
        sort_string = "D"
        # the time-index of the dataset
        index = "Date"
        # where the dataset is stored
        data_place = "Data/sp500_stocks.csv"
        detrend = True
        df = self.data_analyser(data_place, index, sort_string, detrend)
        # prunes the dataset on the desired time period
        df = df.loc[start_date:end_date]
        df = df.dropna(axis=1)
        # invest = Investigation()
        #answer = invest.Granger_Causality(df, lag_sizes)
        GC = Granger_investigation()
        answer = GC.pruning_check(df)
        # answer = GC.znormalization(df, lag_sizes)

        return answer


GC = Granger_investigation()
start = time.time()
local_time_start = time.ctime(start)
print("start", local_time_start)
GC.execution()
end = time.time()
local_time_end = time.ctime(end)
print("end", local_time_end)
