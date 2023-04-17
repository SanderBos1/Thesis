from src.GC_calculation_distance import Grangercalculator_distance
from src.Pruning import Pruning
from src.support import Investigation
from src.top_k_improvement import TopK_improved
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

    def znormalization(self, df):
        array = df.to_numpy()
        standard_scaler = preprocessing.StandardScaler()
        x_scaled = standard_scaler.fit_transform(array)
        df = pd.DataFrame(x_scaled, columns=df.columns)
        return df

    def pruning_check(self, df):

        df = self.znormalization(df)
        features = df.columns.tolist()
        # (y, x), (z, y), (z, x)
        stocks = [["A", "AAL"], ["AAP", "A"], ["AAP", "AAL"]]
        Granger_calculator = Grangercalculator_distance(df, 1, features)
        # parameters define how many variables you put in the casual relationships
        GC_Values = Granger_calculator.GC_calculator_test(stocks)
        return GC_Values

    def execution(self):
        lag_sizes = [1]
        start_date = '2016-01-05'
        end_date = '2016-01-31'

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
        features_size = 20
        features = df.columns.tolist()
        stock = list(features[0:features_size])
        df = df[stock]
        #invest = Investigation()
        #answer = invest.Granger_Causality(df, lag_sizes)
        #GC = Granger_investigation()
        #answer = GC.pruning_check(df)
        df = self.znormalization(df)
        pruning = Pruning(df)
        root = pruning.set_root(df)
        pruning.HierarchicalClustering(root, 4, 3, 0.8, 5, 2)
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
