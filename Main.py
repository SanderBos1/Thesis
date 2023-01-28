from src.top_k_improvement import TopK_improved
from src.Data_Reader import DataManipulator
from src.support import Investigation
from dtw import dtw
import time


class Granger_investigation():

    def __init__(self):
        pass

    # prepares the data to a workable time-series
    def data_analyser(self, data_string, index,  sort_string, detrend = True):

        datamanipulator_function = DataManipulator(data_string)
        df = datamanipulator_function.prepare(sort_string, index)
        df = datamanipulator_function.prep_sp500(df)
        # the next line can be uncommented to create a csv of the prepared dataset
        # df.to_csv('Data/sp500_nodetrending.csv')
        # the next line applies detrending by differencing
        if detrend == True:
            df = datamanipulator_function.detrend(df)
            # removes the first value, since it is Nan
            df = df.iloc[1:, :]
        else:
            print("hai")
        return df

    def top_30_sp500_improved(self, df, window_sizes):
        # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
        features_size = 100
        features = df.columns.tolist()
        stock = list(features[0:features_size])
        df = df[features]
        for i in window_sizes:
            features = df.columns.tolist()
            topk_stocks = TopK_improved(df, i, features)
            # parameters define how many variables you put in the casual relationships
            top_k = topk_stocks.finding_topk_granger(3, stock)
            # saves the results in a csv file
            for i in top_k:
                print(i)
            print("the end")

    def execution(self):
        window_sizes = [30]
        start_date = '2016-01-01'
        end_date = '2016-07-01'

        # Defines the period of a timestep, in this case a day
        sort_string = "D"
        # the time-index of the dataset
        index = "Date"
        # where the dataset is stored
        data_place = "Data/sp500_stocks.csv"
        df = self.data_analyser(data_place, index, sort_string)
        # prunes the dataset on the desired time period
        df = df.loc[start_date:end_date]
        df = df.dropna(axis=1)
        invest = Investigation()
        answer = invest.top_30_sp500(df, window_sizes)
        #answer = invest.plotParameters(df, 30, True, False)
        return answer


GC = Granger_investigation()
start = time.time()
local_time_start = time.ctime(start)
print("start", local_time_start)
GC.execution()
end = time.time()
local_time_end = time.ctime(end)
print("end", local_time_end)
