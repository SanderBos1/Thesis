
from IntervalCount import CountInterval
from GC_calculation import Grangercalculator
from top_k import TopK
from Data_Reader import DataManipulator
import numpy as np
import matplotlib.pyplot as plt
from window_size import WindowSize
import math

# prepares the data to a workable time-series
def data_analyser(data_string, index,  sort_string):

    datamanipulator_function = DataManipulator(data_string)
    df = datamanipulator_function.prepare(sort_string, index)
    df = datamanipulator_function.prep_sp500(df)
    # the next line can be uncommented to create a csv of the prepared dataset
    # df.to_csv('Data/sp500_nodetrending.csv')
    # the next line applies detrending by differencing
    df = datamanipulator_function.detrend(df)
    # removes the first value, since it is Nan
    df = df.iloc[1:, :]
    return df


# applies the top_k method on the desired parameters
def top_30_sp500():
    # desired parameters
    window_sizes = [30]
    start_date = '2016-01-01'
    end_date = '2016-07-01'

    # Defines the period of a timestep, in this case a day
    sort_string = "D"
    # the time-index of the dataset
    index = "Date"
    #where the dataset is stored
    data_place = "Data/sp500_stocks.csv"
    df = data_analyser(data_place, index, sort_string)
    # prunes the dataset on the desired time period
    df = df.loc[start_date:end_date]
    # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
    # features_size = 5
    #features = df.columns.tolist()
    #more testing purposes
    features = ['AEP', 'ALK', 'CB']
    # features = list(features[0:features_size])
    df = df[features]
    for i in window_sizes:
        features = df.columns.tolist()
        topk_stocks = TopK(df, i, features)
        # parameters define how many variables you put in the casual relationships
        top_k = topk_stocks.finding_topk_granger(3)
        # saves the results in a csv file
        np.savetxt("top_k" + str(i) + ".csv",
                   top_k,
                   delimiter=", ",
                   fmt='% s')


def take_GC():
    # desired parameters
    window_sizes = [30]
    start_date = '2016-01-01'
    end_date = '2016-07-01'

    # Defines the period of a timestep, in this case a day
    sort_string = "D"
    # the time-index of the dataset
    index = "Date"
    #where the dataset is stored
    data_place = "Data/sp500_stocks.csv"
    df = data_analyser(data_place, index, sort_string)
    # prunes the dataset on the desired time period
    df = df.loc[start_date:end_date]
    df = df.dropna(axis=1)
    # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
    # features_size = 5
    # features = df.columns.tolist()
    # features = list(features[0:features_size])
    # df = df[features]
    for i in window_sizes:
        features = df.columns.tolist()
        window_experiment = WindowSize(df, i, features)
        # parameters define how many variables you put in the casual relationships
        all = window_experiment.finding_All_granger(3)
        first_below = all[0]
        second_below = all[1]
        first_up = all[-1]
        second_up = all[-2]
        first_middle = all[math.ceil(len(all) / 2)]
        second_middle = all[math.ceil(len(all) / 2) + 1]
        wanted = [first_below, second_below, first_middle, second_middle, second_up, first_up]
        # saves the results in a csv file
        np.savetxt("all_gc" + str(i) + ".csv",
                   wanted,
                   delimiter=", ",
                   fmt='% s')

def Count_intervals():
    # desired parameters
    window_sizes = [30]
    start_date = '2016-01-01'
    end_date = '2016-07-01'
    # Defines the period of a timestep, in this case a day
    sort_string = "D"
    # the time-index of the dataset
    index = "Date"
    #where the dataset is stored
    data_place = "Data/sp500_stocks.csv"
    df = data_analyser(data_place, index, sort_string)
    # prunes the dataset on the desired time period
    df = df.loc[start_date:end_date]
    df = df.dropna(axis=1)
    # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
    #features_size = 30
    #features = df.columns.tolist()
    #features = list(features[0:features_size])
    #df = df[features]
    # creates the df that determines the intervals
    for i in window_sizes:
        features = df.columns.tolist()
        interval_count = CountInterval(df, i, features)
        # parameters define how many variables you put in the casual relationships
        all = interval_count.Count_Intervals(3)
        # saves the results in a csv file
        np.savetxt("histogram" + str(i) + ".csv",
                   all,
                   delimiter=", ",
                   fmt='% s')

# use this method to plot desired features
def plot(features, df):
    df.plot(x=None, y=features, kind="line", subplots=True)
    plt.savefig("Data/plot.png")
    plt.show()

def Granger_Causality():
    # desired parameters
    window_sizes = [5, 30, 40, 50]
    start_date = '2016-01-01'
    end_date = '2016-07-01'
    # Defines the period of a timestep, in this case a day
    sort_string = "D"
    # the time-index of the dataset
    index = "Date"
    #where the dataset is stored
    data_place = "Data/sp500_stocks.csv"
    df = data_analyser(data_place, index, sort_string)
    # prunes the dataset on the desired time period
    df = df.loc[start_date:end_date]
    df = df.dropna(axis=1)
    # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
    #features_size = 30
    #features = df.columns.tolist()
    #features = list(features[0:features_size])
    #df = df[features]
    # creates the df that determines the intervals
    variables = [['AAP', 'AAPL', 'CLX'], ['A', 'ADBE', 'CDNS'], ['AAP', 'CAH', 'CHRW'], ['ALK', 'CAT', 'CI'],
                         ['AEP', 'AFL', 'CAG'], ['AEP', 'ALK', 'CB']]
    for i in window_sizes:
        features = df.columns.tolist()
        Granger_calculator = Grangercalculator(df, i, features)
        # parameters define how many variables you put in the casual relationships
        GC_Values = Granger_calculator.finding_granger(variables)
        # saves the results in a csv file
        np.savetxt("GC_values" + str(i) + ".csv",
                   GC_Values,
                   delimiter=", ", fmt='% s')

# executes the program
#top_30_sp500()
#take_GC()
#Count_intervals()
Granger_Causality()