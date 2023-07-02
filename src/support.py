from src.window_size import WindowSize
from src.IntervalCount import CountInterval
from src.GC_calculation import Grangercalculator
import matplotlib.pyplot as plt
import numpy as np
import math


class Investigation:


    def __init__(self):
        pass

    # aims to calculate the low, middle and high davalue of the Granger causality.
    def take_GC(self, df, lag_sizes):
        # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
        # features_size = 5
        # features = df.columns.tolist()
        # features = list(features[0:features_size])
        # df = df[features]
        for i in lag_sizes:
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
            print(wanted)

    # Calculates a histogram of the Granger causality for the desired lag size.
    # You can choose the number of bins for the histogram.
    def Count_intervals(self, df, lag_sizes, amountOfStocks):

        for i in lag_sizes:
            features = df.columns.tolist()
            interval_count = CountInterval(df, i, features)
            # Parameters define how many variables you put in the causal relationships.
            all_intervals = interval_count.Count_Intervals(amountOfStocks)
            # Saves the results in a csv file
            np.savetxt("histogram" + str(i) + ".csv", all_intervals, delimiter=", ", fmt='% s')

    # Calculates the Granger Causality for the desired variables.
    def Granger_Causality(self, df, lag_size, variables):
        features = df.columns.tolist()
        granger_calculator = Grangercalculator(df, lag_size, features)
        gc_values = granger_calculator.finding_granger(variables)
        return gc_values

    # Plots the stocks based on the provided dataframe and columns.
    def plot_stocks(self, df, columns):
        df[columns].plot()
        plt.xlabel("Date")
        plt.ylabel("Highest Daily stock price")
        plt.tight_layout()
        plt.show()




