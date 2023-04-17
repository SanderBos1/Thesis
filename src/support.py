from itertools import combinations
from src.window_size import WindowSize
from src.IntervalCount import CountInterval
from src.GC_calculation import Grangercalculator
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
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
            # saves the results in a csv file
            print(wanted)

    # calculates a histogram of the grangercausality of desired window_size
    # you are able to choose the amount of Bins
    def Count_intervals(self, df, lag_sizes):
        # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
        #features_size = 5
        #features = df.columns.tolist()
        #features = list(features[0:features_size])
        #df = df[features]
        # creates the df that determines the intervals
        for i in lag_sizes:
            features = df.columns.tolist()
            interval_count = CountInterval(df, i, features)
            # parameters define how many variables you put in the casual relationships
            all = interval_count.Count_Intervals(3)
            # saves the results in a csv file
            np.savetxt("histogram" + str(i) + ".csv",
                       all,
                       delimiter=", ",
                       fmt='% s')

    # Calculates the GC of desired variables
    def Granger_Causality(self, df, lag_sizes):

        variables = [["A", "AAL"], ["AAP", "A"], ["AAP", "AAL"]]

        for i in lag_sizes:
            features = df.columns.tolist()
            Granger_calculator = Grangercalculator(df, i, features)
            # parameters define how many variables you put in the casual relationships
            GC_Values = Granger_calculator.finding_granger(variables)
            for j in GC_Values:
                    print(j[0])

    def plotParameters(self, df, window_size, residuals=False, params=True):

        variables = [['AIG', 'AON', 'AWK'], ['AMAT', 'CB', 'CBOE'], ['BMY', 'CHD', 'CI'], ['AFL', 'AXP', 'BK']
             , ['AIZ', 'BAC', 'CCL'], ['AAP', 'AAPL', 'CLX'], ['AIG', 'APH', 'AWK'], ['AMZN', 'BMY', 'CE'], ['A', 'ADBE', 'CDNS'],
        ['ALLE', 'BRO', 'CHD']]

        for i in variables:
            df_desired = df[i].copy(deep=True)
            model = VAR(df_desired)
            results = model.fit(window_size)

            # plot residual errors
            if residuals == True:
                params = results.resid[i[0]]
                params.plot()
                plt.title(str(i))
                plt.axhline(y=0, color="r", linestyle="-")
                plt.savefig(str(i) + "resids.png")
                plt.show()
            elif params == True:

                # needed to print coefficients for each parameter (1 is first, 2 is second in range):
                param = results.params[i[0]]
                params = []
                variable = 2
                for j in range(variable, param.size, 2):
                    params.append(param[j])
                plt.plot(params, marker="o")
                plt.axhline(y=0, color="r", linestyle="-")
                plt.title(str(i) + " " + str(variable))
                plt.savefig(str(i) + " " + str(variable) + ".png")
                plt.show()
            else:
                df.plot(x=None, y=df_desired.columns.tolist(), kind="line", subplots=True)
                plt.savefig("Data/plot.png")
                plt.show()