from itertools import combinations

from src.Pruning import Pruning
from src.support import Investigation
from src.top_k import TopK_improved
from src.Data_Reader import DataManipulator
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
        for i in lag_sizes:
            features = df.columns.tolist()
            topk_stocks = TopK_improved(df, i, features)
            # parameters define how many variables you put in the casual relationships
            top_k = topk_stocks.finding_topk_granger(3, features)
            for i in top_k:
                print(i)
            print("the end")

    def execution(self):
        start_date = '2016/01/01'
        end_date = '2016/06/01'
        sort_string = "D"  # Defines the period of a timestep, in this case a day
        index = "Date"  # The time-index of the dataset
        data_place = "Data/sp500_stocks.csv"  # Where the dataset is stored
        detrend = True  # Choose if you want to apply detrending by differencing

        df = self.data_analyser(data_place, index, sort_string, detrend)
        df = df.loc[start_date:end_date]  # Prunes the dataset on the desired time period
        df = df.dropna(axis=1)
        # df = df.apply(stats.zscore)  # Z-normalizes the dataset (mean 0, var 1)


        # The following code can be used to prune the dataset for testing purposes
        # features_size = 100
        # features = df.columns.tolist()
        # stock = list(features[0:features_size])
        # df = df[stock]

        W = len(df["A"]) - 1 # Calculates the window_size

        investigation = Investigation() # creates an investigation class

        """ The following five experiments are used in Chapter 4"""
        """
        The following code is used to run experiment 1.
        It aims to calculate the top 30 highest numerical difference between bivariate and multivariate Granger causality
        Args:
            df: The dataset were the algorithm will calculate the top 30.
            lag_sizes: A list of lag sizes. For each lag size the top 30 difference will be calculated.
        Returns:
            List: A list of pairs and tuplse of stocks, indicating the top 30 difference in Granger causality.
        """
        # lag_sizes = [1, 5, 10, 20, 30]
        # self.top_30_sp500_improved(df, lag_sizes)

        """
        The following code is used to run experiment 2 .
        This aims to calculate the Granger causality of the top 3 found in the previous experiment.
        However, it will calculate it on a different time frame.
        This can be done by changing start_date and end_date at the beginning of this method.
        Args:
            df: The dataset on which the Granger causality will be calculated.
            lag_size: A number indicating the lag size it will calculate the model on.
            variables: A list of tuples
        Returns:
            List: A list indicating the Granger causality of the chosen variables.
        """
        # lag_size = 5
        # variables = [['AAPL', 'ABMD', 'ABT'], ['A', 'ABBV', 'ABMD']]
        # answer = investigation.Granger_Causality(df, lag_size, variables)
        # print(answer)

        """
        The code of experiment 2 can also be used to execute experiment 3.
        Keep in mind that you have to change back the start_date and end_date to the appropriate time-frame.
        """

        """
        The following code is used to run experiment 4 .
        It aims to calculate a histogram depicting the Granger causality of all pairs or tuples in the dataset.
        Args:
            df: The dataset on which the Granger causality will be calculated.
            lag_size: A number indicating the lag sizes it will calculate histograms of.
            model_size: A number depicting the amount of time-series used in the model.
                        Choose 2 for bivariate and 3 for multivariate models.
        Returns:
            List: A list indicating the Granger causality of the chosen variables.
        """
        # lag_sizes = [5, 30]
        # model_size = 2
        # investigation.Count_intervals(df, lag_sizes, model_size)

        """
        The following code is used to run experiment 5.
        It aims to find models with low, medium and high Granger causality on the selected lag_size.
        Afterwards they can be put into the code of experiment 2 to calculate the different Granger causalities.
        Args:
            df: The dataset on which the Granger causality will be calculated.
            lag_size: A number indicating the lag sizes it will calculate histograms of.
            nr_comb: A number indicating if you want to calculate it on pairs or tuples.
        Returns:
            List: A list with the corresponding stocks.
        """
        # lag_sizes = [30]
        # nr_comb = 2
        # investigation.take_GC(df, lag_sizes, nr_comb)


        """ The following two experiments are used in Chapter 5"""

        """
        The following code is used to calculate the percentage pruned
        Args:
            amountClusters: A list of amounts of clusters on which the pruning algorithm is calculated on.
            Taus: A list of thresholds.
            Accuracy: False if you want to calculate the percentaged pruned, True if you want to calculate the accuracy.
        Returns:
            Graphs that depicts the percentage pruned.
        """

        amountClusters = [3, 6, 9, 12]
        taus = [0.005, 0.05, 0.5, 1, 5]
        allbipair = len(list(combinations(df.columns.tolist(), 2)))
        pruning = Pruning(df, 10, allbipair)
        pruning.verifyPruning(amountClusters, taus, allbipair, False)

        """
        The following code is used to calculate the accuracy.
        Args:
            amountClusters: A list of amounts of clusters on which the pruning algorithm is calculated on.
            Taus: A list of thresholds.
            Accuracy: False if you want to calculate the percentaged pruned, True if you want to calculate the accuracy.
        Returns:
            Graphs that depict the percentage pruned.
        """
        amountClusters = [3, 6, 9, 12]
        taus = [0.005, 0.05, 0.5, 1, 5]
        allbipair = len(list(combinations(df.columns.tolist(), 2)))
        pruning = Pruning(df, 10, allbipair)
        pruning.verifyPruning(amountClusters, taus, allbipair, True)

GC = Granger_investigation()
start = time.time()
local_time_start = time.ctime(start)
print("start", local_time_start)
GC.execution()
end = time.time()
local_time_end = time.ctime(end)
print("end", local_time_end)
