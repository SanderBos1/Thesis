import itertools
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.GC_calculation_distance import Grangercalculator_distance
from src.VAR import Var
from src.VAR_distance import Var_distance
from src.support import Investigation


class Pruning:

    def __init__(self, df, K, lengthallbipairs):
        self.df = df
        self.total_combinations = []
        self.lengthallbipairs = lengthallbipairs
        self.K = K

    # diveds the time-series of the data set into clusters
    def clustering(self, init):
        clusters = []
        # Convert DataFrame to matrix
        mat = self.df.values.T
        # Makes the clusters by using k-means
        km = KMeans(n_clusters=self.K, init="k-means++", n_init= init)
        km.fit(mat)
        # Get cluster assignment labels
        labels = km.labels_
        # Format results as a DataFrame and add the stock names to it.
        results = pd.DataFrame([labels]).T
        results["stock"] = self.df.columns.tolist()

        # attached each cluster to its own list of stocks
        for i in range(self.K):
            cluster = results[results[0] == i]
            clusters.append(cluster["stock"].tolist())
        return clusters

    # calculates the univariate model with the largest variance, since this has the largest bound.
    def find_largest_univariate(self, variables):
        answer = 0
        best_uni = 0
        for i in variables:
            df2 = self.df[i]
            var = Var(df2, 1)
            uni_var = var.var_univariate()
            if uni_var > answer:
                answer = uni_var
                best_uni = i
        return best_uni

    # calculates the values that are needed for the bounds
    def cluster_calculations(self, cluster):
        cluster_values = []
        # calculates the vector autoregression model
        if len(cluster) == 1:
            cluster_values.append([cluster, 0])
        else:
            for combo in combinations(cluster, 2):
                i = list(combo)
                data = self.df[i].copy(deep=True)
                VAR = Var_distance(data, 1)
                params, l2norm = VAR.var_calculation(i)

                #calculates the value of the past value of the independent variable multipled by the model it's coefficients
                second = data[data.columns[0]].to_numpy()
                second_converted = []
                for i in range(len(second) - 1):
                    answer = params[1] * second[i]
                    second_converted.append(answer)

                # returns the model (x GC y), the l2norm and the value mentioned above
                cluster_values.append([combo, l2norm])

            return cluster_values

    def pruning(self, clusters, best_univariate, stocks_of_clusters, tau):


        all_removed_bicombinations = []

        #adds all combinations inside the same cluster to the set that needs to be calculated.
        for stocks_of_cluster in stocks_of_clusters:
            if len(stocks_of_cluster) > 2:
                self.total_combinations.extend(list(combinations(stocks_of_cluster, 2)))

        list_of_clusternumbers = list(range(self.K))
        list_of_pairsof_clusters = list(combinations(list_of_clusternumbers, 2))
        # loops through each pair of clusters and checks if they can be removed from the combinations
        # that need to be evalued.
        for pair_of_clusters in list_of_pairsof_clusters:
            dependent = pair_of_clusters[0]
            independent = pair_of_clusters[1]
            sum = 0
            stockx = "empty"
            stocky = "empty"
            for i in clusters[independent]:
                distance = i[1]
                if distance > sum:
                    sum = distance
                    stockx = i[0][0]
                    stocky = i[0][1]
            stocks = [[[stockx, best_univariate[dependent]], [stocky,best_univariate[dependent]], [stockx, stocky]]]
            # Calculates the window_size
            W = len(self.df["A"])-1
            # calculates the lower and  upper bound of the distance.
            # Calculates the threshold for which Granger causality > tau holds.
            GC = Grangercalculator_distance(self.df, 1)
            answer = GC.GC_calculator(stocks, W, tau)

            upper = answer[0][0][4]**2/W
            lower = answer[0][0][2]**2/W
            bound = answer[0][0][6]

            if upper < bound:
                combinations_tobe_added = list(itertools.product(stocks_of_clusters[dependent],stocks_of_clusters[independent]))
                self.total_combinations.extend(combinations_tobe_added)
            if lower > bound:
                removed_bicombinations = list(itertools.product(stocks_of_clusters[dependent],stocks_of_clusters[independent]))
                all_removed_bicombinations.extend(removed_bicombinations)
            else:
                combinations_tobe_added = list(itertools.product(stocks_of_clusters[dependent],stocks_of_clusters[independent]))
                self.total_combinations.extend(combinations_tobe_added)


        return self.total_combinations, all_removed_bicombinations, len(all_removed_bicombinations)/self.lengthallbipairs


    def verifyPruning(self, amountClusters, taus, allbipairs):

        for clusteramount in amountClusters:
            all_percentages = []
            all_accuracies = []
            for i in range(5):
                percantages = []
                accuracies = []
                pruning = Pruning(self.df, clusteramount, allbipairs)
                clusters = pruning.clustering(10)
                for tau in taus:
                    calculated_clusters = []
                    best_univariate = []
                    for i in clusters:
                        answer = pruning.find_largest_univariate(i)
                        best_univariate.append(answer)
                        cluster_calculated = pruning.cluster_calculations(i)
                        calculated_clusters.append(cluster_calculated)
                    added_combinations, removed_combinations, percentage = pruning.pruning(calculated_clusters,
                                                                                       best_univariate, clusters, tau)

                    investigation = Investigation()
                    GC_real = investigation.Granger_Causality(self.df, 1, removed_combinations)
                    sum = 0
                    for i in GC_real:
                        if i[0][1] < tau:
                            sum = sum + 1
                    if len(removed_combinations) == 0:
                        accuracies.append(1)
                    else:
                        accuracies.append(sum/len(removed_combinations))
                    percantages.append(percentage)

                all_percentages.append(percantages)
                all_accuracies.append(accuracies)
                print(accuracies)

            average_percentages = np.mean(all_percentages, axis=0)
            average_accuracies = np.mean(all_accuracies, axis=0)

            # choose to plot either the accuracies or percentages.
            plt.plot(taus, average_accuracies, 'o')
            plt.title("Pruning for " + str(clusteramount) + "clusters")
            plt.xlabel("Tau")
            plt.ylabel("accuracy percentage")
            plt.show()
