import itertools
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from k_means_constrained import KMeansConstrained

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

    def clustering(self, init):
        """
        Divides the time-series of the dataset into clusters using k-means.
        Args:
            init (int): The number of times the k-means algorithm will be run with different centroid seeds.
        Returns:
            list: A list of clusters, where each cluster is a list of stock names.
        """
        clusters = []
        mat = self.df.values.T
        km = KMeansConstrained(n_clusters=self.K,size_min=2, init = "k-means++")
        km.fit(mat)
        labels = km.labels_
        results = pd.DataFrame([labels]).T
        results["stock"] = self.df.columns.tolist()

        for i in range(self.K):
            cluster = results[results[0] == i]
            clusters.append(cluster["stock"].tolist())
        return clusters

    def find_largest_univariate(self, variables):
        """
        Calculates the univariate model with the lowest variance.
        Args:
            variables (list): The list of stocks to consider.
        Returns:
            str: The name of the variable with the lowest variance.
        """
        answer = 0
        best_uni = 0
        for i in variables:
            df2 = self.df[i]
            var = Var(df2, 1)
            uni_var = var.var_univariate()
            if uni_var < answer:
                answer = uni_var
                best_uni = i
        return best_uni

    def cluster_calculations(self, cluster):
        """
        Calculates the values needed for the bounds within a cluster.
        Args:
            cluster (list): The list of stocks in the cluster.
        Returns:
            list: The calculated cluster values, including the VAR model and the  l2norm
        """
        cluster_values = []
        if len(cluster) == 1:
            cluster_values.append([cluster, 0])
        else:
            for combo in combinations(cluster, 2):
                i = list(combo)
                data = self.df[i].copy(deep=True)
                VAR = Var_distance(data, 1)
                params, l2norm = VAR.var_calculation(i)

                second = data[data.columns[0]].to_numpy()
                second_converted = []
                for i in range(len(second) - 1):
                    answer = params[1] * second[i]
                    second_converted.append(answer)

                cluster_values.append([combo, l2norm])
            return cluster_values

    def pruning(self, clusters, best_univariate, stocks_of_clusters, tau):
        all_removed_bicombinations = []

        for stocks_of_cluster in stocks_of_clusters:
            if len(stocks_of_cluster) > 2:
                self.total_combinations.extend(list(combinations(stocks_of_cluster, 2)))
        list_of_clusternumbers = list(range(self.K))
        list_of_pairsof_clusters = list(combinations(list_of_clusternumbers, 2))

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
            stocks = [[[stockx, best_univariate[dependent]], [stocky, best_univariate[dependent]], [stockx, stocky]]]
            W = len(self.df["A"]) - 1
            GC = Grangercalculator_distance(self.df, 1)
            answer = GC.GC_calculator(stocks, W, tau)

            upper = answer[0][0][4] ** 2 / W
            lower = answer[0][0][2] ** 2 / W
            bound = answer[0][0][6]

            if upper < bound:
                combinations_tobe_added = list(itertools.product(stocks_of_clusters[dependent], stocks_of_clusters[independent]))
                self.total_combinations.extend(combinations_tobe_added)
            if lower > bound:
                removed_bicombinations = list(itertools.product(stocks_of_clusters[dependent], stocks_of_clusters[independent]))
                all_removed_bicombinations.extend(removed_bicombinations)
            else:
                combinations_tobe_added = list(itertools.product(stocks_of_clusters[dependent], stocks_of_clusters[independent]))
                self.total_combinations.extend(combinations_tobe_added)

        return self.total_combinations, all_removed_bicombinations, len(all_removed_bicombinations) / self.lengthallbipairs

    def verifyPruning(self, amountClusters, taus, allbipairs, accuracy):
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
                        accuracies.append(sum / len(removed_combinations))
                    percantages.append(percentage)

                all_percentages.append(percantages)
                all_accuracies.append(accuracies)

            if accuracy is True:
                average_accuracies = np.mean(all_accuracies, axis=0)
                plt.plot(taus, average_accuracies, 'o')
                plt.title("Pruning for " + str(clusteramount) + "clusters")
                plt.xlabel("Tau")
                plt.ylabel("accuracy percentage")
                plt.show()
            else:
                average_percentages = np.mean(all_percentages, axis=0)
                plt.plot(taus, average_percentages, 'o')
                plt.title("Pruning for " + str(clusteramount) + "clusters")
                plt.xlabel("Tau")
                plt.ylabel("removed percentage")
                plt.show()
