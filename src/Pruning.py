from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.GC_calculation_distance import Grangercalculator_distance
from src.VAR import Var
from src.VAR_distance import Var_distance


class Pruning:

    def __init__(self, df):
        self.df = df
        self.total_combinations = list(combinations(self.df.columns.tolist(), 2))


    def clustering(self, K, init):
        clusters = []

        # Convert DataFrame to matrix
        mat = self.df.values.T
        # Makes the clusters by using k-means
        km = KMeans(n_clusters=K, init="k-means++", n_init= init)
        km.fit(mat)
        # Get cluster assignment labels
        labels = km.labels_
        # Format results as a DataFrame and add the stock names to it.
        results = pd.DataFrame([labels]).T
        results["stock"] = self.df.columns.tolist()

        # attached each cluster to its own list of stocks
        for i in range(K):
            cluster = results[results[0] == i]
            clusters.append(cluster["stock"].tolist())
        print(clusters)
        return clusters

    # calculates the univariate model with the largest variance
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

    def pruning(self, clusters, best_univariate, stocks_of_clusters):

        # calculates the largest bounds between two clusters
        sum = 0
        stockx = "empty"
        stocky = "empty"

        for i in clusters[1]:
            distance = i[1]
            if distance > sum:
                sum = distance
                stockx = i[0][0]
                stocky = i[0][1]

        stocks = [[[stockx, best_univariate[0]], [stocky,best_univariate[0]], [stockx, stocky]]]
        # Calculates the window_size
        W = len(self.df["A"])-1
        #
        # calculates the lower, upper and threshold bounds.
        GC = Grangercalculator_distance(self.df, 1)
        answer = GC.GC_calculator(stocks, W, 0.5)

        bound_left = answer[0][0][2]**2/W
        bound = answer[0][0][6]

        print(bound_left)
        print(bound)
        if bound_left < bound:
            print("keep")
        else:
            print(stocks_of_clusters)
            stock = stocks_of_clusters[0]
            stock.extend(stocks_of_clusters[1])
            combinations_tobe_removed = list(combinations(stock, 2))
            print("total", self.total_combinations)
            print("these have to be removed", combinations_tobe_removed)
            list_after_removing = set(self.total_combinations) - set(combinations_tobe_removed)
            print("this is the list after removing", list_after_removing)
            print(len(list_after_removing))
            self.total_combinations = list_after_removing
            print(len(list(self.total_combinations)))
            print(len(list(combinations_tobe_removed)))
            print(len(self.total_combinations))

        return self.total_combinations


