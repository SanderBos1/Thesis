import numpy as np
from itertools import combinations
from src.VAR import Var
from pyspark.sql import SparkSession


class CountInterval:
    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features

    def GC_calculator(self, dep_var):
        variances = []

        # Calculates the variance of the univariate model
        data = self.df[dep_var[0]].copy(deep=True)
        var_model = Var(data, self.lag)
        variance = var_model.var_univariate()
        variances.append(variance)

        # Calculates the Granger Causality of the multivariate model (or bivariate)
        current = list(dep_var)
        data = self.df[current].copy(deep=True)
        var_model = Var(data, self.lag)
        variance = var_model.var_calculation(current)
        score = np.log(variances[0] / variance)
        return score

    def Count_Intervals(self, nr_comb):
        # Makes a list of all combinations of stocks and creates a list
        granger_variables = list(combinations(self.features, nr_comb))

        # Creates a SparkSession to be used and defines the number of cores the program uses
        spark = SparkSession.builder.master("local[6]").getOrCreate()
        rdd = spark.sparkContext.parallelize(granger_variables)

        # Calculates the Granger Causality of all combinations
        rdd2 = rdd.map(lambda x: self.GC_calculator(x))

        rdd3 = rdd2.histogram(10)

        return rdd3

