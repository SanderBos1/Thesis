import numpy as np
from src.VAR import Var
from pyspark.sql import SparkSession


class Grangercalculator:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features

    def univariate_gc_calculator(self, dep_var):
        data = self.df[dep_var].copy(deep=True)
        VAR = Var(data, self.lag)
        variance = VAR.var_univariate()
        return [dep_var, variance]

    def GC_calculator(self, dep_var, univariate):
        scores = []
        univariate = [element for element in univariate if element[0] == dep_var[0]]
        var_uni = univariate[0][1]
        # calculates the GC of all possible bivariate model and takes the highest

        current = list(dep_var)
        data = self.df[current].copy(deep=True)
        VAR = Var(data, self.lag)
        variance = VAR.var_calculation(current)
        score = np.log(var_uni/variance)
        scores.append([current, score])

        return scores

    def finding_granger(self, stocks):
        # Makes a list of all combinations of stocks and creates a list
        spark = SparkSession.builder.master("local[5]") \
        .getOrCreate()

        univariate_variables = list(self.features)
        uni_rdd = spark.sparkContext.parallelize(univariate_variables)
        uni_rdd1 = uni_rdd.map(lambda x: self.univariate_gc_calculator(x))
        univariate = uni_rdd1.collect()

        rdd = spark.sparkContext.parallelize(stocks)
        # calculates the GC of all combinations
        rdd2 = rdd.map(lambda x: self.GC_calculator(x, univariate))
        return rdd2.collect()
