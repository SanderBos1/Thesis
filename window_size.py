import numpy as np
from itertools import combinations
from VAR import Var
from pyspark.sql import SparkSession

class WindowSize:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features

    def GC_calculator(self, dep_var):
        variances = []

        data = self.df[dep_var[0]].copy(deep=True)
        VAR = Var(data, self.lag)
        univariate = [dep_var[0]]
        variance = VAR.var_calculation(univariate)
        variances.append(variance)

        current = list(dep_var)
        data = self.df[current].copy(deep=True)
        VAR = Var(data, self.lag)
        variance = VAR.var_calculation(current)
        score = np.log(variances[0] / variance)
        return [current, score]

    def finding_topk_granger(self, nr_comb):
        # Makes a list of all combinations of stocks and creates a list
        granger_variables = list(combinations(self.features, nr_comb))

        # creates a sparksession to be used, defines how many cores the program uses
        spark = SparkSession.builder.master("local[5]") \
        .getOrCreate()
        rdd = spark.sparkContext.parallelize(granger_variables)

        # calculates the GC of all combinations
        rdd2 = rdd.map(lambda x: self.GC_calculator(x))

        rdd3 = rdd2.sortBy(lambda x: x[1]).collect()

        return rdd3

