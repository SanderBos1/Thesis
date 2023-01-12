import numpy as np
from src.VAR import Var
from pyspark.sql import SparkSession


class Grangercalculator:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features

    def GC_calculator(self, dep_var):
        variances = []
        scores = []
        # calculates the variance of the univariate model
        data = self.df[dep_var[0]].copy(deep=True)
        VAR = Var(data, self.lag)
        univariate = [dep_var[0]]
        variance = VAR.var_calculation(univariate)
        variances.append(variance)
        # calculates the GC of all possible bivariate model and takes the highest

        for i in range(1, len(dep_var)):
            bivariate = []
            bivariate.append(dep_var[0])
            bivariate.append(dep_var[i])
            data = self.df[bivariate].copy(deep=True)
            VAR = Var(data, self.lag)
            variance = VAR.var_calculation(bivariate)
            # calculates the GC and puts it in a list
            score = np.log(variances[0] / variance)
            if len(scores) == 0:
                scores.append([bivariate, score])
            elif scores[0][1] < score:
                scores.pop(0)
                scores.append([bivariate, score])

        # calculates the GC value of the multivariate models
        if len(dep_var) > 2:
            for i in range(3, len(dep_var)+1):
                current = list(dep_var[:i])
                data = self.df[current].copy(deep=True)
                VAR = Var(data, self.lag)
                variance = VAR.var_calculation(current)
                score = np.log(variances[0] / variance)
                scores.append([current, score])
        return scores

    def finding_granger(self, stocks):
        # Makes a list of all combinations of stocks and creates a list
        granger_variables = stocks

        # creates a sparksession to be used, defines how many cores the program uses
        spark = SparkSession.builder.master("local[*]") \
        .getOrCreate()
        rdd = spark.sparkContext.parallelize(granger_variables)

        # calculates the GC of all combinations
        rdd2 = rdd.map(lambda x: self.GC_calculator(x))
        return rdd2.collect()
