import numpy as np
from itertools import combinations
from VAR import Var
from pyspark.sql import SparkSession

class TopK:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features

    # function that calculates the variance of the residuals of the bi & multivariate model
    def Granger_causality(self, dep_var):
        variances = []
        scores = []

        # calculates the variance of the univariate model

        data = self.df[dep_var[0]].copy(deep=True)
        VAR = Var(data, self.lag)
        univariate = [dep_var[0]]
        variance = VAR.var_calculation(univariate)
        variances.append(variance)

        # calculates the score of all possible bivariate model and takes the highest

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

    # Calculates the difference between the bivariate and multivariate model

    def difference_calc(self, x):
        first = x[0]
        second = x[-1]
        difference = second[1] - first[1]
        y = [x, difference]
        return y

    def finding_topk_granger(self, nr_comb):
        # Makes a list of all combinations of stocks and creates a list
        granger_variables = list(combinations(self.features, nr_comb))

        # creates a sparksession to be used
        spark = SparkSession.builder.master("local[5]") \
        .getOrCreate()
        rdd = spark.sparkContext.parallelize(granger_variables)

        # calculates the GC of bivariate and Multivariate combinations
        rdd2 = rdd.map(lambda x: self.Granger_causality(x))

        # calculates and returns an array of the difference between the combinations
        rdd3 = rdd2.map(lambda x: self.difference_calc(x))

        # takes the top 10 with the largest difference
        rdd4 = rdd3.top(30, key=lambda x: x[1])
        return rdd4
