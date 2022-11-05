import numpy as np
from itertools import combinations
from VAR import Var
from pyspark.sql import SparkSession

class topk:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features

    # function that calculates the variance of the residuals of the bi & multivariate model
    def Granger_causality(self, dep_var):
        variances = []
        scores = []

        #loops through all the columns in dep_var and changes current accordingly
        for i in range(len(dep_var)):
            current = list(dep_var[:i + 1])
            data = self.df[current].copy(deep=True)
            VAR = Var(data, self.lag)
            variance = VAR.varCalculation(current)
            variances.append(variance)
            # calculates the GC and puts it in a list with the model that is considered.
            if len(variances) > 1:
                score = np.log(variances[0]/variances[i])
                scores.append([current, score])
        # checks if  the bivariate model of the first and last value are better.
        current_rev = [dep_var[0], dep_var[2]]
        data = self.df[current_rev].copy(deep=True)
        VAR = Var(data, self.lag)
        variance = VAR.varCalculation(current_rev)
        score = np.log(variances[0]/variance)
        if score > scores[0][1]:
            scores[0] = [current_rev, score]
        return scores

    # Calculates the difference between the bivariate and multivariate model
    def difference_calc(self, x):
        difference = x[1][1] - x[0][1]
        y = [x, difference]
        return y

    def finding_topk_granger(self):
        # Makes a list of all combinations of stocks and creates a list
        granger_variables = list(combinations(self.features, 3))

        # creates a sparksession to be used
        spark = SparkSession.builder.master("local[4]") \
        .getOrCreate()
        rdd = spark.sparkContext.parallelize(granger_variables)

        # calculates the GC of bivariate and Multivariate combinations
        rdd2 = rdd.map(lambda x: self.Granger_causality(x))

        # calculates and returns an array of the difference between the combinations
        rdd3 = rdd2.map(lambda x: self.difference_calc(x))

        # takes the top 10 with the largest difference
        rdd4 = rdd3.top(10, key=lambda x: x[1])
        return rdd4