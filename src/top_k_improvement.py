import numpy as np
from itertools import combinations
from src.VAR import Var
from pyspark.sql import SparkSession


class TopK_improved:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features

    # calculates the variance of the univariate model
    def univariate_gc_calculator(self, dep_var):
        data = self.df[dep_var].copy(deep=True)
        VAR = Var(data, self.lag)
        univariate = [dep_var]
        variance = VAR.var_univariate(univariate)
        return [dep_var, variance]

    # function that calculates the variance of the residuals of the bi & multivariate model
    def GC_calculator(self, dep_var, univariate):
        scores = []

        univariate = [element for element in univariate if element[0] == dep_var[0]]
        var_uni = univariate[0][1]
        # calculates the GC of all possible bivariate model and takes the highest
        for i in range(1, len(dep_var)):
            bivariate = []
            bivariate.append(dep_var[0])
            bivariate.append(dep_var[i])
            data = self.df[bivariate].copy(deep=True)
            VAR = Var(data, self.lag)
            variance = VAR.var_calculation(bivariate)
            # calculates the GC and puts it in a list
            score = np.log(var_uni/variance)
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
                score = np.log(var_uni/variance)
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
        spark = SparkSession.builder.master("local[5]") \
        .getOrCreate()

        #calculate the variance of all univariate models
        univariate_variables = list(self.features)
        uni_rdd = spark.sparkContext.parallelize(univariate_variables)
        uni_rdd1 = uni_rdd.map(lambda x: self.univariate_gc_calculator(x))
        univariate = uni_rdd1.collect()


        # Makes a list of all combinations of stocks and creates a list
        granger_variables = list(combinations(self.features, nr_comb))
        # creates a sparksession to be used, defines how many cores the program uses

        rdd = spark.sparkContext.parallelize(granger_variables)

        # calculates the GC of bivariate and Multivariate combinations
        rdd2 = rdd.map(lambda x: self.GC_calculator(x, univariate))

        # calculates and returns an array of the difference between the combinations
        rdd3 = rdd2.map(lambda x: self.difference_calc(x))

        # takes the top 10 with the largest difference
        rdd4 = rdd3.top(30, key=lambda x: x[1])
        return rdd4
