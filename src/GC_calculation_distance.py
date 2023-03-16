import numpy as np
from src.VAR_distance import Var_distance
from pyspark.sql import SparkSession
from scipy.spatial import distance


class Grangercalculator_distance:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features


    def GC_calculator(self, dep_var):
        current = list(dep_var)
        data = self.df[current].copy(deep=True)
        VAR = Var_distance(data, self.lag)
        params = VAR.var_calculation(current)
        first = data[data.columns[0]].to_numpy()
        second = data[data.columns[0]].to_numpy()
        first_converted = []
        second_converted = []
        for i in range(len(first)):
            answer = params[1] * first[i]
            first_part = first[i] - answer
            first_converted.append(first_part)
        for i in range(len(second)):
            answer = params[2] * second[i]
            second_converted.append(answer)

        dist = distance.euclidean(first_converted, second_converted)
        print(dist)
        return dist

    def finding_granger(self, stocks):
        # Makes a list of all combinations of stocks and creates a list
        spark = SparkSession.builder.master("local[5]") \
        .getOrCreate()

        rdd = spark.sparkContext.parallelize(stocks)
        # calculates the GC of all combinations
        rdd2 = rdd.map(lambda x: self.GC_calculator(x))
        return rdd2.collect()
