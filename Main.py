import numpy as np
import pandas as pd
from itertools import combinations
from VAR import Var
from pyspark.sql import SparkSession

# defines how much lags you watch back
from pyspark.sql.types import IntegerType, StringType

windowlength = 50
lag = int(windowlength/10)

# conversion of csv file to desired data

df = pd.read_pickle("Data/stocks_prepared.pkl")
features_size = 5
features = df.columns.tolist()

features = list(features[0:features_size])
df = df[features]

# function that calculates if there is Granger causality
def Granger_causality(dep_var):
    variances = []
    scores = []

    for i in range(len(dep_var)):
        current = list(dep_var[:i+1])
        data = df[current].copy(deep=True)
        VAR = Var(data, lag)
        variance = VAR.varCalculation(current)
        variances.append(variance)
        if len(variances) > 1:
            score = np.log(variances[0]/variances[i])
            scores.append([current, score])
    return scores

def difference_calc_3(x):
    difference = x[1][1] - x[0][1]
    y = [x, difference]
    return y

def finding_topk_granger_3():
    # Makes a list of all combinations of stocks and creates a list
    granger_variables = list(combinations(features, 3))
    for i in granger_variables:
        print(i)
    # creates a sparksession to be used
    spark = SparkSession.builder.master("local[1]") \
        .appName("SparkByExamples.com").getOrCreate()
    rdd = spark.sparkContext.parallelize(granger_variables)

    # calculates the GC of bivariate and Multivariate combinations
    rdd2 = rdd.map(lambda x: Granger_causality(x))

    # calculates and returns an array of the difference between the combinations
    rdd3 = rdd2.map(lambda x: difference_calc_3(x))

    #takes the top 10 with the largest difference
    rdd4 = rdd3.top(10, key=lambda x: x[1])
    return rdd4


granger = finding_topk_granger_3()
for i in granger:
    print(i)
