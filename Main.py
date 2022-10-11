import numpy as np
import pandas as pd

import pyspark.conf
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

print(len(features))
features = features[0:features_size]
print("features", features)

for j in features:
    # loop through each of the features
    for i in range(1, lag + 1):
        # add lag i of feature j to the dataframe
            df[f"{j}_Lag_{i}"] = df[j].shift(i)
df = df.dropna()

print(df)

# function that calculates if there is Granger causality
def Granger_causality(dep_var, features, df):
    notneeded = []
    variances = []
    scores = []

    for feature in features:
            notneeded.append(feature)
    for i in range(len(dep_var)):
        current = dep_var[:i+1]
        VAR = Var(df, lag)
        notneeded.remove(dep_var[i])
        variance = VAR.varCalculation(current, notneeded)
        variances.append(variance)
        if len(variances) > 1:
            print(len(variances))
            print("this is i", i)
            score = np.log(variances[0]/variances[i])
            print("this is the score", score)
            print("this is current", current)
            scores.append([current, score])
            print("this is scores", scores)
    print("this is scores zero", scores[0])
    return scores


dep_var = []
for i in range(len(features)-2):
        dep_var.append([features[i], features[i + 1],features[i+2]])
        dep_var.append([features[i+1], features[i], features[i + 2]])

print(dep_var)


answers_spark = []
spark = SparkSession.builder.master("local[1]") \
    .appName("SparkByExamples.com").getOrCreate()
rdd=spark.sparkContext.parallelize(dep_var)
rdd2 = rdd.map(lambda x: Granger_causality(x, features, df))
for element in rdd2.collect():
    if element[0] != -1:
        answers_spark.append(element)
print(answers_spark)