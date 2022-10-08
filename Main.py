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

features_size = 15
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
    first = dep_var[0]
    both = dep_var.copy()
    notneeded = []
    aic_scores = []

    for feature in features:
        if feature not in both:
            notneeded.append(feature)
    VAR = Var(df, lag)
    r, f, aic = VAR.varCalculation(both, notneeded)
    aic_scores.append(aic)
    both.remove(first)
    for i in both:
        notneeded.append(i)
    VAR = Var(df, lag)
    r, f, aic = VAR.varCalculation(first, notneeded)
    aic_scores.append(aic)
    for score in aic_scores:
        if score < aic_scores[-1]:
            return dep_var
    return [-1]

dep_var = []
for i in range(len(features)-2):
        dep_var.append([features[i], features[i+1]])
        dep_var.append([features[i+1], features[i]])
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