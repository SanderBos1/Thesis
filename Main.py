import pandas as pd
from VAR import Var
from sklearn.model_selection import train_test_split

#defines how much lags you watch back
optimal_lag = 10

# conversion of csv file to dataframe
df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")

# set columname as index
df = df.set_index("SETTLEMENTDATE")

# drop columns that are not needed
df = df.drop(["REGION", "PERIODTYPE"], axis=1)

# get column names
features = df.columns

# loop through each lag
for i in range(1, optimal_lag + 1):
    # loop through each of the features
    for j in features:
        # add lag i of feature j to the dataframe
        df[f"{j}_Lag_{i}"] = df[j].shift(i)
df = df.dropna()
print(df)

VAR = Var(df)
VAR.varCalculation()