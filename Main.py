import pandas as pd
from Granger import Granger

optimal_lag = 100

#conversion of csv file to dataframe
df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")

#set columname as index
df = df.set_index("SETTLEMENTDATE")

#drop columns that are not needed
df = df.drop(["REGION", "PERIODTYPE"], axis=1)

#get column names
features = df.columns

#loop through each lag
for i in range(1, optimal_lag + 1):
    #loop through each features
    for j in features:
        #add lag i of feature j to the dataframe
        df[f"{j}_Lag_{i}"] = df[j].shift(i)
df = df.dropna()

#extract the first variables.
y_TotalDemand = df["TOTALDEMAND"]
y_RRP = df['RRP']
df = df.drop(["TOTALDEMAND","RRP" ], axis=1)

#inser intercept column with all value of 1
df.insert(0, "Intercept", 1)

#transform into numpy array
X = df.to_numpy()
y_TotalDemand.to_numpy()
y_RRP.to_numpy()

print(df)
Gr = Granger(df)
Gr.varCalculation()
