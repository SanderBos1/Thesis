import pandas as pd
from VAR import Var

# defines how much lags you watch back
windowlength = 50
lag = int(windowlength/10)

# conversion of csv file to desired data
df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")

# data preparation

# set columname as index
df = df.set_index("SETTLEMENTDATE")
features = df.columns.tolist()

for j in features:
    # loop through each of the features
    for i in range(1, lag + 1):
        # add lag i of feature j to the dataframe
            df[f"{j}_Lag_{i}"] = df[j].shift(i)
df = df.dropna()

dep_var = [[["TOTALDEMAND"], ["TOTALDEMAND", "RRP"]], [["RRP"],["RRP", "TOTALDEMAND"]]]
# desired variables


# var calculation desired_var
for k in dep_var:
    best_f = 0
    best_aic = 0
    best_r = 0
    first_value = [k[0]]
    for j in k:
       notwanted = []
       for i in features:
            if i not in j:
                notwanted.append(i)
       VAR = Var(df, lag)
       r,  f, aic = VAR.varCalculation(j, notwanted)
       if aic < best_aic or best_aic == 0:
            best_aic = aic
            best_f = f
            best_r = r
       print("best_parameters", lag, best_aic, best_r)
