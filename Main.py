import pandas as pd
from VAR import Var
from sklearn.model_selection import train_test_split

#defines how much lags you watch back
optimal_lag = [1, 2, 3, 4, 5, 6, 7]
#defines how many points you predict
PointsAhead = 20

# conversion of csv file to dataframe
df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")



#var calculation
best_aic = 0
for i in optimal_lag:
    VAR = Var(df, i, PointsAhead)
    r, f, m, aic, b = VAR.varCalculation("TOTALDEMAND", "SETTLEMENTDATE")
    if aic < best_aic or best_aic == 0:
        found_parameters = b
        best_aic = aic
    print("best_aic", best_aic, found_parameters)
for i in optimal_lag:
    VAR2 = Var(df, i, PointsAhead)
    VAR2.varCalculation("RRP", "SETTLEMENTDATE")
