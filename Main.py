import pandas as pd
from VAR import Var
from sklearn.model_selection import train_test_split

#defines how much lags you watch back
optimal_lag_a = [1, 2, 3, 4, 5, 6, 7]
optimal_lag_b = [1, 2, 3]

#defines how many points you predict
PointsAhead = 20

# conversion of csv file to dataframe
df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")

dep_var = ["TOTALDEMAND"]
#desired variables
possible_granger = ["TOTALDEMAND", "RRP"]


#var calculation desired_var
best_aic = 0
best_f = 0

for i in optimal_lag_a:
    for j in optimal_lag_b:
        VAR = Var(df, i, j, PointsAhead)
    #VAR.varPlot("SETTLEMENTDATE", dep_var)
        r, m, f, aic, b = VAR.varCalculation(dep_var, "SETTLEMENTDATE")
        if aic < best_aic or best_aic == 0:
            best_lag_a = i
            best_lag_b = j
            found_parameters = b
            best_aic = aic
            best_f = f
        print("best_parameters",i, j, best_aic, found_parameters)

#var calculation granger test
best_aic = 0
found_f = 0
for i in optimal_lag_a:
    for j in optimal_lag_b:
        VAR = Var(df, i, j, PointsAhead)
    #VAR.varPlot("SETTLEMENTDATE", dep_var)
        r, m, f, aic, b = VAR.varCalculation(possible_granger, "SETTLEMENTDATE")
        if aic < best_aic or best_aic == 0:
            found_parameters = b
            best_aic = aic
            found_f = f
            best_lag_a = i
            best_lag_b = j
        print("best paramters",i, j, best_aic, found_parameters)
print(found_f)
if found_f > best_f:
    print("there is granger causality")
else:
    print("no granger causality")

    print("best_aic", best_aic, found_parameters)

