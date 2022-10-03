import pandas as pd
from VAR import Var

# defines how much lags you watch back
optimal_lag_a = [1, 2, 3, 4, 5, 6, 7]
optimal_lag_b = [0, 1, 2, 3]

# defines how many points you predict
PointsAhead = 20

# conversion of csv file to dataframe
df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")

dep_var = [["TOTALDEMAND", "RRP"], ["RRP", "TOTALDEMAND"]]
# desired variables


# var calculation desired_var
for k in dep_var:
    best_f = 0
    best_lag = 0
    best_aic = 0
    best_r = 0
    first_value = [k[0]]
    for i in optimal_lag_a:
        VAR = Var(df, i, 0)
        r,  f, aic  = VAR.varCalculation(first_value, "SETTLEMENTDATE")
        if aic < best_aic or best_aic == 0:
            best_lag = i
            best_aic = aic
            best_f = f
            best_r = r
    print("best_parameters", best_lag, best_aic, best_r)
    # var calculation granger test
    best_aic = 0
    found_f = 0
    best_lag_b = 0
    best_lag_a = 0
    best_r = 0
    for i in optimal_lag_a:
        for j in optimal_lag_b:
            VAR2 = Var(df, i, j)
            r, f, aic = VAR2.varCalculation(k, "SETTLEMENTDATE")
            if aic < best_aic or best_aic == 0:
                best_aic = aic
                found_f = f
                best_lag_a = i
                best_lag_b = j
                best_r = r
    print("best parameters", best_lag_a, best_lag_b, best_aic, best_r)
    if found_f < best_f:
        print("there is granger causality of", k)
    else:
        print("no granger causality of", k)
