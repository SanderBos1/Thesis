import pandas as pd
from VAR import Var
from sklearn.model_selection import train_test_split

#defines how much lags you watch back
optimal_lag = 5
#defines how many points you predict
PointsAhead = 20

# conversion of csv file to dataframe
df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")



#var calculation
VAR = Var(df, optimal_lag, PointsAhead)
VAR.varCalculation("TOTALDEMAND", "SETTLEMENTDATE")
VAR2 = Var(df, optimal_lag, PointsAhead)
VAR2.varCalculation("RRP", "SETTLEMENTDATE")
