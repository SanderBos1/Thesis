import pandas as pd
from VAR import Var

# defines how much lags you watch back
windowlength = 50
lag = int(windowlength/10)

# conversion of csv file to desired data
df = pd.read_csv("Data/sp500_stocks.csv")

# data preparation

# set columname as index
df = df.set_index("Date")


keep = ['Symbol', 'High']
AllColumns = df.columns.tolist()
for column in AllColumns:
    if column not in keep:
        df = df.drop(column, axis=1)

df = df.pivot_table(index="Date", columns='Symbol', values="High")


features = df.columns.tolist()
for j in features:
    # loop through each of the features
    for i in range(1, lag + 1):
        # add lag i of feature j to the dataframe
            df[f"{j}_Lag_{i}"] = df[j].shift(i)
df = df.dropna()

print(df)
dep_var = [[["A"], ["A", "AAL"]], [["AAL"],["A", "AAL"]]]
# desired variables


# var calculation desired_var
for k in dep_var:
    best_f = 0
    best_aic = 0
    best_r = 0
    first_value = [k[0]]
    for j in k:
        aic_values = []
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
        aic_values.append(best_aic)
        print("best_parameters", j, best_aic, "the R^2 value is:", best_r)
    for i in range(len(aic_values)):
        if i < aic_values[0]:
            print("there is granger causality between", k)
        else:
            print("there is no granger causality between", k)

