import pandas as pd

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
df.to_pickle("Data/stocks_prepared.pkl")

