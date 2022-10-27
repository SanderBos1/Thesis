import pandas as pd
from pyspark.sql import SparkSession
from scipy import signal

class datamanipulator:

    def __init__(self):
        self.data = pd.read_csv("Data/sp500_stocks.csv")

    def prepare(self):
        # data preparation
        df = self.data.set_index('Date')
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period("D")
        keep = ['Symbol', 'High']
        AllColumns = df.columns.tolist()
        for column in AllColumns:
            if column not in keep:
                df = df.drop(column, axis=1)

        df = df.pivot_table(index="Date", columns='Symbol', values="High")
        df.to_pickle("Data/stocks_prepared.pkl")

    def detrend(self, df):
        for column in df:
            df[column] = df[column].diff(periods=1)
        return df

