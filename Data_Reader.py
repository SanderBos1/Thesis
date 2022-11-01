from datetime import datetime

import pandas as pd


class datamanipulator:

    def __init__(self, csv):
        self.data = pd.read_csv(csv)

    def prepare(self, features_size, pre, period, index, start_date, end_date):
        # data preparation
        df = self.data.set_index(index)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.index = pd.to_datetime(df.index)
        df = df.loc[start_date:end_date]
        df.index = df.index.to_period(period)

        if pre == True:
            keep = ['Symbol', 'High']
            AllColumns = df.columns.tolist()
            for column in AllColumns:
                if column not in keep:
                    df = df.drop(column, axis=1)
            df = df.pivot_table(index="Date", columns='Symbol', values="High")
        features = df.columns.tolist()
        features = list(features[0:features_size])
        df = df[features]

        return df

    def prepare_allcolumns(self, features_size, index, start_date, end_date, drop):
        df = self.data.set_index(index)
        df = df.drop(drop, axis=1)
        df.index = pd.to_datetime(df.index,  format='%Y%m%d')
        df = df.loc[start_date:end_date]
        df.index = df.index.to_period("D")
        features = df.columns.tolist()
        features = list(features[0:features_size])
        df = df[features]
        return df

    def detrend(self, df):
        for column in df:
            df[column] = df[column].diff(periods=1)
        return df

