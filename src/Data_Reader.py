import pandas as pd


class DataManipulator:

    def __init__(self, csv):
        self.data = pd.read_csv(csv)

    # converts the data to a python temporal dataset
    def prepare(self, period, index):
        # data preparation
        df = self.data.set_index(index)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period(period)
        return df

    # applies detreding to the dataframe
    def detrend(self, df):
        for column in df:
            df[column] = df[column].diff(periods=1)
            # the following line can be uncommented to store the result in a csv file
            # df.to_csv('Data/sp500_detrended.csv')
        return df

    # the sp500 dataset is not in desired form, so this method makes sure that each column belongs to a unique stock
    def prep_sp500(self, df):
        keep = ['Symbol', 'High']
        AllColumns = df.columns.tolist()
        for column in AllColumns:
            if column not in keep:
                df = df.drop(column, axis=1)
        df = df.pivot_table(index="Date", columns='Symbol', values="High")
        return df
