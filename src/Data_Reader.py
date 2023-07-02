import pandas as pd


class DataManipulator:
    def __init__(self, csv):
        self.data = pd.read_csv(csv)

    def prepare(self, period, index):
        """
        Converts the data to a Python temporal dataset.

        Args:
            period (str): The desired frequency of the index, e.g., 'D' for daily, 'M' for monthly.
            index (str): The name of the column to set as the index.

        Returns:
            pandas.DataFrame: The prepared dataframe.
        """
        df = self.data.set_index(index)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period(period)
        return df

    def detrend(self, df):
        """
        Applies detrending to the dataframe.
        Args:
            df (pandas.DataFrame): The dataframe to detrend.
        Returns:
            pandas.DataFrame: The detrended dataframe.
        """
        for column in df:
            df[column] = df[column].diff(periods=1)
            # The following line can be uncommented to store the result in a CSV file
            # df.to_csv('Data/sp500_detrended.csv')
        return df

    def prep_sp500(self, df):
        """
        Prepares the sp500 dataset by ensuring each column belongs to a unique stock.
        Args:
            df (pandas.DataFrame): The dataframe to prepare.
        Returns:
            pandas.DataFrame: The prepared dataframe.
        """
        keep = ['Symbol', 'High']
        all_columns = df.columns.tolist()
        for column in all_columns:
            if column not in keep:
                df = df.drop(column, axis=1)
        df = df.pivot_table(index="Date", columns='Symbol', values="High")
        return df
