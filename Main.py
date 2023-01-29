from itertools import combinations

from src.top_k_improvement import TopK_improved
from src.Data_Reader import DataManipulator
from src.support import Investigation
from sklearn.neighbors import NearestNeighbors

from sklearn import preprocessing
import numpy as np
import pandas as pd
import time


class Granger_investigation():

    def __init__(self):
        pass

    # prepares the data to a workable time-series
    def data_analyser(self, data_string, index,  sort_string, detrend):

        datamanipulator_function = DataManipulator(data_string)
        df = datamanipulator_function.prepare(sort_string, index)
        df = datamanipulator_function.prep_sp500(df)
        if detrend:
            df = datamanipulator_function.detrend(df)
            # removes the first value, since it is Nan
            df = df.iloc[1:, :]
        # the next line can be uncommented to create a csv of the prepared dataset
        # df.to_csv('Data/sp500_nodetrending.csv')
        return df

    def top_30_sp500_improved(self, df, window_sizes):
        # the following lines can be uncommented to prune the dataframe. This is done for testing purposes
        features_size = 100
        features = df.columns.tolist()
        stock = list(features[0:features_size])
        df = df[features]
        for i in window_sizes:
            features = df.columns.tolist()
            topk_stocks = TopK_improved(df, i, features)
            # parameters define how many variables you put in the casual relationships
            top_k = topk_stocks.finding_topk_granger(3, stock)
            # saves the results in a csv file
            for i in top_k:
                print(i)
            print("the end")

    def znormalization(self, df, window_sizes):

        variables =[['AMGN',  'MSFT'],  ['FCX',  'WY'],   ['CFG',  'LYB'],  ['CTLT',  'MTCH'],    ['LIN',  'MPC'],
        ['CME', 'NLOK'],    ['AKAM',  'PEG'],    ['ANSS',  'EW'],   ['ABMD',  'VLO'],    ['AEP',  'ALK'],
        ['EPAM',  'STZ'],   ['ALL',  'MOH'],   ['MCK',  'ROK'], ['C',  'L'],   ['FANG',  'GPN'], ['BF-B',  'CDW'],
        ['DXC',  'GIS'],   ['DE',  'PPL'],    ['EA',  'MSFT'],    ['HSIC',  'TDY'],    ['CZR',  'PH'],  ['AEE', 'MET'],
        ['GL',  'PSA'],   ['CPT',  'VRSN'],   ['BIO',  'DFS'],   ['ATO',  'RJF'],    ['DXCM',  'GPN'],['CNC',  'EXR'],
        ['EA',  'EL'],    ['CMG',  'WAT']]
        #
        # variables =[['AMGN',  'NEE'], ['FCX',  'VRSK'], ['CFG',  'MCK'], ['CTLT',  'GNRC'], ['LIN',  'LYV'], ['CME',  'CVS'],
        # ['AKAM',  'EXPE'], ['ANSS',  'ESS'], ['ABMD',  'REGN'],['AEP',  'CB'], ['EPAM',  'PSX'], ['ALL',  'CSGP'],
        # ['MCK',  'WELL'],['C',  'GWW'],['FANG',  'VRSK'],['BF-B',  'EPAM'],['DXC',  'ETR'],['DE',  'TYL'],['EA',  'FRT'],
        # ['HSIC',  'IBM'],['CZR',  'WELL'],['AEE',  'ED'],['GL',  'KO'], ['CPT',  'INCY'], ['BIO',  'NWSA'], ['ATO',  'NFLX'],
        # ['DXCM',  'FLT'], ['CNC',  'DFS'], ['EA',  'JKHY'],['CMG',  'WHR']]

        # variables =[[ 'MSFT',  'NEE'],[ 'WY', 'VRSK'], ['LYB',  'MCK'], ['MTCH',  'GNRC'], ['MPC',  'LYV'],
        # [ 'NLOK',  'CVS'],[ 'PEG',  'EXPE'], ['EW', 'ESS' ], [ 'VLO',  'REGN'], ['ALK',  'CB'], ['STZ',  'PSX'],
        # ['MOH',  'CSGP'],['ROK',  'WELL'], ['L',  'GWW'],['GPN',  'VRSK'],['CDW',  'EPAM'],['GIS',  'ETR'], [ 'PPL',  'TYL'],
        # ['MSFT',  'FRT'],['HSIC',  'IBM'],['PH',  'WELL'],['MET',  'ED'],['PSA',  'KO'],['VRSN',  'INCY'],[ 'DFS',  'NWSA'],
        # [ 'RJF',  'NFLX'],['GPN',  'FLT'],['EXR',  'DFS'],['EL',  'JKHY'],['WAT',  'WHR'],]

        # variables =[["AMGN", "MSFT", "NEE"] ,["FCX", "VRSK", "WY"], ["CFG", "LYB", "MCK"],["CTLT","GNRC", "MTCH"],
        # ["LIN", "LYV", "MPC"],  ["CME","CVS", "NLOK"], ["AKAM","EXPE", "PEG"],["ANSS", "ESS", "EW"], ["ABMD", "REGN", "VLO"], ["AEP", "ALK", "CB"]]
        #
        # variables = [['AIG', 'AON'], ['AMAT', 'CB'], ['BMY', 'CI'], ['AFL', 'BK'],  ['AIZ', 'CCL'], ['AAP', 'CLX'], ['AIG', 'APH'],
        # ['AMZN', 'BMY'],  ['A', 'CDNS'], ['ALLE', 'CHD'],  ['AIG', 'CB'],  ['AON', 'BALL'], ['AEE', 'AES'],  ['AON', 'CE'],
        # ['BMY', 'CAH'],  ['AME', 'CB'],  ['AMAT', 'BIIB'], ['APH', 'CB'],  ['ANSS', 'BALL'],  ['ADI', 'BRO'], ['ALK', 'CME'],
        # ['ABMD', 'ATVI'], ['AEE', 'ARE'],['ALLE', 'CHD'],  ['APTV', 'CAH'], ['ALL', 'APH'],  ['AFL', 'CHTR'],  ['AIG', 'CLX'], ['AEP', 'CDW'],
        # ['BIIB', 'CMI']]

        #
        # variables = [ ['AIG', 'AWK'], ['AMAT', 'CBOE'], ['BMY', 'CHD' ], ['AFL', 'AXP'], ['AIZ', 'BAC'], ['AAP', 'AAPL'],
        # ['AIG', 'AWK'], ['AMZN', 'CE'], ['A',  'ADBE'],['ALLE', 'BRO'], ['AIG', 'CL'], ['AON', 'BR'], ['AEE', 'BBY'],
        # ['AON',  'BKNG'], ['BMY', 'CI'], ['AME', 'BXP'], ['AMAT', 'APH'], ['APH',  'AVB'], ['ANSS',  'CMG'], ['ADI', 'AJG' ],
        # ['ALK',  'BKR'], ['ABMD', 'CDNS'], ['AEE', 'CB'], ['ALLE', 'BIIB'], ['APTV',  'CB'], ['ALL', 'CHD'],['AFL',  'ARE'],
        # ['AIG',  'ATO'], ['AEP',  'AMAT'], ['BIIB',  'CDW']]

        # variables = [[ 'AON', 'AWK'], [ 'CB', 'CBOE'], ['CHD', 'CI'], [ 'AXP', 'BK'], ['BAC', 'CCL'], ['AAPL', 'CLX'],
        # ['APH', 'AWK'], [ 'BMY', 'CE'], [ 'ADBE' ,'CDNS'], [ 'BRO', 'CHD'], [ 'CB' ,'CL'], [ 'BALL', 'BR'],
        # [ 'AES', 'BBY'], [ 'BKNG', 'CE'], [ 'CAH', 'CI'], [ 'BXP', 'CB'], ['APH', 'BIIB'], [ 'AVB', 'CB'], [ 'BALL', 'CMG'],
        # ['AJG', 'BRO'], ['BKR', 'CME'], ['ATVI', 'CDNS'], [ 'ARE', 'CB'], ['BIIB', 'CHD'], [ 'CAH', 'CB'],
        # [ 'APH', 'CHD'], [ 'ARE', 'CHTR'], [ 'ATO', 'CLX'], ['AMAT','CDW'], [ 'CDW', 'CMI']]

        # variables = [['AIG', 'AON', 'AWK'], ['AMAT', 'CB', 'CBOE'], ['BMY', 'CHD', 'CI'], ['AFL', 'AXP', 'BK']
        #      , ['AIZ', 'BAC', 'CCL'], ['AAP', 'AAPL', 'CLX'], ['AIG', 'APH', 'AWK'], ['AMZN', 'BMY', 'CE'], ['A', 'ADBE', 'CDNS'],
        # ['ALLE', 'BRO', 'CHD']]

        array = df.to_numpy()
        standard_scaler = preprocessing.StandardScaler()
        x_scaled = standard_scaler.fit_transform(array)
        df = pd.DataFrame(x_scaled, columns=df.columns)

        # features_size = 5
        # features = df.columns.tolist()
        # features = list(features[0:features_size])
        # df = df[features]


        test =['AMGN',  'MSFT',  'FCX',  'WY',   'CFG',  'LYB',  'CTLT',  'MTCH',    'LIN',  'MPC',
        'CME', 'NLOK',  'AKAM',  'PEG', 'ANSS',  'EW', 'ABMD',  'VLO', 'AEP',  'ALK',
        'EPAM',  'STZ',   'ALL',  'MOH',   'MCK',  'ROK', 'C',  'L',   'FANG',  'GPN', 'BF-B',  'CDW',
        'DXC',  'GIS', 'DE',  'PPL', 'EA',  'MSFT',    'HSIC',  'TDY',    'CZR',  'PH',  'AEE', 'MET',
        'GL',  'PSA',   'CPT',  'VRSN',   'BIO',  'DFS', 'ATO',  'RJF',  'DXCM',  'GPN', 'CNC',  'EXR',
        'EA',  'EL',  'CMG',  'WAT']
        df = df[test]

        time_series = df.to_numpy().T
        n_neighbors = 5
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(time_series)
        distances, indices = nbrs.kneighbors(time_series)

        neighbors_stock = []
        indices = list(indices)
        for i in indices:
            neighbors = []
            for j in i:
                j = test[j]
                neighbors.append(j)
            neighbors_stock.append(neighbors)
        print(neighbors_stock)




        # #finds the distance of raw time-series
        # for i in variables:
        #     eDistance = np.linalg.norm(df[i[0]]-df[i[1]])




    def execution(self):
        window_sizes = [30]
        start_date = '2016-01-01'
        end_date = '2016-07-01'

        # Defines the period of a timestep, in this case a day
        sort_string = "D"
        # the time-index of the dataset
        index = "Date"
        # where the dataset is stored
        data_place = "Data/sp500_stocks.csv"
        detrend = False
        df = self.data_analyser(data_place, index, sort_string, detrend)
        # prunes the dataset on the desired time period
        df = df.loc[start_date:end_date]
        df = df.dropna(axis=1)
        # invest = Investigation()
        # answer = invest.top_30_sp500(df, window_sizes)
        # answer = invest.plotParameters(df, 30, True, False)
        GC = Granger_investigation()
        answer = GC.znormalization(df, window_sizes)

        return answer


GC = Granger_investigation()
start = time.time()
local_time_start = time.ctime(start)
print("start", local_time_start)
GC.execution()
end = time.time()
local_time_end = time.ctime(end)
print("end", local_time_end)
