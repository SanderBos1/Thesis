import numpy as np

from src.VAR_distance import Var_distance
from scipy.spatial import distance


class Grangercalculator_distance:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features


    def GC_calculator_test(self, stocks):
        distances = []
        dist_x_y = 0
        print(stocks)

        for j in range(len(stocks)):
            print(stocks[j])
            data = self.df[list(stocks[j])].copy(deep=True)
            VAR = Var_distance(data, self.lag)
            params = VAR.var_calculation(stocks[j])
            first = data[data.columns[0]].to_numpy()
            second = data[data.columns[1]].to_numpy()
            first_converted = []
            second_converted = []
            for i in range(len(first)-1):
                answer = params[1] * first[i]
                first_part = first[i+1] - answer
                first_converted.append(first_part)
            for i in range(len(second)-1):
                answer = params[2] * second[i]
                second_converted.append(answer)
            if j == 0:
                a2x = second_converted
            if j == 1:
                b2y = second_converted
                dist_x_y = distance.euclidean(a2x,b2y)
            dist = distance.euclidean(first_converted, second_converted)
            distances.append([stocks[j], dist])
        upper_bound = 2 * distances[0][1] + dist_x_y + distances[1][1]
        lower_bound = abs(abs(abs(dist_x_y - distances[1][1])-distances[0][1])-distances[0][1])
        real_value = distances[2][1]
        print(distances[0][1])
        print(distances[1][1])
        print(dist_x_y)
        print(lower_bound, real_value, upper_bound)
        return distances, lower_bound, real_value, upper_bound


    def GC_calculator(self, stocks):
        distances = []
        dist_x_y = 0
        print(stocks)

        data = self.df[list(stocks)].copy(deep=True)
        VAR = Var_distance(data, self.lag)
        params = VAR.var_calculation(list(stocks))
        print(params)
        first = data[data.columns[0]].to_numpy()
        print(first)
        second = data[data.columns[1]].to_numpy()
        print(second)
        first_converted = []
        second_converted = []
        for i in range(len(first)-1):
            answer = params[1] * first[i]
            first_part = first[i+1] - answer
            first_converted.append(first_part)
            print(first_converted)
        for i in range(len(second)-1):
            answer = params[2] * second[i]
            second_converted.append(answer)
            print(second_converted)

        dist_x_y = distance.euclidean(first_converted,second_converted)
        first = np.delete(first,0)
        distance_xgcy = distance.euclidean((first - first_converted),second_converted)
        return dist_x_y, distance_xgcy

