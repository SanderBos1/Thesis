import numpy as np
from src.VAR_distance import Var_distance
from scipy.spatial import distance


class Grangercalculator_distance:

    def __init__(self, df, lag, features):
        self.df = df
        self.lag = lag
        self.features = features


    def GC_calculator(self, stocks):
        distances = []
        dist_x_y = 0

        for j in range(len(stocks)):
            data = self.df[stocks[j]].copy(deep=True)
            VAR = Var_distance(data, self.lag)
            params = VAR.var_calculation(stocks[j])
            first = data[data.columns[0]].to_numpy()
            second = data[data.columns[0]].to_numpy()
            first_converted = []
            second_converted = []
            for i in range(len(first)):
                answer = params[1] * first[i]
                first_part = first[i] - answer
                first_converted.append(first_part)
            for i in range(len(second)):
                answer = params[2] * second[i]
                second_converted.append(answer)
            if j == 0:
                a2x = second_converted
            if j == 1:
                b2y = second_converted
                dist_x_y = distance.euclidean(a2x,b2y)
            dist = distance.euclidean(first_converted, second_converted)
            distances.append([stocks[j], dist])
        lower_bound = 2 * distances[0][1] + dist_x_y + distances[1][1]
        upper_bound = abs(abs(abs(distances[1][1] -dist_x_y) - distances[0][1]) - distances[0][1])
        print(lower_bound)
        print(upper_bound)
        return distances, lower_bound, upper_bound
