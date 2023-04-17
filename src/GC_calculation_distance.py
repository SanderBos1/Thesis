import numpy as np

from src.VAR_distance import Var_distance
from scipy.spatial import distance


class Grangercalculator_distance:

    def __init__(self, df, lag):
        self.df = df
        self.lag = lag

    def GC_calculator(self, stocks):
        distances = []
        dist_x_y = 0
        for j in range(len(stocks)):
            data = self.df[list(stocks[j])].copy(deep=True)
            VAR = Var_distance(data, self.lag)
            params = VAR.var_calculation(list(stocks[j]))
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
                mean_zx = (np.mean((first-first_converted)-second_converted))**2
                euclidean_zx = distance.euclidean((first - first_converted),second_converted)
            if j == 1:
                distance_yGCz = distance.euclidean((first - first_converted), second_converted)
                y_inygcZ = second_converted
            else:
                distance_xGCy = distance.euclidean((first - first_converted), second_converted)
                x_inxgcy = second_converted
        return "mean_zx", mean_zx,"euclidean_zx", euclidean_zx, "distance_yGCz", distance_yGCz, "y_inygcZ", y_inygcZ,\
               "distance_xGCy", distance_xGCy, "distance_xGCy", distance_xGCy, "x_inxgcy", x_inxgcy

