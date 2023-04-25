import math

import numpy as np

from src.VAR import Var
from src.VAR_distance import Var_distance
from scipy.spatial import distance


class Grangercalculator_distance:

    def __init__(self, df, lag):
        self.df = df
        self.lag = lag

    def GC_calculator(self, stocks):
        distances = []
        dist_x_y = 0
        for k in stocks:
            for j in range(len(k)):
                data = self.df[list(k[j])].copy(deep=True)
                VAR = Var_distance(data, self.lag)
                params, l2norm, whole = VAR.var_calculation(list(k[j]))
                second = data[data.columns[0]].to_numpy()
                second_converted = []
                for i in range(len(second)-1):
                    answer = params[1] * second[i]
                    second_converted.append(answer)
                # x GC Z
                if j == 0:
                    #print(stocks[j][1])
                    data_uni = self.df[k[j][1]].copy(deep=True)
                    VAR_uni = Var(data_uni, self.lag)
                    variance_uni = VAR_uni.var_univariate()
                    print("var", variance_uni)
                    mean_zx = (np.mean(whole))**2
                    print("mean_zx", mean_zx)
                    real = l2norm
                # Y GC Z
                if j == 1:
                    distance_yGCz = l2norm
                    y_inygcZ = second_converted
                # X GC Y
                if j == 2:
                    distance_xGCy = l2norm
                    x_inxgcy = second_converted

                    xy = distance.euclidean(x_inxgcy, y_inygcZ)
                    Tau = 0.05
                    Tau_exp = math.exp(Tau)
                    print("tau_exp", Tau_exp)
                    W = 60
                    lower_bound = (abs(abs(abs(xy- distance_yGCz) - distance_xGCy)-distance_xGCy))
                    Upperbound = (distance_yGCz + xy + 2*distance_xGCy)
                    GC_Threshold = (variance_uni/Tau_exp)
                    distances.append([[Tau, W, lower_bound, real, Upperbound, GC_Threshold, (real)**2/W], k])
        #print("interest", distance_yGCz, euclidean_zx,xy, distance_xGCy,  Tau_exp, mean_zx)

        return distances

