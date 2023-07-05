import math

from src.VAR import Var
from src.VAR_distance import Var_distance
from scipy.spatial import distance


class Grangercalculator_distance:

    def __init__(self, df, lag):
        self.df = df
        self.lag = lag

    def GC_calculator(self, bi_pairs, W, tau):
        distances = []
        tau_exp = math.exp(tau)
        for pair in bi_pairs:
            formula = []
            second_value = []
            for j in range(len(pair)):
                data = self.df[list(pair[j])].copy(deep=True)
                VAR = Var_distance(data, self.lag)

                params, l2norm = VAR.var_calculation(list(pair[j]))

                # Calculates the value of the past values of the second stock * it's coefficient value in the formula.
                second = data[data.columns[0]].to_numpy()
                second_converted = []
                for i in range(len(second)-1):
                    answer = params[1] * second[i]
                    second_converted.append(answer)

                second_value.append(second_converted)
                formula.append(l2norm)
                # Calculates the variance of the residuals univariate model of Z.
                if j == 0:
                    data_uni = self.df[pair[j][1]].copy(deep=True)
                    VAR_uni = Var(data_uni, self.lag)
                    variance_uni = VAR_uni.var_univariate()
            xy = distance.euclidean(second_value[1], second_value[2])
            lower_bound = (abs(abs(abs(xy - formula[1]) - formula[2])-formula[2]))
            Upperbound = (formula[1] + xy + 2*formula[2])
            # calculates the value for which formula[0]**2/W must be smaller so that GC holds
            GC_Threshold = (variance_uni/tau_exp)
            distances.append([[tau, W, lower_bound, formula[0], Upperbound,  formula[0]**2/W, GC_Threshold], pair])
        return distances

