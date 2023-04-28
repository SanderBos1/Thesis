import numpy as np
from statsmodels.tsa.api import VAR


class Var_distance:

    def __init__(self, data, lag):
        self.data = data
        self.lag = lag

    def var_calculation(self, variables):
        # For more than one time-serie a Vector autoregression model is calculated
        model = VAR(self.data)
        results = model.fit(self.lag)

        # takes the coefficients of the var model
        params = results.params[results.params.columns[1]].to_numpy()
        resid = np.array(results.resid[variables[1]])
        coefficient = params[0]
        whole = resid+coefficient
        l2norm = np.linalg.norm((whole))
        print(np.mean(whole), "whole mean")
        return params, l2norm, whole




