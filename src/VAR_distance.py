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

        params = results.params[results.params.columns[1]].to_numpy()
        resid = np.array(results.resid[variables[1]])
        print("residuals_bivariate", np.var(resid))
        coefficient = params[0]
        whole = resid-coefficient
        l2norm = np.linalg.norm((whole), ord=2)

        return params, l2norm, whole




