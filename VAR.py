import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR


class Var:

    def __init__(self, data, lag):
        self.data = data
        self.lag = lag

    def var_calculation(self, variables):
        print(variables)
        print(len(variables))
        # takes an autoregression model if only one time-series is considered
        if len(variables) == 1:
            print("do i get here")
            model = AutoReg(self.data, self.lag)
            model = model.fit()
            resid = model.resid
            var = np.var(resid)
        # otherwise it uses vector autoregression to calculate the residuals
        else:
            model = VAR(self.data)
            results = model.fit(self.lag)
            resid = results.resid
            var = np.var(resid[variables[0]])

        return var




