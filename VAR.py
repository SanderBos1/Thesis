import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR


class Var:

    def __init__(self, data, lag):
        self.data = data
        self.lag = lag

    def var_calculation(self, variables):
        # calculates an autoregression model if only one time-series is considered
        if len(variables) == 1:
            model = AutoReg(self.data, self.lag)
            model = model.fit()
            # calculate the residuals of the model and takes its variance
            resid = model.resid
            var = np.var(resid)
        # For more than one time-serie a Vector autoregression model is calculated
        else:
            model = VAR(self.data)
            results = model.fit(self.lag)
            # calculate the residuals of the model and takes its variance
            resid = results.resid
            var = np.var(resid[variables[0]])

        return var




