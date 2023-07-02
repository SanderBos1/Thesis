import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR


class Var:
    def __init__(self, data, lag):
        self.data = data
        self.lag = lag

    def var_univariate(self):
        model = AutoReg(self.data, self.lag)
        model = model.fit()
        # Calculate the residuals of the model and take its variance
        resid = model.resid
        var = np.var(resid)
        return var

    def var_calculation(self, variables):
        # For more than one time series, a Vector Autoregression (VAR) model is calculated
        model = VAR(self.data)
        results = model.fit(self.lag)
        # Calculate the residuals of the model and take the variance of the specified variables
        resid = results.resid
        var = np.var(resid[variables[0]])
        return var




