import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

class Var:

    def __init__(self, data, lag):
        self.data = data
        self.lag = lag

    def varCalculation(self, variables):

        if len(variables) == 1:
            model = AutoReg(self.data, self.lag)
            model = model.fit()
            resid = model.resid
            var = np.var(resid)
        else:
            model = VAR(self.data)
            results = model.fit(self.lag)
            resid = results.resid
            var = np.var(resid[variables[0]])

        return var




