import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR


class Var_distance:

    def __init__(self, data, lag):
        self.data = data
        self.lag = lag

    def var_calculation(self, variables):
        # For more than one time-serie a Vector autoregression model is calculated
        model = VAR(self.data)
        results = model.fit(self.lag)
        params = results.params[results.params.columns[0]].to_numpy()

        return params




