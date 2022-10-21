import statistics

import numpy as np
from sklearn import linear_model
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.api import VAR

class Var:

    def __init__(self, data, lag):
        self.data = data
        self.lag = lag

    """ Estimation of VAR models
    Input:
    X: data with size [number of variables, number of observations]
    First_Values: a list of arrays of the starting values of variables we are interested in.
    Output:
    VAR_Y : Coefficient matrix
    e: the error term
    """


    def varCalculation(self, variables):

        if len(variables) == 1:
            ar_model = ar_select_order(self.data, self.lag)
            model = ar_model.model.fit()
            resid = model.resid
            var = np.var(resid)
        else:
            print("do we get here")
            y_True = self.data[variables[0]]
            model = VAR(self.data, y_True)
            model.select_order(self.lag)
            results = model.fit()
            print(results.summary())
            resid = results.resid
            print("these are the residuals", resid)
            print("check", resid[variables[0]])
            var = np.var(resid[variables[0]])
            print("this is the var", var)
        return var








