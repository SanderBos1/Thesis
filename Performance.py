import numpy as np

from statsmodels.tools.eval_measures import aic

class Diagnostics:

    def __init__(self, y_predict, y_true):
        self.y_predict = y_predict
        self.y_true = y_true


    def logVariance(self):

        mean_error = np.mean((self.y_true - self.y_predict)**2)
        var = np.mean(((self.y_true - self.y_predict) - mean_error)**2)

        return var

    def results(self):
        logvar = self.logVariance()
        return logvar
