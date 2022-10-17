import statistics


class Diagnostics:

    def __init__(self, y_predict, y_true):
        self.y_predict = y_predict
        self.y_true = y_true

    def logVariance(self):

        residual = self.y_true - self.y_predict
        var = statistics.variance(residual)

        return var

    def results(self):
        logvar = self.logVariance()
        return logvar
