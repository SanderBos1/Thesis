import numpy as np
from statistics import mean


class Diagnostics:
    # y = Values of the dependent variable
    # b = Parameter estimates
    # data = the data rows
    # optimal_lag = the amount of variables

    def __init__(self, data, y, optimal_lag):
        self.data = data
        self.y = y
        self.n = len(self.y)
        self.mean = mean(self.y)
        self.optimal_lag = optimal_lag
        self.p = optimal_lag + 1

    def SStot(self):
        SStot = np.sum((self.y - self.mean)**2)
        return SStot

    def SSreg(self):
        SSreg = np.sum((self.data - self.mean)**2)
        return SSreg

    def SSres(self):
        SSres = np.sum((self.y - self.data)**2)
        return SSres

    def rsquared(self):
        ssres = self.SSres()
        sstot = self.SStot()
        R = 1 - ssres / sstot
        return R

    def MSE(self):
        ssres = self.SSres()
        M = ssres/self.n
        return M

    def Fstat(self):
        ssres = self.SSres()
        ssreg = self.SSreg()
        F = (ssreg/self.p) / (ssres/(self.n-self.p))
        return F

    def Akaike(self):
        ssres = self.SSres()
        aic = self.n * np.log(ssres/self.n) + 2*self.optimal_lag
        return aic


    def results(self):
        R = self.rsquared()
        M = self.MSE()
        F = self.Fstat()
        aic = self.Akaike()
        return R, M, F, aic
