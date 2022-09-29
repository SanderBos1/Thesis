import numpy as np
from statistics import mean


class Diagnostics:

    # y = Values of the dependent variable
    # b = Parameter estimates
    # data = the data rows
    # p =

    def __init__(self, data, y, p):
        self.data = data
        self.y = y
        self.n = len(self.y)
        self.p = p
        self.mean = mean(self.y)

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
        print(self.n)
        print(ssres)
        M = ssres/self.n
        return M

    def Fstat(self):
        ssres = self.SSres()
        ssreg = self.SSreg()
        F = (ssreg/self.p) / (ssres/(self.n-self.p))
        return F

    def results(self):
        R = self.rsquared()
        M = self.MSE()
        F = self.Fstat()
        print(f"The R-squared is: {round(R, 2)}")
        print(f"The MSE is: {round(M, 2)}")
        print(f"The F-statistic is: {round(F, 2)}")