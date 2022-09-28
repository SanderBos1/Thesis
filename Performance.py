import numpy as np


class Diagnostics:

    # y = Values of the dependent variable
    # b = predicted parameters
    # data = the real data

    def __init__(self, data, y, b, p):
        self.data = data
        self.b = b
        self.y = y
        self.n = len(self.y)
        self.p = p

    def SStot(self):
        SStot = np.matmul(self.y.T, self.y)
        return SStot

    def SSreg(self):
        SSreg = np.matmul(np.matmul(self.y.T, self.data), self.b)
        return SSreg

    def SSres(self):
        SSres = np.matmul((self.y-np.matmul(self.data, self.b)).T, (self.y-np.matmul(self.data, self.b)))
        return SSres

    def rsquared(self):
        ssres = self.SSres()
        sstot = self.SStot()
        R = 1 - ssres / (sstot - sum(self.y)**2/self.n)
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

    def results(self):
        R = self.rsquared()
        M = self.MSE()
        F = self.Fstat()
        print(f"The R-squared is: {round(R, 2)}")
        print(f"The MSE is: {round(M, 2)}")
        print(f"The F-statistic is: {round(F, 2)}")