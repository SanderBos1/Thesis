import numpy as np
from Performance import Diagnostics

class Var:
    def __init__(self, data, df):
        self.data = data
        self.df = df


    """ Estimation of VAR models
    Input:
    X: data with size [number of variables, number of observations]
    First_Values: a list of arrays of the starting values of variables we are interested in.
    Output:
    VAR_Y : Coefficient matrix
    e: the error term
    """

    def varCalculation(self, first_values):
        y_td = first_values[0]
        y_rrp = first_values[1]
        X = self.data

        #parameter estimate td
        b_td = self.normalEquations(X, y_td)

        #parameter estimate rrp
        b_rrp = self.normalEquations(X, y_rrp)


        td_estimate = np.matmul(X, b_td)
        rrp_estimate = np.matmul(X, b_rrp)

        p = len(self.df.columns)
        diagn = Diagnostics(self.data, y_td, b_td, p)
        R = diagn.rsquared()
        M = diagn.MSE()
        F = diagn.Fstat()
        print(f"The R-squared is: {round(R, 2)}")
        print(f"The MSE is: {round(M, 2)}")
        print(f"The F-statistic is: {round(F, 2)}")

    def normalEquations(self, X, y):
        #calculation of normal equations
        XtX = np.matmul(X.T, X)
        Xty = np.matmul(X.T, y)
        XtX_Inv = np.linalg.inv(XtX)

        b = np.matmul(XtX_Inv, Xty)
        return b