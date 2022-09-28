import numpy as np
from Performance import Diagnostics
from sklearn.model_selection import train_test_split


class Var:

    def __init__(self, data):
        self.data = data

    """ Estimation of VAR models
    Input:
    X: data with size [number of variables, number of observations]
    First_Values: a list of arrays of the starting values of variables we are interested in.
    Output:
    VAR_Y : Coefficient matrix
    e: the error term
    """

    def varCalculation(self):
        # split in test and train data
        test_size = 0.2
        train_data, test_data = train_test_split(self.data, test_size=test_size, shuffle=False)
        p = len(train_data.columns)

        # extract the first variables.
        y_TotalDemand = train_data["TOTALDEMAND"]
        y_RRP = train_data['RRP']
        train_data = train_data.drop(["TOTALDEMAND", "RRP"], axis=1)

        # insert intercept column with all value of 1
        train_data.insert(0, "Intercept", 1)

        # extract the first variables.
        y_TotalDemand_test = test_data["TOTALDEMAND"]
        y_RRP_test = test_data['RRP']
        test_data = test_data.drop(["TOTALDEMAND", "RRP"], axis=1)

        # insert intercept column with all value of 1
        test_data.insert(0, "Intercept", 1)

        # transform to numpy for usage


        train_td = y_TotalDemand.to_numpy()
        train_rdd = y_RRP.to_numpy()
        train_data = train_data.to_numpy()
        test_td = y_TotalDemand_test.to_numpy()
        test_rdd = y_RRP_test.to_numpy()
        test_data = test_data.to_numpy()


        # parameter estimate td
        b_td = self.normalEquations(train_data, train_td)

        # parameter estimate rrp
        b_rrp = self.normalEquations(train_data, train_rdd)


        td_estimate = np.matmul(train_data, b_td)
        rrp_estimate = np.matmul(train_data, b_rrp)

        #Here starts the prediction code

        # calculation of diagnostics
        diagntd = Diagnostics(train_data, train_td, b_td, p)
        diagntd.results()



    def normalEquations(self, X, y):
        #calculation of normal equations
        XtX = np.matmul(X.T, X)
        Xty = np.matmul(X.T, y)
        XtX_Inv = np.linalg.inv(XtX)

        b = np.matmul(XtX_Inv, Xty)

        return b