import numpy as np
from Performance import Diagnostics
from sklearn.model_selection import train_test_split
from formulas import normalEquations, dotProduct
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

    def varCalculation(self, variables, notwanted):
        # drop columns that are not needed

        features = self.data.columns
        if len(variables) == 1:
            self.data = self.data.filter(like=variables[0])
        else:
            for i in notwanted:
                self.data = self.data.drop([i], axis=1)
            for j in notwanted:
                for i in range(1, self.lag + 1):
                    self.data = self.data.drop([f"{j}_Lag_{i}"], axis=1)

        # extract the first variables.
        y_True = self.data[variables[0]]
        self.data = self.data.drop(variables[0], axis=1)


        # insert intercept column with all value of 1
        self.data.insert(0, "Intercept", 1)

        # transform to numpy for usage
        y_True = y_True.to_numpy()
        self.data = self.data.to_numpy()

        # parameter estimate
        b = normalEquations(self.data, y_True)

        # Calculated the y values with the help of the predicted parameters b
        y_values = []
        for i in self.data:
            y = dotProduct(i, b)
            y_values.append(y)
        amount_var = len(features) * self.lag
        diagntd = Diagnostics(y_values, y_True, amount_var)
        r, f, aic = diagntd.results()
        return r, f, aic

    #makes a plot of the data
    def varPlot(self, index, ind_var):
        self.data.plot(x=index, y=ind_var, kind='line')
        plt.show()




