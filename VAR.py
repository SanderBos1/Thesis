import numpy as np
from Performance import Diagnostics
from sklearn.model_selection import train_test_split
from formulas import normalEquations, dotProduct
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Var:

    def __init__(self, data, optimal_lag_a, optimal_lag_b):
        self.data = data
        self.optimal_lag_a = optimal_lag_a
        self.optimal_lag_b = optimal_lag_b

    """ Estimation of VAR models
    Input:
    X: data with size [number of variables, number of observations]
    First_Values: a list of arrays of the starting values of variables we are interested in.
    Output:
    VAR_Y : Coefficient matrix
    e: the error term
    """

    def varCalculation(self, variables, index):
        #data preparation

        # set columname as index
        self.data = self.data.set_index(index)
        features = self.data.columns

        # drop columns that are not needed
        for i in features:
            if i not in variables:
                self.data = self.data.drop([i], axis=1)

        # loop through each of the features
        for j in variables:
            # loop through each lag
            if j == variables[0]:
                optimal_lag = self.optimal_lag_a
            else:
                optimal_lag = self.optimal_lag_b
            for i in range(1, optimal_lag+1):
                # add lag i of feature j to the dataframe
                self.data[f"{j}_Lag_{i}"] = self.data[j].shift(i)
        self.data = self.data.dropna()

        # extract the first variables.
        y_True = self.data[variables[0]]

        for i in range(len(variables)):
            self.data = self.data.drop([variables[i]], axis=1)

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
        amount_var = self.optimal_lag_a + self.optimal_lag_b
        diagntd = Diagnostics(y_values, y_True, amount_var)
        r, f, aic = diagntd.results()
        return r, f, aic

    #makes a plot of the data
    def varPlot(self, index, ind_var):
        self.data.plot(x=index, y=ind_var, kind='line')
        plt.show()




