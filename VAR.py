import numpy as np
from Performance import Diagnostics
from sklearn.model_selection import train_test_split
from formulas import normalEquations, dotProduct
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Var:

    def __init__(self, data, optimal_lag_a, optimal_lag_b,  PointsAhead):
        self.data = data
        self.optimal_lag_a = optimal_lag_a
        self.optimal_lag_b = optimal_lag_b
        self.PointsAhead = PointsAhead

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
        # split in test and train data

        test_size = 0.2
        train_data, test_data = train_test_split(self.data, test_size=test_size, shuffle=False)
        p = len(train_data.columns)

        # extract the first variables.
        y_TotalDemand = train_data[variables[0]]
        y_TotalDemand_test = test_data[variables[0]]

        for i in range(len(variables)):
            train_data = train_data.drop([variables[i]], axis=1)
            test_data = test_data.drop([variables[i]], axis=1)

        # insert intercept column with all value of 1
        train_data.insert(0, "Intercept", 1)
        test_data.insert(0, "Intercept", 1)
        print(test_data)
        # transform to numpy for usage
        train_td = y_TotalDemand.to_numpy()
        train_data = train_data.to_numpy()
        test = y_TotalDemand_test.to_numpy()
        test_data = test_data.to_numpy()


        # parameter estimate
        b = normalEquations(train_data, train_td)

        #Here starts the prediction code
        # points ahead defines the amount of points we want to predict
        prediction = []

        # loop over all variables in the test set
        for j in range(len(test_data)):
            x = test_data[j].tolist()
            for i in range(self.PointsAhead):
                predicted_td = dotProduct(x, b)
                x = x[:-1]
                x.insert(1, predicted_td)

            estimate = np.matmul(x, b)
            prediction.append(estimate)

        prediction = np.array(prediction)


        # calculation of diagnostics
        amount_var = self.optimal_lag_a + self.optimal_lag_b
        diagntd = Diagnostics(prediction, test, p, amount_var)
        r, m, f, aic = diagntd.results()
        return r, m, f, aic, b


    def varPlot(self, index, ind_var):
        self.data.plot(x=index, y=ind_var, kind='line')
        plt.show()


