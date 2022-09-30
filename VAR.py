import numpy as np
from Performance import Diagnostics
from sklearn.model_selection import train_test_split
from formulas import normalEquations, dotProduct

class Var:

    def __init__(self, data, optimal_lag, PointsAhead):
        self.data = data
        self.optimal_lag = optimal_lag
        self.PointsAhead = PointsAhead

    """ Estimation of VAR models
    Input:
    X: data with size [number of variables, number of observations]
    First_Values: a list of arrays of the starting values of variables we are interested in.
    Output:
    VAR_Y : Coefficient matrix
    e: the error term
    """

    def varCalculation(self, ind_var, index):
        #data preparation

        # set columname as index
        self.data = self.data.set_index(index)
        features = self.data.columns

        # drop columns that are not needed
        for i in features:
            if i != ind_var:
                self.data = self.data.drop([i], axis=1)

        # get column names
        features = self.data.columns


        # loop through each lag
        for i in range(1, self.optimal_lag + 1):
            # loop through each of the features
            for j in features:
                # add lag i of feature j to the dataframe
                self.data[f"{j}_Lag_{i}"] = self.data[j].shift(i)
        self.data = self.data.dropna()

        # split in test and train data

        test_size = 0.2
        train_data, test_data = train_test_split(self.data, test_size=test_size, shuffle=False)
        p = len(train_data.columns)

        # extract the first variables.
        y_TotalDemand = train_data[ind_var]
        train_data = train_data.drop([ind_var], axis=1)

        # insert intercept column with all value of 1
        train_data.insert(0, "Intercept", 1)

        # extract the first variables.
        y_TotalDemand_test = test_data[ind_var]
        test_data = test_data.drop([ind_var], axis=1)

        # insert intercept column with all value of 1
        test_data.insert(0, "Intercept", 1)

        # transform to numpy for usage


        train_td = y_TotalDemand.to_numpy()
        train_data = train_data.to_numpy()
        test = y_TotalDemand_test.to_numpy()
        test_data = test_data.to_numpy()


        # parameter estimate td
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
        diagntd = Diagnostics(prediction, test, p, self.optimal_lag)
        r, m, f, aic = diagntd.results()
        print(f"The R-squared is: {round(r, 2)}")
        print(f"The MSE is: {round(m, 2)}")
        print(f"The F-statistic is: {round(f, 2)}")
        print(f"The aic is: {round(aic, 2)}")
        return r, m, f, aic, b




