import numpy as np
from Performance import Diagnostics
from sklearn.model_selection import train_test_split


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

    def varCalculation(self):
        #data preparation

        # set columname as index
        self.data = self.data.set_index("SETTLEMENTDATE")

        # drop columns that are not needed
        self.data = self.data.drop(["REGION", "PERIODTYPE"], axis=1)

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
        test_rrp= y_RRP_test.to_numpy()
        test_data = test_data.to_numpy()


        # parameter estimate td
        b_td = self.normalEquations(train_data, train_td)

        # parameter estimate rrp
        b_rrp = self.normalEquations(train_data, train_rdd)

        #Here starts the prediction code
        # points ahead defines the amount of points we want to predict
        td_ahead = []
        rrp_ahead = []

        # loop over all variables in the test set
        for j in range(len(test_data)):
            x = test_data[j].tolist()
            #print("this is the selected x", x)

            for i in range(self.PointsAhead):
                predicted_td = np.dot(x, b_td)
                predicted_rrp = np.dot(x, b_rrp)

                x = x[:-2]
                x.insert(1, predicted_rrp)
                x.insert(1, predicted_td)

            td_estimate = np.matmul(x, b_td)
            td_ahead.append(td_estimate)

            rrp_estimate = np.matmul(x, b_rrp)
            rrp_ahead.append(predicted_rrp)

        td_ahead = np.array(td_ahead)
        rrp_ahead = np.array(rrp_ahead)


        # calculation of diagnostics
        diagntd = Diagnostics(td_ahead, test_td, p)
        r, m, f = diagntd.results()
        print(f"The R-squared is: {round(r, 2)}")
        print(f"The MSE is: {round(m, 2)}")
        print(f"The F-statistic is: {round(f, 2)}")
        return r, m, f


    def normalEquations(self, X, y):
        #calculation of normal equations
        XtX = np.matmul(X.T, X)
        Xty = np.matmul(X.T, y)
        XtX_Inv = np.linalg.inv(XtX)

        b = np.matmul(XtX_Inv, Xty)

        return b