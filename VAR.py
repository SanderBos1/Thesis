import statistics
import statsmodels.api as sm

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

    def varCalculation(self, variables):

        for j in variables:
            # loop through each of the features
            for i in range(1, self.lag + 1):
                # add lag i of feature j to the dataframe
                self.data[f"{j}_Lag_{i}"] = self.data[j].shift(i)
        self.data = self.data.dropna()

        # extract the first variables.
        y_True = self.data[variables[0]]

        self.data = self.data.drop(variables, axis=1)

        model = sm.OLS(y_True, self.data)
        results = model.fit()
        resid = results.resid
        var = statistics.variance(resid)
        print("these are the resid", var)
        return var








