class Granger:
    def __init__(self, data):
        self.data = data

    """ Estimation of VAR models
    Input:
    X: data with size [number of variables, number of observations]
    m: Model Order
    Output:
    VAR_Y : Coefficient matrix
    e: the error term
    """

    def varCalculation(self):
        print(5)
