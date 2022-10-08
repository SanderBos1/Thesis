import numpy as np


# calculation of normal equations

def normalEquations(X, y):
    XtX = np.matmul(X.T, X)
    Xty = np.matmul(X.T, y)
    XtX_Inv = np.linalg.inv(XtX)

    b = np.matmul(XtX_Inv, Xty)

    return b

# calculates the dot product of two lists
def dotProduct(x, b):
    predicted = 0
    for k in range(len(x)):
        predicted = predicted + b[k] * x[k]
    return predicted