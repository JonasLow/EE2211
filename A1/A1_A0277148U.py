import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0277148U(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    XT = X.transpose()
    InvXTX = np.linalg.inv(np.matmul(XT, X))
    w = np.matmul(InvXTX, np.matmul(XT, y))

    # return in this order
    return InvXTX, w

'''
print(A1_A0277148U(np.array([[1, 1], [4, 2], [4, 6], [3, -6], [0, -10]]), np.array([[-3], [2], [1], [5], [4]])))
'''
