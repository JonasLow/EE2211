import numpy as np
from numpy import cos, sin

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0277148U(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    # Part (a)
    a = 2.5
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)

    for i in range(num_iters):
        gradientA = 5 * (a ** 4)
        a -= (learning_rate * gradientA)
        a_out[i] = a
        f1_out[i] = a ** 5

    
    # Part (b)
    b = 0.5
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)

    for j in range(num_iters):
        gradientB = 2 * sin(b) * cos(b)
        b -= (learning_rate * gradientB)
        b_out[j] = b
        f2_out[j] = sin(b) ** 2

    
    # Part (c)
    c = 2.0
    d = 4.0
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)

    for k in range(num_iters):
        gradientC = 3 * (c ** 2)
        c -= (learning_rate * gradientC)
        c_out[k] = c

        gradientD = (2 * d * sin(d)) + ((d ** 2) * cos(d))
        d -= (learning_rate * gradientD)
        d_out[k] = d

        f3_out[k] = c ** 3 + (d ** 2) * sin(d)


    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 
