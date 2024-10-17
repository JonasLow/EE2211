"""
Submit a single python file with filename “A3_StudentMatriculationNumber.py”. 
Remember to rename “A3_StudentMatriculationNumber.py” using your student matriculation number,
like "A3_A1234567R.py".
(but do not rename “A3” function). 

You can only put the code in the corresponding areas and do not modify other areas.
""" 


# Only allow importing the following two packages
import numpy as np
import math # you may use math.cos(x) and math.sin(x)


# do not modify the function name in this assignment, just let it as "A3"
def A3(learning_rate, num_iters):
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
    
    # Task 1
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    #<<<<<<<<<<<<<<<<<<<<
    # Put your task 1 code here.
    
    a = np.array([2.5])

    for i in range(0,num_iters):
        a = a - learning_rate*(5*a**4)
        a_out[i] = a
        f1_out[i] = a**5
    
    #>>>>>>>>>>>>>>>>>>>>


    # Task 2
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    #<<<<<<<<<<<<<<<<<<<<
    # Put your task 2 code here.
    
    b = np.array([0.5])
    
    for j in range(0,num_iters):
        b = b - learning_rate*(2)*math.cos(b)*math.sin(b)
        b_out[j] = b
        f2_out[j] = math.sin(b)**2

    #>>>>>>>>>>>>>>>>>>>>


    # Task 3
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    #<<<<<<<<<<<<<<<<<<<<
    # Put your task 3 code here.

    c = 2.0
    d = 4.0
    
    for k in range(0,num_iters):
        c -= (learning_rate* 3 * (c ** 2))
        c_out[k] = c
        d -= (learning_rate*((2 * d * math.sin(d)) + (d ** 2 * math.cos(d))))
        d_out[k] = d
        f3_out[k] = c ** 3 + (d ** 2) * math.sin(d)
    
    #>>>>>>>>>>>>>>>>>>>>

    
    # Return in this order. Do not modify it.
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 

