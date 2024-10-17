import numpy as np


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0252168A(learning_rate, num_iters):
    """
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
    a=2.5
    b=0.5
    c=2.0
    d=4.0

    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)

    #Part A
    for i in range(num_iters):
        grad = 5 * a*a*a*a
        a -= learning_rate * grad
        a_out[i] = a
        f1_out[i] = a*a*a*a*a

    #Part B
    for j in range(num_iters):
        grad = 2 * np.cos(b) * np.sin(b)
        b -= learning_rate * grad
        b_out[j] = b
        f2_out[j] = np.sin(b)**2

    #Part C
    for k in range(num_iters):
        grad_c = 3*c**2
        grad_d = 2*d*np.sin(d) + (d**2)*np.cos(d)
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        c_out[k] = c
        d_out[k] = d
        f3_out[k] = c**3 + (d**2)*np.sin(d)

    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out

print(A3_A0252168A(0.1,1))