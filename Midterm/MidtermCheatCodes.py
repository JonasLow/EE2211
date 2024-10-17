import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from numpy.linalg import det
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

'''
Cheat Code Guides:

msolve(X, y, ridge=False/True, lamd=0)
> Purpose: Solves linear system Xw = y
    > Exact Solution (Square Matrix)
    > Least Squares Solution (Over-Determined)
    > Right-Inverse Solution (Under-Determined)
    > ...
> Set ridge=True and specify lambda value

inv_test(X)
> Purpose: Test whether the left and right inverses of matrix X exists depending on matrix dimension

add_bias(X)
> Purpose: Adds a bias (intercept) column to the matrix X

htov(X)
> Purpose: Converts a 1D horizontal array into a vertical vector

left_inv(X) & right_inverse(X)
> Purpose: Self-explainatory.

normal_cdf(lower, upper, miu, sigma)
> Purpose: Calculate P(lower <= X <= upper)

b_classify(y)
> Purpose: Converts target vector into binary class (i.e. check positive or negative)

one_hot_encoding(y)
> Purpose: Converts a categorical vector into one-hot encoded format

m_classify(y_exp)
> Purpose: Classifies each row of a matrix y_exp based on the maximum value in that row

poly_transform(X, degree)
> Purpose: Performs polynomial feature transformation on matrix X up to the specified degree

pearson_r(x, y)
> Purpose: Computes the Pearson correlation coefficient between vectors x and y
'''

# Helper function to calculate least squares error
def least_squares(X, y, w):
    y_pred = X @ w
    residuals = y - y_pred
    squared_errors = residuals.T @ residuals
    return squared_errors[0, 0]


# Helper function to calculate least squares error loss
def error_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    squared_diff = (y_true - y_pred) ** 2
    mse = np.mean(squared_diff)
    return mse


def msolve(X, y, ridge=False, lambd=0.1):
    '''
    takes in X-matrix: (n x d) AND a y-vector: (m x 1)
    outputs w --> for Xw = y AND nature of w
    m = number of samples
    d = number of features
    
    For ridge regression: specify lambda
    '''
    if X.shape[0] != y.shape[0]:
        print("Inconsistent System", '\n')
        return
    
    md = X.shape if len(X.shape) > 1 else (X.shape[0], 1) #(m, d)

    # Square Matrix
    if md[0] == md[1]:
        print("X is a square matrix")
        print("Even-determined system", '\n')
        if det(X) == 0:
            print("X matrix determinant is 0")
            print("There are rows/columns in X that are linearly dependent")
            print("X matrix dimensions: ", md)
            print("rank(X): ", matrix_rank(X))
            print("----------")
        if ridge == False and det(X) != 0:
            w = inv(X) @ y
            return w
        elif ridge == True:
            print("performing ridge regression in primal form... (square matrix)")
            reg_L = lambd*np.identity((X.T @ X).shape[0])
            w = inv(X.T @ X + reg_L) @ X.T @ y
            return w
    
    # Over-determined system (left-inverse)
    elif md[0] > md[1]:
        print("Over-determined system: more samples/equations than parameters/features/variables")
        print("i.e. more rows than columns")
        print("X is a tall matrix", '\n')
        if ridge == False:
            # check if (X^T X) is invertible
            if det(X.T @ X) == 0:
                print("X.T @ X is a singular matrix")
                print("X.T @ X expected to be a square matrix of: ", md[1], " size")
                print("rank X.T @ X: ", matrix_rank(X.T @ X))
                return
            # check if solution is exact or approximate
            if matrix_rank(X) == matrix_rank(np.c_[X, y]):
                print("solution is exact as rank(X) = rank([X y])")
            print("Left-inverse APPROXIMATION")
            print("----------")
            w = inv(X.T @ X) @ X.T @ y
            return w
        elif ridge == True:
            print("performing ridge regression in primal form...")
            reg_L = lambd*np.identity((X.T @ X).shape[0])
            w = inv(X.T @ X + reg_L) @ X.T @ y
            return w
        
    #Under-determined system (right-inverse)
    elif md[0] < md[1]:
        print("Under-determined system: more variables/features/parameters than samples/equations")
        print("i.e. more columns than rows")
        print("X is a wide matrix", '\n')
        if ridge == False:
            # Check if (X X^T) is invertible
            if det(X @ X.T) == 0:
                print("X @ X.T is a singular matrix")
                print("X @ X.T expected to be a square matrix of: ", md[0], " size")
                print("rank X @ X.T: ", matrix_rank(X @ X.T))
                return
            if matrix_rank(X) < matrix_rank(np.c_[X, y]):
                print("NO solution as rank(X) < rank([X y]) [INCONSISTENT SYSTEM]")
            print("Right-inverse Approximation")
            print("----------")
            w = X.T @ inv(X @ X.T) @ y
            return w
        elif ridge == True:
            print("performing ridge regression in dual form...")
            reg_L = lambd*np.identity((X @ X.T).shape[0])
            w = X.T @ inv((X @ X.T) + reg_L) @ y
            return w

# # Example msolve usage (UNCOMMENT TO USE, replace ridge and lambda value yourself...)
# t = 6
# X = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([[1], [1], [1]])
# w = msolve(X, y)
# print(w)

# # Least Squares Error
# print(least_squares(X, y, w))


def inv_test(X):
    md = X.shape if len(X.shape) > 1 else (X.shape[0], 1) # (m, d)
    print("Size of X: ", md, '\n')

    print("Right-inverse exists: ", bool(det(X @ X.T) > 0.001))
    print("det(X @ X.T) = ", det(X @ X.T) if det(X @ X.T) > 0.001 else 0, '\n')
    print("Left-inverse exists: ", bool(det(X.T @ X) > 0.001))
    print("det(X.T @ X) = ", det(X.T @ X) if det(X.T @ X) > 0.001 else 0, '\n')

    # Square matrix
    if md[0] == md[1]:
        print("X is a square matrix")
        print("Inverse exists: ", bool(det(X)))
        print("det(X): ", det(X))
    else:
        print("Inverse does not exist as X is not a square matrix")
        print("Matrix has no determinant")

# # Example inv_test usage (UNCOMMENT TO USE)
# X = np.array([[1, 4, 2], [-3, 10, 5]])
# inv_test(X)


def add_bias(X):
    nrows = X.shape[0]
    return np.hstack((np.ones((nrows, 1)), X))

# Example add_bias usage (UNCOMMENT TO USE)
# X = np.array([[50, 10], [40, 7], [65, 12], [70, 5], [75, 4]])
# y = np.array([[3], [7], [6], [1], [9]])
# X_bias = add_bias(X)
# print(X_bias)
# w = msolve(X_bias, y)
# print(w, '\n')

# # Prediction of new values
# print((np.array([[1, 42, 8]]) @ w), '\n')

# Least Squared Error Loss
# y2 = X_bias @ w
# print(error_loss(y, y2))


def htov(X):
    '''
    Input: 1D row [a, b, c]
    Output: 2D [[a], [b], [c]]
    '''
    return X.reshape(-1, 1)

# # Example htov usage (UNCOMMENT TO USE)
#X = np.array([2, 3, 4])
#X_vector = htov(X)
#print(X_vector)


def left_inv(X):
    return inv(X.T @ X)


def right_inv(X):
    return inv(X @ X.T)

# # Example left_inverse usage (UNCOMMENT TO USE, for right -> replace "left" to "right")
#X = np.array([[2, 1], [3, 4], [1, 2]])
#left_inverse = left_inv(X)
#print(left_inverse)


def normal_cdf(lower, upper, miu, sigma):
    p_lower = stats.norm.cdf(lower, miu, sigma)
    p_upper = stats.norm.cdf(upper, miu, sigma)
    # Probability of the interval - P
    P = p_upper - p_lower

    # print results
    print(f"Normal distribution: mean = {miu}, std dev = {sigma} \n")
    print(f"Probability of occuring between {lower} and {upper}: ")
    # round off to 1dp
    print(f"--> inside interval P = {round(P * 100, 1)}%")
    print(f"--> outside interval 1 - P = {round((1 - P) * 100, 1)}% \n")

# # Example normal_cdf usage (UNCOMMENT TO USE)
#normal_cdf(-1, 1, 0, 1)  # Standard normal distribution


def b_classify(y):
    '''
    Output: 1 (positive), -1 (negative)
    '''
    return np.sign(y)

# # Example b_classify usage (UNCOMMENT TO USE)
# y = np.array([[-3, 0, 4], [2, 4, -1], [1, 0, -10]])
# classes = b_classify(y)
# print(classes)


def one_hot_encoding(y):
    '''
    Output: one-hot encoded target vector, y
    '''
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(y)
    return onehot_encoded

# # Example one_hot_encoding usage (UNCOMMENT TO USE)
#y = np.array([[200], [300], [300]])
#one_hot_encoded = one_hot_encoding(y)
#print(one_hot_encoded)


def m_classify(y_exp):
    '''
    Output: (1-Based) Index of the max number in each output row
    '''
    print("these are the classes: ", np.argmax(y_exp, axis=1) + 1)
    print("one-hot format: ", [[1 if y == max(x) else 0 for y in x] for x in y_exp])

# # Example m_classify usage (UNCOMMENT TO USE)
#y_exp = np.array([[0.2, 0.8], [0.9, 0.1]])
#m_classify(y_exp)


# Helper function to solve for a certain x_value for a given polynomial
def solve_polynomial_value(w, degree, x_array):
    """
    Solves for the polynomial value at a given x using the weight vector.
    
    Parameters:
    - w: Weight vector (polynomial coefficients).
    - degree: Degree of the polynomial.
    - x_array: The x-array to evaluate the polynomial.
    
    Returns:
    - Computed y-value for the given x.
    """
    poly = PolynomialFeatures(degree)    
    x_poly = poly.fit_transform(x_array)
    y_value = x_poly @ w
    
    print(f"Polynomial value at x = {x_array}: \n {y_value}")


def poly_transform(X, degree):
    poly = PolynomialFeatures(degree)
    poly.fit(X)
    print("Polynomial Transform: The features are: ", poly.get_feature_names_out())
    print("length = ", len(poly.get_feature_names_out()))
    print("----------")
    return poly.transform(X)

# # Example poly_transform usage (UNCOMMENT TO USE)
# X = np.array([[1, 4], [5, -1], [2, 3]])
# y = np.array([[1], [3], [1]])
# deg = 3
# X_poly = poly_transform(X, degree=deg)
# print(X_poly) # Polynomial expansion matrix
# w = msolve(X_poly, y)
# print(w, '\n')

# # solve_polynomial_value(w_example, degree_example, x_array_example)
# n = np.array([[42, 8]])
# solve_polynomial_value(w, deg, n)


def pearson_r(x, y):
    '''
    This ensures that both x and y are 1D ndarrays
    '''
    return stats.pearsonr(x, y)[0]

# # Example pearson_r usage (UNCOMMENT TO USE)
#x = np.array([2, 3, 4])
#y = np.array([5, 6, 7])
#r = pearson_r(x, y)
#print(f"Pearson correlation coefficient: {r}")
