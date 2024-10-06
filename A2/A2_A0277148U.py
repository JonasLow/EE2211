import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0277148U(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=N)

    encoder = OneHotEncoder(sparse_output=False)
    Ytr = encoder.fit_transform(y_train.reshape(-1, 1))
    Yts = encoder.transform(y_test.reshape(-1, 1))
    
    Ptrain_list = []
    Ptest_list = []
    w_list = []

    error_train_array = np.zeros(10)
    error_test_array = np.zeros(10)

    lambda_val = 0.0001

    for order in range(1, 11):
        poly = PolynomialFeatures(degree=order)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.fit_transform(X_test)
        
        Ptrain_list.append(X_poly_train)
        Ptest_list.append(X_poly_test)

        if X_poly_train.shape[0] > X_poly_train.shape[1]:  # Primal
            reg = lambda_val * np.identity(X_poly_train.shape[1])
            w = inv(X_poly_train.T @ X_poly_train + reg) @ X_poly_train.T @ Ytr
        else:  # Dual
            reg = lambda_val * np.identity(X_poly_train.shape[0])
            w = X_poly_train.T @ inv(X_poly_train @ X_poly_train.T + reg) @ Ytr

        w_list.append(w)

        y_train_pred = X_poly_train @ w
        y_test_pred = X_poly_test @ w

        y_train_pred_labels = np.argmax(y_train_pred, axis=1)
        y_test_pred_labels = np.argmax(y_test_pred, axis=1)

        error_train_array[order - 1] = np.sum(y_train_pred_labels != y_train)
        error_test_array[order - 1] = np.sum(y_test_pred_labels != y_test)

    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array

#X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array = A2_A0277148U(5)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
#print(Ytr)
#print(Yts)
#print(Ptrain_list)
#print(Ptest_list)
#print(w_list)
#print(error_train_array)
#print(error_test_array)
