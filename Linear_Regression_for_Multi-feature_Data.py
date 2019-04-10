import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import pandas as pd


# To plot formatted figures
# %matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = '.'

# Read comma separated training data,  single feature
data = np.loadtxt("./ex1data1.txt", dtype = np.float32, delimiter=',')
# multi_feature data
data_2 = np.loadtxt("./ex1data2.txt", dtype = np.float32, delimiter=',')

X = np.array([data[:,0]])# Plot training data

plt.plot(data[:, 0], data[:, 1], 'bx')
plt.xlabel("Population of city in 10,000's", fontsize=12)
plt.ylabel("Profit in $10,000's", rotation=90, fontsize=12)
plt.title('Scatter plot of Training data')
# plt.axis([0, 2, 0, 15])
plt.show()

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1) == theta_1 is lecture notes of linear regression
    b -- initialized scalar (corresponds to the bias) == theta_0
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim,1), dtype=float)
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

#Test run
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
#**expected output**
# w = [[0.]
#  [0.]]
# b = 0
print("1 done")


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the hypothesis proposed

    Arguments:
    w -- weights, a numpy array of size (nx, 1)
    b -- bias, a scalar
    X -- data of size (nx, number of examples)
    Y -- output data of size(1, number of examples)

    Return:
    cost -- root mean square difference of the hypothesis and output value
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]
   
    # FORWARD PROPAGATION (FROM X ->  H ->  COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    #H = np.dot(w.T,X)+b
    H = np.dot(w.T,X)+b
    cost = (1/(2*m))*np.sum((H-Y)**2)
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1/m * np.dot(X,(H-Y).T)
    db = 1/m * np.sum(H-Y)
    ### END CODE HERE ###

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


#Test Run
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
print(cost)
print("2 done")


#**Expected output**
# dw = [[16.]
#  [36.]]
# db = 10.0
# cost = 52.0

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (nx, 1)
    b -- bias, a scalar
    X -- data of shape (nx, number of examples)
    Y -- output data of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1 lins of code)
        ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# Test Run
params, grads, costs = optimize(w, b, X, Y, num_iterations= 5, learning_rate = 0.009, print_cost = False)
print('')
print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("3 done")
print('')


#**Expected output**
# w = [[-0.08140605]
#  [-0.24334521]]
# b = 1.419030420562572
# dw = [[ 0.08670598]
#  [-0.02296021]]
# db = -0.05483309171519252


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_hat = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "H" predicted profit for given population of a city
    ### START CODE HERE ### (≈ 1 line of code)
    Y_hat = np.dot(w.T,X)+b
    ### END CODE HERE ###

    assert (Y_hat.shape == (1, m))

    return Y_hat


# Test Run
print("predictions = " + str(predict(w, b, X)))
# Test Run
# predictions = [[ 9 12]]


def model(X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds  Linear regression model by calling the function you have implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (nx, m_train)
    Y_train -- training output of shape (1, m_train)  city
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 10 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###
    # initialize parameters (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈1 lines of code)
    Y_hat = predict(w, b, X_train)
    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_hat - Y_train))))

    #print('History = ', history.size())

    d = {"costs": costs,
         "Y_prediction_train": Y_hat,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

# Multi
X22 = np.array([data_2[:,0:2]])
X22_disp = np.array([data_2[:,0:2]])
Y22 = np.array([data_2[:,2]])
X22 = (X22 - np.mean(X22))/np.std(X22)
Y22 = (Y22 - np.mean(Y22))/np.std(Y22)
m = len(Y22)

feature = X22[0].T

print('feature = ', feature)
print('feature shape = ', feature.shape)

d = model(feature, Y22, num_iterations=1000, learning_rate=0.005, print_cost=True)

#    1 = 0  2 = Kolom   3 = Attribute
print('X22[0][0][0] = ', X22[0].T)
print('X22[0][0][1] = ', X22[0].T)

w = d["w"]
b = d["b"]
print('w = ', w)
print('b = ', b)

# X_test   try to plot the hypothesis function
X_test = np.array([[3500.0],[8.0]])
pred = predict(w, b, X_test)
print("House size = 3500, Total bedroom = 8, The prediction of house value $"+str(pred))

X_test = np.array([[3500.0],[2.0]])
pred = predict(w, b, X_test)
print("House size = 3500, Total bedroom = 2, The prediction of house value $"+str(pred))

X_test = np.array([[7500.0],[2.0]])
pred = predict(w, b, X_test)
print("House size = 7500, Total bedroom = 2, The prediction of house value $"+str(pred))

X_test = np.array([[7500.0],[8.0]])
print(X_test)
pred = predict(w, b, X_test)
print("House size = 7500, Total bedroom = 8, The prediction of house value $"+str(pred))

#feature_pred = X22_disp
pred_copy = np.zeros((Y22.size,),dtype=np.int32)
feature_1 = np.zeros((Y22.size,),dtype=np.int32)
feature_2 = np.zeros((Y22.size,),dtype=np.int32)

print('feature0 = ', feature[0])
print('feature1 = ', feature[1])
print('feature0.shape = ', feature[0].shape)
print('feature1.shape = ', feature[1].shape)

pred = predict(w, b, feature)   
print('pred = ', pred[0].shape)
print('feature shape = ', feature.shape)

features = X22_disp[0].T

print('featureeeeeee shape = ', features.shape)
print('feature0.shape = ', features[0].shape)
print('feature1.shape = ', features[1].shape)

plt.plot(features[0].T, abs(pred[0]), 'rx')
plt.xlabel("House Size ", fontsize=12)
plt.ylabel("Prices ", rotation=90, fontsize=12)
plt.title('Scatter plot of Prediction')
plt.show()

plt.plot(features[1].T, abs(pred[0]), 'bx')
plt.xlabel("Number of Bedroom ", fontsize=12)
plt.ylabel("Prices ", rotation=90, fontsize=12)
plt.title('Scatter plot of Prediction')
plt.show()
