import numpy as np                          # fundamental package for scientific computing with Python
import matplotlib.pyplot as plt             # a library for plotting graphs in Python
from testCases_v2 import *                  # provides some test examples
import sklearn                              # provides simple and efficent tools for data mining and data analysis
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets              # provides various usefull functions used in this assignment

np.random.seed(1)                           # set a seed so that the results are consistent

X, Y = load_planar_dataset()                # will load "flower" 2-class dataset into variables X and Y
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral);             # Returns a contiguous flattened array
plt.show()

# - a numpy-array (matrix) X that contains your features (x1, x2)
# - a numpy-array (vector) Y that contains your labels (red:0, blue:1)

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

print("The shape of X is: " + str(shape_X))
print("The shape of Y is: " + str(shape_Y))
print("I have m = %d training examples!" %(m))

# Before building a full neural network, lets first see how logistic regression performs on this problem.
# You can use sklearn's built-in functions to do that. Run the code below to train a logistic regression classifier on the dataset.

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())
plt.title("Logistic Regression")
plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print("Accuracy of logistic regression: %d " % float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100)
      + "% " + "(percentage of correctly labelled datapoints)")

# Interpretation: The dataset is not linearly separable, so logistic regression doesn't perform well.
                # Hopefully a neural network will do better. Let's try this now!

# Defining the Neural Network Structure
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return(n_x, n_h, n_y)

X_assess, Y_assess = layer_sizes_test_case()
n_x, n_h, n_y = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x: " + str(n_x))
print("The size of the hidden layer is: n_h: " + str(n_h))
print("The size of the output layer is: n_y: " + str(n_y))

# Initialize the model's parameters
def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing our parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)   # we set up a seed so that your output matches ours although the initialization is random.
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b1 = " + str(parameters["b2"]))

# The Loop -- forward prop -- compute loss -- bacward prop to get gradients -- update parameters(gradient descent)

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing our parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement forward propagation to calculate A2(probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    # Values needed in the back propagation are stored in "cache". The cache will be given as an input to the backpropagation function.
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours
print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]) )
# 0.262818640198 0.091999045227 -1.30766601287 0.212877681719 --- expected output

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation 13

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    [Note that the parameters argument is not used in this function, but the auto-grader currently expects this parameter.
     Future version of this notebook will fix both the notebook and the auto-grader so that 'parameters' is not needed.
     For now, please include 'parameters' in the function signature, and also when invoking this function.]

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    m = Y.shape[1]      # number of example
    # compute cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = -np.sum(logprobs) / m
    # cost = -(np.dot(np.log(A2), Y) + np.dot(np.log(1-A2), 1-Y)) / m        # not worked

    cost = float(np.squeeze(cost))          # makes sure cost is in the dimension we expect.
    assert(isinstance(cost, float))
    return cost

A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess, parameters)))          # expected output: cost = 0.6930587610394646

def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
    parameters -- python library containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Bacward propagtion: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule

    Arguments:
    parameters -- python dictionary containing parameters
    grads -- python dictionary containing gradients

    Returns:
    parameters -- python dictionary containing updated parameters
    """

    # Retrieve each parameter from the dictionary 'parameters'
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary 'grads'
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # perform Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# In order to build Neural Network model use previous functions in the right order
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shpae (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_ cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters"
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

    return parameters

X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations = 10000, print_cost = True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# Predictions
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing parameters
    X -- input data of size (n_x, m)

    Returns:
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

parameters, X_assess = predict_test_case()
predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))               # expected output:   predictions mean	0.666666666667


# It is time to run the model and see how it performs on a planar dataset.
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True)
# plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.ravel())
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
# print accuracy
predictions = predict(parameters, X)
print("Accuracy %d" %float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + "%")     # Accuracy: 90%
# Accuracy is really high compared to Logistic Regression.
# The model has learnt the leaf patterns of the flower!
# Neural Networks are able to learn even highly non-linear decision boundries, unlike logistic regression.
# Now, let's try out several hidden layer sizes.

# Tuning Hidden Layer Sizes
plt.figure(figsize = (16,32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(3, 3, i+1)
    plt.title("Hidden layer of size %d" %n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 10000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.ravel())
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T)) / float(Y.size)*100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()


# Interpretation
# The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data.
# The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to fits the data well without also incurring noticable overfitting.
# We will also learn later about regularization, which lets us use very large models (such as n_h = 50) without much overfitting.




# Performance on other datasets             # I will have to try other datasets later

# datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = "noisy_moons"

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y % 2

# visualize the data
plt.scatter(X[0, :], X[1, :], c = Y.ravel(), s = 40, cmap = plt.cm.Spectral)
plt.show()
