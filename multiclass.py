import numpy as np
import matplotlib as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import time

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def cost_function(theta, X, y, regularized = False, lamb = 1.0):

    m, n = X.shape

    g = np.dot(X,theta)
    h = sigmoid(g)

    J = - ( np.sum(y*np.log(h)) + np.sum((1 - y)*np.log(1 - h)) )/m

    if regularized:
        J = J + lamb/(2.0*m) * np.dot(theta[1:], theta[1:])

    return J

def cost_gradient(theta, X, y, regularized = False, lamb = 1.0):

    m,n = X.shape

    g = np.dot(X, theta)
    h = sigmoid(g)

    grad = np.dot(X.T, h - y)/m

    if regularized:
        grad[1:] = grad[1:] + (lamb/m)*theta[1:]

    return grad

def multiclassifier(X, y, K, learn_rate):
    m, n = X.shape
    X = np.concatenate((np.ones([m,1]),X), axis=1)
    n = n+1

    all_theta = np.zeros([K,n])
    theta0 = np.zeros(n)

    for j in range(K):
        yj = (y==j)*1
        result = minimize(cost_function, theta0, jac=cost_gradient, args=(X, yj, True, learn_rate), method='CG')
        all_theta[j] = result.x

    return all_theta

def predict(X, theta):
    m, n = X.shape
    X = np.concatenate((np.ones([m,1]), X), axis=1)

    h  = sigmoid(np.dot(X,theta.T))
    outcomes = np.argmax(h, axis=1)

    return outcomes

## -------------------------------------------------------------------
start_time = time.time()

datapath = "datafiles/"
data = loadmat(datapath + "ex3data1.mat")
X = data['X']
y = data['y']
m = len(y)
y[np.where(y==10)] = 0 
y = y.reshape(m)

K = 10
lamb = 0.1

all_theta = multiclassifier(X, y, K, lamb)

end_time = time.time()
print 'Time to run: ' +  str(end_time - start_time)

outcomes = predict(X, all_theta)
accuracy = 100.0*np.mean(outcomes == y)
print 'Accuracy of prediction is', accuracy, '%'
