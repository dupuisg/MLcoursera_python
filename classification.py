import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fmin_tnc

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

## At the moment this is written for one example/data point
def predict(X, theta):
    h = sigmoid(np.dot(X,theta))
    if X.shape[0] == 1:
      if h >= 0.5:
          print 'Admitted, with probability ', h
      else:
          print 'Not admitted, admission probability ', h

    outcomes = (h >= 0.5)*1.0
    return outcomes

def map_features(X, k):
    # polynomial terms of two features - extend to n features later
    m = X.shape[0]
    x1 = X[:,0]
    x2 = X[:,1]
    for i in range(2,k+1):
        for j in range(i+1):
          poly_term = x2**j * x1**(i-j)
          X = np.concatenate((X, poly_term.reshape([m,1])), axis=1)
    return X

#---------------------------------- Linear case, unregularized -----------------------------------
datapath = "datafiles/"
data = np.loadtxt(datapath + "ex2data1.txt", delimiter=",")

##Plot data
pos_values = data[data[:,-1] == 1]
neg_values = data[data[:,-1] == 0]
plt.figure()
plt.plot(pos_values[:,0], pos_values[:,1], color='b', marker='o', linestyle='', label='Admitted')
plt.plot(neg_values[:,0], neg_values[:,1], color='r', marker='x', linestyle ='', label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc='upper right')
plt.savefig("figures/admissions.png")

X = data[:, 0:-1]
y = data[:, -1]
m = len(X)
X = np.concatenate((np.ones([m,1]), X), axis=1)
n = X.shape[1]

theta0 = np.zeros(n)

result = minimize(cost_function, theta0, jac=cost_gradient, args=(X,y), method='Newton-CG') #, options = {'disp':True})
theta_fit = result.x

## Checking result
#test_scores = np.array([1, 45,85])
#print 'For exam scores ', test_scores[1], 'and ', test_scores[2], ':',
#predict(test_scores, theta_fit)
y_predicted = predict(X, theta_fit)
accuracy = np.mean(y_predicted == y)*100.0
print 'Accuracy for linear case: ', accuracy

### Plot decision boundary, defined by theta_0 + theta_1*x_1 + theta_2*x_2 = 0
x1 = np.linspace(0.0, 100.0, 100)
x2 = -1.0/(theta_fit[2]) * (theta_fit[0] + theta_fit[1]*x1)
plt.figure()
plt.plot(x1, x2, color='k')
plt.plot(pos_values[:,0], pos_values[:,1], color='b', marker='o', linestyle='', label='Admitted')
plt.plot(neg_values[:,0], neg_values[:,1], color='r', marker='x', linestyle ='', label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc='upper right')
plt.savefig("figures/admissions_decisionbndry.png")


#---------------------------------- Non-linear case, with regularization  -----------------------------------
data = np.loadtxt(datapath + "ex2data2.txt", delimiter=",")

##Plot data
pos_values = data[data[:,-1] == 1]
neg_values = data[data[:,-1] == 0]
plt.figure()
plt.plot(pos_values[:,0], pos_values[:,1], color='g', marker='o', linestyle='', label='Accepted')
plt.plot(neg_values[:,0], neg_values[:,1], color='r', marker='x', linestyle ='', label='Rejected')
plt.xlabel('Microchip test 1')
plt.ylabel('Microchip test 2')
plt.legend(loc='upper right')
plt.savefig("figures/microchip_tests.png")

X = data[:, 0:-1]
y = data[:, -1]
m = len(X)
X = map_features(X, 6)
X = np.concatenate((np.ones([m,1]), X), axis=1)
n = X.shape[1]

theta0 = np.zeros(n)

result = minimize(cost_function, theta0, jac=cost_gradient, args=(X,y, True, 1.0), method='Newton-CG') #, options = {'disp':True})
theta_fit = result.x

y_predicted = predict(X, theta_fit)
accuracy = np.mean(y_predicted == y)*100.0
print 'Accuracy for nonlinear case: ', accuracy

##Plot decision boundary
x1 = np.arange(-1.0, 1.5, 0.05)
x2 = np.arange(-1.0, 1.5, 0.05)
X1, X2 = np.meshgrid(x1,x2)
Z = np.zeros(X1.shape) + theta_fit[0]
for i in range(1,6):
    for j in range(i+1):
        n = sum(range(i+1)) + j
        Z = Z + theta_fit[n]*X2**j * X1**(i-j)
plt.figure()
plt.contour(X1, X2, Z, levels=[0])
pos_values = data[data[:,-1] == 1]
neg_values = data[data[:,-1] == 0]
plt.plot(pos_values[:,0], pos_values[:,1], color='g', marker='o', linestyle='', label='Accepted')
plt.plot(neg_values[:,0], neg_values[:,1], color='r', marker='x', linestyle ='', label='Rejected')
plt.xlabel('Microchip test 1')
plt.ylabel('Microchip test 2')
plt.legend(loc='upper right')
plt.savefig("figures/microchip_boundary.png")


'''
## Original versions which assumed matrices
## optimization routines require np arrays, and array operations

def cost_function(theta, X, y, regularized = False, lamb = 1.0):
    # assumes X, y, are matrices
    # theta must be a 1d array for minimization routines, convert in fcn

    m, n = X.shape
    theta = theta.reshape([n,1])

    J = -(y.T*np.log(sigmoid(X*theta)) + (1 - y.T)*np.log(1 - sigmoid(X*theta)))/m

    if regularized:
        J = J + lamb/(2.0*m) * (theta.T*theta)
        
    return J[0,0]

def cost_gradient(theta, X, y, regularized = False, lamb = 1.0):

    m,n = X.shape
    theta = theta.reshape([n,1])
 
    grad = (X.T * (sigmoid(X*theta) - y))/m

    if regularized:
        grad[1:] = grad[1:] + (lamb/m)*theta[1:]

    return np.array(grad)

'''
