import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def cost_function(theta, X, y, lamb):
    m,n = X.shape
    theta = theta.reshape([n,1])
   
    h = np.dot(X, theta)
    J = np.dot((h-y).T, h-y)/(2.0*m) + lamb * np.dot(theta[1:].T, theta[1:])/(2.0*m)

    grad = np.dot(X.T, h-y)/m 
    grad[1:] += lamb*theta[1:]/m

    J = J[0,0]
    grad = grad.reshape(n)

    return (J, grad)

def feat_norm(X):

    mus = np.mean(X, axis=0)
    sigmas = np.std(X, axis=0)

    normed_X = (X - mus)/sigmas

    return normed_X, mus, sigmas

def poly_map(X, p):
    X_poly = np.zeros([X.shape[0],p])
    
    for i in range(p):
        X_poly[:,i] = X[:,0]**(i+1)

    return X_poly

def trainLinReg(X, y, lamb):
    n = X.shape[1]
    theta_init = np.zeros(n)

    result = minimize(cost_function, theta_init, jac=True, args=(X,y,lamb), method='CG', options={'maxiter':250})
    return result.x

def learningcurve(X, y, Xval, yval, lamb):
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    Xval = np.concatenate((np.ones([Xval.shape[0],1]),Xval), axis=1)
    X = np.concatenate((np.ones([m,1]), X), axis=1)

    for i in range(m):
        X_subset = X[0:i+1, :]
        y_subset = y[0:i+1]
        theta = trainLinReg(X_subset, y_subset, lamb)
        error_train[i], grad = cost_function(theta, X_subset, y_subset, 0.0)
        error_val[i], grad = cost_function(theta, Xval, yval, 0.0)        

    return error_train, error_val

## Calculate cost as a function of lambda
def validationcurve(X, y, Xval, yval):
    lambda_vector = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    n = len(lambda_vector)
    error_train = np.zeros(n)
    error_val = np.zeros(n)
    m = len(X)
    mval = len(Xval)
    X = np.concatenate((np.ones([m,1]), X), axis=1)
    Xval = np.concatenate((np.ones([mval,1]),Xval),axis=1)

    for i in range(n):
        lamb = lambda_vector[i]
        theta = trainLinReg(X, y, lamb)
        error_train[i], grad = cost_function(theta, X, y, 0.0)
        error_val[i], grad = cost_function(theta, Xval, yval, 0.0)

    return lambda_vector, error_train, error_val

##-------------------------------------------------------------------------
datapath = "datafiles/"
data = loadmat(datapath + "ex5data1.mat")
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
m = X.shape[0]
bias_col = np.ones([m,1])

# Test cost function
'''
theta0 = np.array([1,1])
test_result = cost_function(theta0, np.concatenate((bias_col,X),axis=1), y, 1.0)
print 'Testing the cost function.......'
print 'At point theta=', theta0,
print 'cost=', test_result[0],
print 'gradient=', test_result[1]
'''
# Perform linear fit to test set
theta1 = trainLinReg(np.concatenate((bias_col,X),axis=1), y, 0.0)

# Plot data with linear fit
xs = np.linspace(-50, 50, 100, endpoint=True)
ys = theta1[0] + theta1[1]*xs
fig1 = plt.figure()
plt.plot(X[:,0], y, 'r+', markersize=7, markeredgewidth=1.5, label='Training data')
plt.plot(xs, ys, 'b', label='Linear fit')
plt.xlim(-50, 40)
plt.xlabel('Change in water level')
plt.ylabel('Outflow')
plt.legend(loc='upper left')
fig1.savefig('figures/waterdata_linearfit.png')

#Plot the learning curve
errors = learningcurve(X, y, Xval, yval, 1.0)

fig2 = plt.figure()
mvals = np.linspace(1, m, m)
plt.plot(mvals, errors[0], 'b', label='Training error')
plt.plot(mvals, errors[1], 'g', label='Cross validation error')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(loc='upper right')
fig2.savefig('figures/learningcurve_highbias.png')

#Perform polynomial mapping to order 8, and feature scaling
p=8
X_poly = poly_map(X,p)
X_poly, means, stdevs = feat_norm(X_poly) 

Xval_poly = poly_map(Xval,p)
Xval_poly = (Xval_poly - means)/stdevs
Xtest_poly = poly_map(Xtest,p) 
Xtest_poly = (Xtest_poly - means)/stdevs

#Train the polynomial hypothesis
lamb = 1.0
thetap = trainLinReg(np.concatenate((bias_col,X_poly),axis=1), y, lamb)

#Plot test data with polynomial fits
h = np.dot(np.concatenate((bias_col, X_poly), axis=1), thetap.reshape([p+1,1]))
h = h.flatten()
x = X[:,0]
h = h[np.argsort(x)]
x = x[np.argsort(x)]

fig3 = plt.figure()
plt.plot(X[:,0], y, 'r+', markeredgewidth=1.5, markersize=7, label='Training data')
plt.plot(x, h, 'm', label='Polynomial fit')
plt.xlabel('Change in water level')
plt.ylabel('Outflow')
plt.legend(loc='upper left')
fig3.savefig('figures/waterdata_polyfit.png')

#Plot the learning curve for the polynomial case to see overfititng
errors = learningcurve(X_poly, y, Xval_poly, yval, lamb)

fig4 = plt.figure()
plt.plot(mvals, errors[0], 'b', label='Training error')
plt.plot(mvals, errors[1], 'r', label='Cross validation error')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(loc='upper right')
fig4.savefig('figures/learningcurve_highvariance.png')

#Calculate and plot validation curve (errs as fcn of lambda)
lambda_values, error_train, error_val = validationcurve(X_poly, y, Xval_poly, yval)

fig5 = plt.figure()
plt.plot(lambda_values, error_train, 'b', label='Training error')
plt.plot(lambda_values, error_val, 'g', label='Cross validation error')
plt.xlabel(r'$\lambda$')
plt.ylabel('Error')
plt.legend(loc='upper right')
fig5.savefig('figures/validationcurve.png')

#Pick best lambda and evaluate error on test set (this error has lambda = 0 once calculated)
lamb_min = lambda_values[np.argmin(error_val)]
theta = trainLinReg(np.concatenate((bias_col,X_poly),axis=1), y, lamb_min)
test_error, grad = cost_function(theta, np.concatenate((np.ones([Xtest_poly.shape[0],1]), Xtest_poly),axis=1), ytest, 0.0)
print 'Optimal value of lambda: ', lamb_min
print 'Test error: ', test_error


