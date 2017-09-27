import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import time

## See below for updated/improved version of this function
#def feat_norm(X):
     
#    m = len(X)
#    n = X.shape[1]
#    normed_X = np.empty(X.shape)

#    mus = np.mean(X, axis=0)
#    sigmas = np.std(X, axis=0)
#    stat_params = np.array([mus, sigmas])

#    for i in range(n):
#        normed_X[:,i] = (X[:,i] - mus[i])/sigmas[i] 

#    return normed_X, stat_params

def feat_norm(X):

    mus = np.mean(X, axis=0)
    sigmas = np.std(X, axis=0)

    normed_X = (X - mus)/sigmas

    return normed_X, mus, sigmas

def compute_cost(X, y, theta):
    # X, y, theta assumed to be matrices    
    m = len(X)
    J = 1/(2.0*m) * (X*theta - y).T * (X*theta - y)

    return J[0,0]

def gradient_descent(X, y, alpha, max_iters, theta_init, cnvrg=1.0e-4):
    # X, y, theta are matrices
    J_values = []
    n = X.shape[1]
    temp = np.matrix(np.zeros([n,1]))
    theta = theta_init    
    J_values += [compute_cost(X, y, theta)]

    for i in range(max_iters):
        temp = theta - (alpha/m) * X.T * (X*theta - y) 
        theta = temp
        J = compute_cost(X, y, theta)
        J_values += [J]
        if i > 0 and abs(J_values[i] - J_values[i-1])/J_values[i-1] <= cnvrg:
            print 'Cost function converged after ' + str(i) + ' iterations'
            break

    return J_values, theta

def normal_eqn(X, y):
    """ Solve for theta by normal equation method. """
    theta = (X.T * X).I * X.T * y    
    return theta

##---------------------------------- 1d fit --------------------------------------
## Import data
datapath = 'datafiles/' 
data = np.loadtxt(datapath + "ex1data1.txt", delimiter = ",")

start_time = time.clock()

X = data[:,0:-1]
y = data[:,-1:]
m = len(data)

## Perform feature scaling and add column with x0 (i.e. constant term)
X, means stdevs = feat_norm(X)
X = np.concatenate((np.ones([m,1]), X), axis=1) 
n = X.shape[1] #includes x0 in number of features
## convert data to matrices
X = np.matrix(X)
y = np.matrix(y)

## Initialize theta, learning rate and set iteration number 
theta_0 = np.zeros([n,1])
alpha = 0.03
N = 500

## Call gradient descent 
J_values, theta_fit = gradient_descent(X, y, alpha, N, theta_0, 1.0e-6)

end_time = time.clock()

#print "Best-fit hypothesis: y = ",
#for i in range(n):
#    print str(theta_fit[i,0]) + ' x_' + str(i),
#    if i < n-1:
#        print ' + ',
#    else:
#        print ""

print 'Coefficients: ' + str(theta_fit)
print 'Time to run 1d linear fit: ' + str(end_time - start_time)

y_fit = np.array(X*theta_fit).flatten()

## Plot evolution of cost function
plt.figure()
plt.plot(J_values, 'm', label='alpha=' + str(alpha))
plt.xlabel("No. of iterations")
plt.ylabel("Cost function")
plt.legend(loc='upper right')
plt.savefig("figures/costfunction_1d.png")


##---------------------------------- 2d fit --------------------------------------
data = np.loadtxt(datapath + "ex1data2.txt", delimiter = ",")

start_time = time.clock()

X = data[:,0:-1]
y = data[:,-1:]
m = len(data)

## Perform feature scaling and add column with x0 (i.e. constant term)
X, means, stdevs = feat_norm(X)
X = np.concatenate((np.ones([m,1]), X), axis=1) 
n = X.shape[1] #includes x0 in number of features
## convert data to matrices
X = np.matrix(X)
y = np.matrix(y)

## Initialize theta, learning rate and set iteration number 
theta_0 = np.zeros([n,1])
alpha = 0.1
N = 500
 
## Call gradient descent 
J_values, theta_fit = gradient_descent(X, y, alpha, N, theta_0, 1.0e-5)

end_time = time.clock()

print "Best-fit hypothesis: y = ",
for i in range(n):
    print str(theta_fit[i,0]) + ' x_' + str(i),
    if i < n-1:
        print ' + ',
    else:
        print ""

print 'Time to run 2d linear fit: ' + str(end_time - start_time)

## Plot evolution of cost function
plt.figure()
plt.plot(J_values, 'c', label='alpha=' + str(alpha))
plt.xlabel("No. of iterations")
plt.ylabel("Cost function")
plt.legend(loc='upper right')
plt.savefig("figures/costfunction_2d.png")


##---------------------------------- Fit using scikit  --------------------------------------
## 1d
data = np.loadtxt(datapath + "ex1data1.txt", delimiter = ",")

start_time = time.clock()

X = data[:,0:-1]
y = data[:,-1:]

#X, stats = feat_norm(X)

model = linear_model.LinearRegression()
model.fit(X, y)

end_time = time.clock()

print 'Fit from scikit, '
print 'Coefficients: ' + str(model.coef_)
print 'Intercept: ' + str(model.intercept_)
print 'Running time: ' + str(end_time - start_time)

sk_fit = model.predict(X)

## Plot the best fit with data - for 1D case only
plt.figure()
plt.plot(data[:,0], data[:,-1], 'rx', label='Training data')
plt.plot(data[:,0], y_fit, 'b', label='Prediction')
plt.plot(data[:,0], sk_fit, 'm', label='Scikit preciction')
plt.xlabel("Population (in 10,000s)")
plt.ylabel("Profit (in $10,000s)")
plt.xlim(0.0, data[0,-1])
plt.legend(loc='lower right')
plt.savefig("figures/fit_1d.png")

## 2d
data = np.loadtxt(datapath + "ex1data2.txt", delimiter = ",")

start_time = time.clock()

X = data[:,0:-1]
y = data[:,-1:]

model = linear_model.LinearRegression()
model.fit(X, y)

end_time = time.clock()

print 'Fit from scikit, '
print 'Coefficients: ' + str(model.coef_)
print 'Intercept: ' + str(model.intercept_)
print 'Running time: ' + str(end_time - start_time)


