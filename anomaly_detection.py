import numpy as np
from numpy.linalg import inv, det
import math
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def gaussian_params(X):
    m, n = X.shape
    mu = np.sum(X, axis=0)/m
    sigmasq = np.sum((X-mu)**2, axis=0)/m
 
    return mu, sigmasq #1d arrays, length n

def multivariate_gaussian(X, mu, sigma):
    m,n = X.shape
    if np.ndim(sigma) < 2:
        sigma = np.diag(sigma)
 
    X = np.matrix(X)
    mu = np.matrix(mu)
    sigma = np.matrix(sigma)
 
    sigma_inv = inv(sigma)
    norm = 1.0/((2.0*math.pi)**(n/2.0) * math.sqrt(det(sigma)))

    P = norm * np.exp( -0.5*np.diag((X-mu)*sigma_inv*(X-mu).T) )

    #P = np.zeros(m)
    #for i in range(m):
    #    P[i] = norm*np.exp(-(X[i,:]-mu) * sigma_inv * (X[i,:]-mu).T /2.0)[0,0]
            
    return P

def select_threshold(yval, pval):
    eps_min, eps_max = np.amin(pval), np.amax(pval)
    bestF1 = 0.0    
    besteps = 0.0

    for eps in np.linspace(eps_min, eps_max, 1000):
        CVpred = pval < eps

        tp = sum( (CVpred==1) & (yval==1) )
        fp = sum( (CVpred==1) & (yval==0) )
        fn = sum( (CVpred==0) & (yval==1) )

        try:
            prec = float(tp)/(tp+fp)
            rec =  float(tp)/(tp+fn)
            F1 = 2.0*prec*rec/(prec + rec)
        except ZeroDivisionError:
            continue 

        if F1 > bestF1:
            bestF1 = F1
            besteps = eps

    return besteps, bestF1

## ------------------------------------------------------------------------------

## 2-d example
datapath = 'datafiles/'
data = loadmat(datapath + 'ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']
yval = yval.flatten()

mean, var = gaussian_params(X)

pval = multivariate_gaussian(Xval, mean, var)
epsilon, Fscore = select_threshold(yval, pval)
print 'Optimal value of epsilon: ', epsilon, 'with F1 score', Fscore

xx, yy = np.meshgrid(np.linspace(0,30,100), np.linspace(0,30,100))
z = multivariate_gaussian(np.c_[xx.flatten(), yy.flatten()], mean, var)
z = z.reshape(xx.shape)
p = multivariate_gaussian(X, mean, var)
plt.figure()
plt.plot(X[:,0], X[:,1], marker='o', linestyle='none', color='darkblue', alpha=0.4)
plt.contour(xx, yy, z, levels=[1.0e-10, 1.0e-5, 1.0e-4, 1.0e-2, 2.0e-2], colors='c')
plt.plot(X[p < epsilon,0], X[p < epsilon, 1], marker='o', color='crimson', linestyle='non', markeredgewidth=2,fillstyle='none')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.savefig('figures/anomaly_detection_2d.png')

## Higher dimensional dataset
data = loadmat(datapath + 'ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']
yval = yval.flatten()

mean, var = gaussian_params(X)
pval = multivariate_gaussian(Xval, mean, var)
epsilon, Fscore = select_threshold(yval, pval)
print 'Optimal value of epsilon: ', epsilon, 'with F1 score', Fscore
num_anomalies = sum(pval < epsilon)
print num_anomalies, 'anomalies found in cross validation set'
p = multivariate_gaussian(X, mean, var)
num_anomalies = sum(p < epsilon)
print num_anomalies, 'anomalies found in training set'
