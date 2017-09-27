import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

## ------------------------- Example 1 ------------------------
datapath = "datafiles/"
data = loadmat(datapath + "ex6data1.mat")
X = data['X']
y = data['y']
y = y.flatten()

#Fit with svm classifier
lin_clf = svm.LinearSVC(C=1.0)
lin_clf.fit(X,y)
#print lin_clf.coef_
#print lin_clf.intercept_[0]

#For plotting boundary
x1values = np.linspace(0, 5, 51, endpoint=True)
bndry1 = -(lin_clf.intercept_[0] + lin_clf.coef_[0,0]*x1values)/lin_clf.coef_[0,1]

lin_clf = svm.LinearSVC(C=100.0)
lin_clf.fit(X,y)
#print lin_clf.coef_
#print lin_clf.intercept_[0]

#For plotting boundary
bndry2 = -(lin_clf.intercept_[0] + lin_clf.coef_[0,0]*x1values)/lin_clf.coef_[0,1]

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(10,5))
ax1.plot(X[np.where(y==0),0], X[np.where(y==0),1], 'o', markeredgecolor='k', markerfacecolor='lightskyblue', label='neg. example')
ax1.plot(X[np.where(y==1),0], X[np.where(y==1),1], '+', color='firebrick', label='pos. example') 
ax1.plot(x1values, bndry1, color='darkmagenta', linestyle='solid')
ax2.plot(X[np.where(y==0),0], X[np.where(y==0),1], 'o', markeredgecolor='k', markerfacecolor='lightskyblue', label='neg. example')
ax2.plot(X[np.where(y==1),0], X[np.where(y==1),1], '+', color='firebrick', label='pos. example') 
ax2.plot(x1values, bndry2, color='darkmagenta', linestyle='solid')
for ax in (ax1,ax2):
    ax.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('$C=1$', color='darkmagenta')
ax2.set_title('$C=100$', color='darkmagenta')
#legend
plt.savefig("figures/linear_bdry_svm.png")

## ------------------------- Example 2 -------------------------
data = loadmat(datapath + "ex6data2.mat")
X = data['X']
y = data['y']
y = y.flatten() 

clf = svm.SVC(C=1, gamma=1/(2*0.1**2))
clf.fit(X,y)
### Which svm params are which, and how to define boundary?

## Plotting boundary 
x1range = X[:,0].min(), X[:,0].max() + 0.1
x2range = X[:,1].min(), X[:,1].max() + 0.1
x1, x2 = np.meshgrid(np.arange(x1range[0], x1range[1], 0.05), np.arange(x2range[0], x2range[1], 0.05))
z = clf.predict(np.c_[x1.flatten(), x2.flatten()])
z = z.reshape(x1.shape)

contour_colours = plt.cm.get_cmap(name='Paired')
fig = plt.figure()
plt.plot(X[np.where(y==0),0], X[np.where(y==0),1], 'o', markeredgecolor='k', markerfacecolor='lightskyblue', label='neg. example')
plt.plot(X[np.where(y==1),0], X[np.where(y==1),1], '+', color='firebrick', label='pos. example')
plt.contourf(x1, x2, z, 1, colors = ('lightblue', 'lightpink'), alpha=0.6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig("figures/nonlinear_ex2_svm.png")

## ------------------------- Example 3 -------------------------
data = loadmat(datapath + "ex6data3.mat")
X = data['X']
y = data['y']
y = y.flatten()
Xval = data['Xval']
yval = data['yval']
yval = yval.flatten()

Cvalues = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
sigmavalues = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
gammavalues = 1.0/(2*sigmavalues**2)

accuracy = np.zeros([len(Cvalues), len(gammavalues)])
for i in range(len(Cvalues)):
    for j in range(len(gammavalues)):
        clf = svm.SVC(C=Cvalues[i], gamma=gammavalues[j])
        clf.fit(X,y)
        accuracy[i,j] = clf.score(Xval, yval) #np.mean(prediction != yval)

row, col = np.unravel_index(np.argmax(accuracy), accuracy.shape)
C_opt = Cvalues[row]
gamma_opt = gammavalues[col]

# Fit with optimal parameters
clf = svm.SVC(C=C_opt, gamma=gamma_opt)
clf.fit(X,y)

# Plot data with boundary
x1range = X[:,0].min(), X[:,0].max() + 0.05
x2range = X[:,1].min(), X[:,1].max() + 0.05
x1, x2 = np.meshgrid(np.arange(x1range[0], x1range[1], 0.01), np.arange(x2range[0], x2range[1], 0.05))
z = clf.predict(np.c_[x1.flatten(), x2.flatten()])
z = z.reshape(x1.shape)

fig = plt.figure()
plt.plot(X[np.where(y==0),0], X[np.where(y==0),1], 'o', markeredgecolor='k', markerfacecolor='lightskyblue', label='neg. example')
plt.plot(X[np.where(y==1),0], X[np.where(y==1),1], '+', color='firebrick', label='pos. example')
plt.contourf(x1, x2, z, 1, colors = ('lightblue', 'lightpink'), alpha=0.6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig("figures/nonlinear_ex3_svm.png")

