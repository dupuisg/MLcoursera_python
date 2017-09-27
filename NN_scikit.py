import numpy as np
import time
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

start_time = time.time()

datapath = "datafiles/"
data = loadmat(datapath + "ex4data1.mat")
weights = loadmat(datapath + "ex4weights.mat")

num_labels = 10
num_input_units = 400
num_hidden_units = 25

X = data['X']
y = data['y']
y[np.where(y==10)] = 0
m = len(y)
y = y.reshape(m)
Y = np.zeros([m, num_labels])
for i in range(m):
    Y[i,y[i]] = 1

mlp_cl = MLPClassifier(activation='logistic', hidden_layer_sizes=(num_hidden_units,), solver='lbfgs')
mlp_cl.fit(X,Y)

predictions = mlp_cl.predict(X)
p = np.argmax(predictions, axis=1)
accuracy = 100.0*np.mean(p == y)
print 'Accuracy on training set: ', accuracy, "%"

end_time = time.time()
print 'Time to run: ', end_time - start_time

