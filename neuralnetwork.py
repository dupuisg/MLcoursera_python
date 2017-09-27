import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def grad_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def init_rand_weights(N_in, N_out, eps):
    return 2.0*eps*np.random.rand(N_out, N_in + 1) - eps

def init_debug(N_in, N_out):
    # For initializing test data and weights in gradient checking
    # Following example from course

    num_elements = N_out*(N_in + 1)
    W = np.linspace(1, num_elements, num_elements)
    W = np.sin(W)/10.0
    W = W.reshape([N_out, N_in +1])

    return W
     
def cost_function(theta_vector, X, Y, N_labels, N_input_units, N_hidden_units, lamb=0.0):
    ## Assumes only one hidden layer
    m = Y.shape[0]
    theta1_elem = N_hidden_units*(N_input_units+1)
    Theta1 = theta_vector[:theta1_elem].reshape((N_hidden_units, N_input_units+1))
    Theta2 = theta_vector[theta1_elem:].reshape((N_labels, N_hidden_units+1))

    A1 = np.concatenate((np.ones((1,m)), X.T), axis=0)
    A2 = sigmoid(np.dot(Theta1,A1))
    A2 = np.concatenate((np.ones((1,m)), A2), axis=0)
    h = sigmoid(np.dot(Theta2,A2))

#    J = -1.0/m * (np.trace( np.dot(Y, np.log(h)) ) + np.trace( np.dot(1-Y, np.log(1-h)) ))
    J = 0
    for i in range(m):
        J +=  np.dot(Y[i,:], np.log(h[:,i]))
        J +=  np.dot(1-Y[i,:], np.log(1 - h[:,i])) 
    J = -J/m 

    # regularization terms
    J = J + lamb/(2.0*m) * (np.trace(np.dot(Theta1[:,1:], Theta1[:,1:].T)) 
       + np.trace(np.dot(Theta2[:,1:], Theta2[:,1:].T)))
    
    ## This takes longer
    #theta1_term = 0
    #theta2_term = 0
    #for i in range(N_hidden_units):
    #    theta1_term += np.dot(Theta1[i, 1:], Theta1[i, 1:]) 

    #for i in range(N_labels):
    #    theta2_term += np.dot(Theta2[i, 1:], Theta2[i, 1:])

    #J += lamb/(2.0*m) * (theta1_term + theta2_term)

    theta1_grad = np.zeros(Theta1.shape)
    theta2_grad = np.zeros(Theta2.shape)
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)    

    for i in range(m):
        #Initialize i-th example, feed forward
        a1 = X[i,:].reshape([N_input_units,1])
        a1 = np.insert(a1, 0, 1,axis=0)
        a2 = sigmoid(np.dot(Theta1, a1))
        a2 = np.insert(a2, 0, 1,axis=0)
        a3 = sigmoid(np.dot(Theta2, a2))

        #Backpropagation of errors and accumulate gradient terms
        output = Y[i,:].reshape([N_labels,1])
        del3 = a3 - output
        del2 = np.dot(Theta2[:,1:].T, del3) * grad_sigmoid(np.dot(Theta1,a1))

        Delta1 = Delta1 + np.dot(del2, a1.T)
        Delta2 = Delta2 + np.dot(del3, a2.T)

    #Assign to gradients and add regularization terms
    theta1_grad = Delta1/m
    theta2_grad = Delta2/m

    theta1_grad[:,1:] = theta1_grad[:,1:] + (lamb/m)*Theta1[:,1:]
    theta2_grad[:,1:] = theta2_grad[:,1:] + (lamb/m)*Theta2[:,1:]

    #unroll theta_i gradients
    gradients = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))    

    return (J, gradients)

def grad_check(eps, lamb=0.0):
    #Checking gradient of cost function (as defined below) using numerical approximation

    ## Define test data, and create a small test neural network
    N_labels = 3
    N_input_units = 3
    N_hidden_units = 5
    m = 5

    X = init_debug(N_input_units -1, m)
    y = 1 + np.mod(np.linspace(1, m, m), N_labels*np.ones(m))
    Y = np.zeros([m, N_labels])
    for i in range(m):
        Y[i, y[i] - 1] = 1
    Theta1 = init_debug(N_input_units, N_hidden_units)
    Theta2 = init_debug(N_hidden_units, N_labels)

    theta_vector = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    ## Numerical approximation of gradient
    N = len(theta_vector)
    grad_approx = np.zeros(N)
    for j in range(N):
        disp = np.zeros(N)
        disp[j] = eps

        P = cost_function(theta_vector + disp, X, Y, N_labels, N_input_units, N_hidden_units, lamb)
        M = cost_function(theta_vector - disp, X, Y, N_labels, N_input_units, N_hidden_units, lamb)
        grad_approx[j] = (P[0] - M[0])/(2.0*eps)

    result = cost_function(theta_vector, X, Y, N_labels, N_input_units, N_hidden_units, lamb)
    grad_calc = result[1]

    print 'Calculated Result', '      ', 'Approximation'
    for i in range(N):
        print grad_calc[i], '      ', grad_approx[i]

    print 'Relative difference: ', np.linalg.norm(grad_calc - grad_approx)/np.linalg.norm(grad_calc + grad_approx)

def predict(nn_params, X, N_labels, N_input_units, N_hidden_units):

    theta1_elem = N_hidden_units*(N_input_units+1)
    Theta1 = nn_params[:theta1_elem].reshape((N_hidden_units, N_input_units+1))
    Theta2 = nn_params[theta1_elem:].reshape((N_labels, N_hidden_units+1))

    m = X.shape[0]
    A1 = np.concatenate((np.ones((1,m)), X.T), axis=0)
    A2 = sigmoid(np.dot(Theta1,A1))
    A2 = np.concatenate((np.ones((1,m)), A2), axis=0)
    H = sigmoid(np.dot(Theta2,A2))

    outputs = np.argmax(H, axis=0) + 1

    return outputs

## --------------------------------------------------------------------------------- ##
# Load in training data and initial weights

start_time = time.time()

datapath = "datafiles/"
data = loadmat(datapath + "ex4data1.mat")
weights = loadmat(datapath + "ex4weights.mat")
X = data['X']
y = data['y']

num_labels = 10
num_input_units = 400
num_hidden_units = 25

m = len(y)
y = y.reshape(m)
#y[np.where(y==10)] = 0 ## see above, keeping matlab convention to be consistent with weights
Y = np.zeros((m,num_labels))
for i in range(m):
    #Y[i, y[i]] = 1
    Y[i, y[i]-1] = 1 #weights and y were defined for matlab, mapping 0->10
                     #need to keep this convention 

Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

theta_vector = np.concatenate((Theta1.flatten(), Theta2.flatten()))

### Testing
'''
start_time = time.time()
cost = cost_function(theta_vector, X, Y, num_labels, num_input_units, num_hidden_units)
end_time = time.time()
print 'Value of unregularized cost function: ', cost[0]
print 'Time: ', end_time - start_time

cost = cost_function(theta_vector, X, Y, num_labels, num_input_units, num_hidden_units, lamb=1.0)
print 'Value of regularized cost function: ', cost[0]

grad_check(1.0e-4, 0.0)
'''

## Randomly initialize weights, train parameters using function minimization and calculate accuracy of prediction
Theta1_init = init_rand_weights(num_input_units, num_hidden_units, 0.12) 
Theta2_init = init_rand_weights(num_hidden_units, num_labels, 0.12)
theta_init = np.concatenate((Theta1_init.flatten(), Theta2_init.flatten()))

result = minimize(cost_function, theta_init, jac=True, args=(X, Y, num_labels, num_input_units, num_hidden_units, 1.0), method='TNC', options={'maxiter':250, 'disp':True}) 

nn_params = result.x
print result

output = predict(nn_params, X, num_labels, num_input_units, num_hidden_units)

accuracy = 100.0*np.mean(output == y)
print 'Accuracy on training set: ', accuracy, '%'

end_time = time.time()
print 'Time to run: ', end_time - start_time
