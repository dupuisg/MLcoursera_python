import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def cost_function(params, Y, R, N_movies, N_users, N_features, lamb=0.0):
    X = params[:N_movies*N_features].reshape(N_movies, N_features)
    Theta = params[N_movies*N_features:].reshape(N_users, N_features)

    err_sq = (np.dot(X,Theta.T) - Y)**2
    J = np.sum(err_sq * R)/2.0 + (lamb/2.0) * ( np.sum(X**2) + np.sum(Theta**2) )

    err_term = (np.dot(X, Theta.T) - Y)*R
    X_grad = np.dot(err_term, Theta) + lamb*X
    Theta_grad = np.dot(err_term.T, X) + lamb*Theta

    grad = np.concatenate([X_grad.flatten(), Theta_grad.flatten()])

    return J, grad

def grad_check(lamb=0.0):
    #Checking gradient of cost function (as defined avove) using numerical approximation
    eps = 1.0e-4

    ## Define a small set of test data
    N_movies, N_users, N_features  = 5, 4, 3
    Xi = np.random.rand(N_movies,N_features)
    Thetai = np.random.rand(N_users, N_features)

    Y =  np.dot(Xi, Thetai.T) 
    Y[np.random.rand(N_movies,N_users) > 0.5] = 0 #remove selection of elements in Y (i.e. not rated)
    R = 1.0*(Y != 0) 
    X = np.random.randn(N_movies, N_features)
    Theta = np.random.randn(N_users, N_features)

    param_vector = np.concatenate([X.flatten(), Theta.flatten()])

    ## Numerical approximation of gradient
    N = len(param_vector)
    grad_approx = np.zeros(N)
    for j in range(N):
        disp = np.zeros(N)
        disp[j] = eps

        P = cost_function(param_vector + disp, Y, R, N_movies, N_users, N_features, lamb)
        M = cost_function(param_vector - disp, Y, R, N_movies, N_users, N_features, lamb)
        grad_approx[j] = (P[0] - M[0])/(2.0*eps)

    result = cost_function(param_vector, Y, R, N_movies, N_users, N_features, lamb)
    grad_calc = result[1]

    print 'Calculated Result', '      ', 'Approximation'
    for i in range(N):
        print grad_calc[i], '      ', grad_approx[i]

    print 'Relative difference: ', np.linalg.norm(grad_calc - grad_approx)/np.linalg.norm(grad_calc + grad_approx)

def mean_normalize(Y,R):
    m, n = Y.shape

    Y_normed = np.zeros(Y.shape)
    Ymeans = np.zeros(m)
    for i in range(m):
        Ymeans[i] = np.mean(Y[i, R[i,:]==1])
        Y_normed[i, R[i,:]==1] = Y[i, R[i,:]==1] - Ymeans[i]

    return Y_normed, Ymeans

def get_movielist(filename):
    movielist = {}
    f = open(filename, 'r')
    for line in f:
        index, title = line.split(None, 1)
        movielist[int(index)] = title.rstrip()

    return movielist

##---------------------------------------------------------------------------
datapath = 'datafiles/'
data = loadmat(datapath + 'ex8_movies.mat')
Y = data['Y']
R = data['R']

N_movies, N_users = Y.shape
N_features = 10

check_costfunction = False

# Pre-trained weights
Weights = loadmat(datapath + 'ex8_movieParams.mat')
X = Weights['X']
Theta = Weights['Theta']
N_u = Weights['num_users'][0,0]
N_m = Weights['num_movies'][0,0]
N_feat = Weights['num_features'][0,0]

if check_costfunction:
    ## Use subset of these for testing cost function
    n_movies, n_users, n_features = 5, 4, 3
    X_subset = X[:n_movies, :n_features]
    Theta_subset = Theta[:n_users, :n_features]
    Y_subset = Y[:n_movies, :n_users]
    R_subset = R[:n_movies, :n_users]

    test_params = np.concatenate([X_subset.flatten(), Theta_subset.flatten()])
    J, grad = cost_function(test_params, Y_subset, R_subset, n_movies, n_users, n_features)
    print 'Cost at test parameters:', J
    print 'Checking gradient.......'
    grad_check(0.0)

    J, grad = cost_function(test_params, Y_subset, R_subset, n_movies, n_users, n_features, 1.5)
    print 'With regularization (lambda=1.5), cost is', J
    print 'Checking gradient......'
    grad_check(1.5)

#Mean normalization on Y
Ynormed, Ymeans = mean_normalize(Y,R)

#Randomly initialize X0, Theta0 using randn
X_init = np.random.randn(N_movies, N_features)
Theta_init = np.random.randn(N_users, N_features)
initial_params = np.concatenate([X_init.flatten(), Theta_init.flatten()])

#Get parameters with minimize
lamb = 10.0
result = minimize(cost_function, initial_params, args=(Y, R, N_movies, N_users, N_features, lamb), jac=True, method='CG',options={'maxiter':100})
parameters = result.x
X = parameters[:N_movies*N_features].reshape(N_movies, N_features)
Theta = parameters[N_movies*N_features:].reshape(N_users, N_features)

#Get predictions and select for one user
pred = np.dot(X, Theta.T)
sample_pred = pred[:, 5] + Ymeans
indices = np.argsort(-sample_pred)[0:10] #indices of movies with 10 highest ratings

#Look up in movie index list
filename = datapath + 'movie_ids.txt'
movie_indices = get_movielist(filename)
print 'Ten highest rated movies:'
for idx in indices:
    print movie_indices[idx], 'rated', sample_pred[idx] #possible that the movie title contains a newline character

