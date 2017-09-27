import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat

def feat_norm(X):

    mus = np.mean(X, axis=0)
    sigmas = np.std(X, axis=0)

    normed_X = (X - mus)/sigmas

    return normed_X, mus, sigmas

def pca(X):
    X = np.matrix(X)
    m,n = X.shape

    Sigma = (X.T * X)/m
    U, S, V = np.linalg.svd(Sigma)

    return U, S
    

def project_data(X, U, k):

    Z = X * U[:, 0:k]
    return Z

def recover_data(Z, U, k):

    X = Z * U[:, 0:k].T
    return X

#--------------------------------Example 2d/1d reduction ------------------------------------
datapath = 'datafiles/'
data = loadmat(datapath + 'ex7data1.mat')
X = data['X']

X_normed, mus, sigmas = feat_norm(X)
U, S = pca(X_normed)
print U

Z = project_data(X_normed, U, 1)
X_recovered = recover_data(Z, U, 1)

X_recovered = sigmas*np.array(X_recovered) + mus

plt.figure()
plt.plot(X[:,0], X[:,1], color='darkblue', marker='o', linestyle='none')
plt.plot(X_recovered[:,0], X_recovered[:,1], color='firebrick', marker='o', linestyle='none')
plt.savefig('figures/pca_reduction_example.png')

#--------------------------------Facial Images------------------------------------
data = loadmat(datapath + 'ex7faces.mat')
X = data['X']
m, n = X.shape #5000, 1024: 5000 32x32 images

## Display the original data, first 100 images in 10x10 grid
N = 10 # image grid size in number of images
for row in range(N):
    for col in range(N):
        n = row*N + col
        try:
            display_row = np.c_[display_row, X[n,:].reshape([32,32])]
        except NameError:
            display_row = X[n,:].reshape([32,32])
    try:
        sample_display = np.r_[sample_display, display_row]
    except NameError:
        sample_display = display_row
    del display_row        

plt.imsave('figures/faces_dataset.png', sample_display, cmap=cm.Greys)

k = 100
X_normed, mus, sigmas = feat_norm(X)
U, S = pca(X_normed)
Z = project_data(X_normed, U, k)
X_recovered = recover_data(Z, U, k)

#Display original and recovered image for one face
sample_face = X_normed[5,:].reshape([32,32])
display_image = np.c_[sample_face, X_recovered[5,:].reshape([32,32])]
plt.imsave('figures/sample_faces.png', display_image, cmap=cm.Greys)

