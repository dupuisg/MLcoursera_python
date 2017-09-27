import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans

def feat_norm(X):

    mus = np.mean(X, axis=0)
    sigmas = np.std(X, axis=0)

    normed_X = (X - mus)/sigmas

    return normed_X, mus, sigmas

def init_centroids(X, K):
    m, n = X.shape
    indices = np.random.permutation(m)
    centroids = X[indices[:K], :]

    return centroids

def closest_centroids(X, centroids):
    K = centroids.shape[0]
    m, n = X.shape
    
    ids = np.zeros(m)
    for i in range(m):
        distances = np.sum((centroids - X[i,:])**2, axis=1)        
        ids[i] = np.argmin(distances)

    return ids

def compute_centroids(X, ids, K):
    m, n = X.shape
    centroids = np.zeros([K, n])

    for k in range(K):
        xc = X[ids == k, :]
        #xc = X[np.where(ids == k)[0], :]
        centroids[k,:] = np.sum(xc, axis=0)/len(xc)

    return centroids

def run_Kmeans(X, initial_centroids, max_iter):

    m, n = X.shape
    K = len(initial_centroids)
    centroids = initial_centroids

    print 'Running K-means...........'

    for i in range(max_iter):
        ids = closest_centroids(X, centroids)
        centroids = compute_centroids(X, ids, K) 

    print 'Finished,', max_iter, 'iterations'
    return centroids, ids

## ----------------------------------------------------------------

# Import data
datapath = 'datapath/'
data = loadmat(datapath + 'ex7data2.mat')
X = data['X']

# Run K-means
K = 3
initial_centroids = init_centroids(X, K)
centroid_locations, centroid_ids = run_Kmeans(X, initial_centroids, max_iter=50)

# Plot the original data, and the data coloured by cluster, with cluster centroids
cluster_color = ['crimson', 'lightskyblue', 'seagreen', 'darkorange']
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)

plt.plot(X[:,0], X[:,1], 'ko', markersize = 6, markeredgewidth=0.8, fillstyle='none')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.subplot(1,2,2)
for i in range(K):
    plt.plot(X[centroid_ids==i,0], X[centroid_ids==i,1], marker='o', markersize=6, color=cluster_color[i], linestyle='none', alpha=0.8)
    plt.plot(centroid_locations[i,0], centroid_locations[i,1], marker='x', markersize=7, markeredgewidth=2, color='k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('figures/cluster_example.png')

#------------------------------- Clustering with scikit -------------------------------
clst = KMeans(n_clusters=3, init='random')
clst.fit(X)
centroid_locations = clst.cluster_centers_
centroid_ids = clst.labels_

## Plot scikit results
cluster_color = ['indigo', 'steelblue', 'darkorange']
plt.figure(figsize=(10,10))

for i in range(K):
    plt.plot(X[centroid_ids==i,0], X[centroid_ids==i,1], marker='o', markersize=8, color=cluster_color[i], linestyle='none', alpha=0.8)
    plt.plot(centroid_locations[i,0], centroid_locations[i,1], marker='x', markersize=9, markeredgewidth=2, color='k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('figures/cluster_example_sk.png')

#------------------------------- Imagine compression w Kmeans -------------------------------
A = plt.imread(datapath + 'bird_small.png')
pix_rows, pix_cols, vals = A.shape
A = A.reshape([pix_rows*pix_cols, vals])

clst = KMeans(n_clusters=16, init='random')
clst.fit(A)

A_compressed = clst.cluster_centers_[clst.labels_]
A_compressed = A_compressed.reshape([pix_rows, pix_cols, vals])
#A_compressed = np.zeros(A.shape)
#for i in range(16):
#    A_compressed[clst.labels_ == i, :] = clst.cluster_centers_[i]
plt.imsave('figures/bird_image_compressed.png', A_compressed)
