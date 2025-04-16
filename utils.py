import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_distances(X, Y, L=2):
    nX = X.shape[0]
    nY = Y.shape[0]
    
    dists = np.zeros((nX, nY))
    
    for i in prange(nX):
        for j in range(nY):
            if L == 1:   # L1-norm
                dists[i, j] = np.sum(np.abs(X[i] - Y[j]))
            elif L == 2: # L2-norm
                dists[i, j] = np.sqrt(np.sum((X[i] - Y[j])**2))
    
    return dists