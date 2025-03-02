import numpy as np
from numba import njit, prange


@njit(parallel=True)
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


def svm_loss(W, X, y, reg):
    '''
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, 
    and we operate on minibatches of N examples.

    Inputs:
    - W: A numpy array of shape (D+1, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W

    References:
    - https://github.com/mantasu/cs231n
    - https://github.com/jariasf/CS231n
    '''

    X  = np.hstack((X, np.ones((X.shape[0],1)))) # the last column is 1: to allow augmentation of bias vector into W

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[ np.arange(num_train), y].reshape(num_train,1)
    margin = np.maximum(0.0, scores - correct_class_scores + 1.0)
    margin[ np.arange(num_train), y] = 0.0 # do not consider correct class in loss
    loss = margin.sum() / num_train + reg * np.sum(W * W)

    # Compute gradient
    margin[margin > 0.0] = 1.0
    valid_margin_count = margin.sum(axis=1)
    # Subtract in correct class (-s_y)
    margin[np.arange(num_train), y] -= valid_margin_count
    dW = (X.T).dot(margin) / num_train

    # Regularization gradient
    dW = dW + 2.0 * reg * W

    return loss, dW