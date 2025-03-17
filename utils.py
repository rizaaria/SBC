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


@njit(parallel=True, fastmath=True)
def svm_loss(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D+1, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss: loss as single float
    - dW: gradient with respect to weights W; an array of same shape as W

    References:
    - https://github.com/mantasu/cs231n
    - https://github.com/jariasf/CS231n

    """
    X  = np.hstack((X, np.ones((X.shape[0],1)))) # the last column is 1: to allow augmentation of bias vector into W

    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in prange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in prange(num_classes):
            if j == y[i]:
                continue
            
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]    # update gradient for incorrect label
                dW[:, y[i]] -= X[i] # update gradient for correct label

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    loss += reg * np.sum(W * W) # add regularization to the loss.
    dW /= num_train   # scale gradient ovr the number of samples
    dW += 2 * reg * W # append partial derivative of regularization term

    return loss, dW


@njit(parallel=True, fastmath=True)
def softmax_loss(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss: loss as single float
    - dW: gradient with respect to weights W; an array of same shape as W

    References:
    - https://github.com/mantasu/cs231n
    - https://github.com/jariasf/CS231n
    
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    N = X.shape[0] # num samples
    X  = np.hstack((X, np.ones((X.shape[0],1)))) # the last column is 1: to allow augmentation of bias vector into W

    for i in prange(N):
        y_hat = X[i] @ W                    # raw scores vector
        y_exp = np.exp(y_hat - y_hat.max()) # numerically stable exponent vector
        softmax = y_exp / y_exp.sum()       # pure softmax for each score
        loss = loss - np.log(softmax[y[i]]) # append cross-entropy
        softmax[y[i]] = softmax[y[i]] - 1   # update for gradient
        dW += np.outer(X[i], softmax)       # gradient

    loss = loss / N + reg * np.sum(W**2)    # average loss and regularize 
    dW = dW / N + 2 * reg * W               # finish calculating gradient

    return loss, dW
