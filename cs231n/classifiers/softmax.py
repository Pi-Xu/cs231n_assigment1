from builtins import range
import numpy as np
from random import shuffle
from numpy.ma.extras import mask_cols
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
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
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_class = W.shape[1]
    num_samples = X.shape[0]
    for i in range(num_samples):
      # compute the score
      score = X[i,:].dot(W)
      score -= np.max(score)
      # compute the loss
      loss -= np.log(np.exp(score[y[i]])/np.sum(np.exp(score)))

      # initialize the partial L/ partial y
      dy = np.zeros((1, num_class))

      for j in range(num_class):
        if j == y[i]:
          dy[0,j] = np.exp(score[j])/np.sum(np.exp(score)) - 1
        else:
          dy[0,j] = np.exp(score[j])/np.sum(np.exp(score))

      dW += X[i, :].reshape(-1,1).dot(dy)

    loss = loss/num_samples
    dW = dW/num_samples

    # add the regularization
    loss += np.sum(W*W)*reg
    dW += 2*W*reg

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # num_class = W.shape[1]
    num_samples = X.shape[0]

    # compute the score and maitain the numeric stability
    score = X.dot(W)
    score_max = - np.max(score, axis = 1, keepdims = True)
    score += score_max

    # compute the loss 
    loss = -np.sum(np.log(np.exp(score[np.arange(0,num_samples), y]) / np.sum(np.exp(score), axis = 1)))
    loss /= num_samples

    # compute ds
    ds = 1 / np.sum(np.exp(score), axis = 1, keepdims = True)

    ds = np.exp(score) * ds
    ds[np.arange(0,num_samples), y] -= 1
    
    # compute dW
    dW = X.T.dot(ds)
    dW /= num_samples

    # add the regularization term
    loss += np.sum(W*W)*reg
    dW += 2*W*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
