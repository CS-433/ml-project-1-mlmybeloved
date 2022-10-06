# -*- coding: utf-8 -*-
"""ML functions to be used in the training"""
from helpers import *
import numpy as np

def compute_mse(y, tx, w)
    """Calculate the mse loss
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
        
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    N = np.size(y)
    return (1/(2*N))*np.sum((y-tx@w)**2)

def compute_mse_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """

    return -1/np.size(y)*tx.T@(y-(tx@w))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm using MSE.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar."""
    
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma*grad
    mse = compute_mse(y, tx, w)
    return (w, mse)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) using MSE (batch size 1).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar."""
    
    w = initial_w
    for n_iter in range(max_iters):
        minibatch_y, minibatch_tx = next(batch_iter(y, tx, 1))
        grad = compute_mse_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad
    mse = compute_mse(y, tx, w)
    return (w, mse)

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar."""

    w = np.linalg.solve(tx.T@tx, tx.T@y)
    mse = compute_mse(y, tx, w)
    return (w, mse)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar."""
    
    w = np.linalg.inv(tx.T@tx + 2*np.size(y)*lambda_*np.eye(tx.shape[1]))@tx.T@y
    mse = compute_mse(y, tx, w)
    return (w, mse)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    
    return (w, loss)



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    
    return (w, loss)