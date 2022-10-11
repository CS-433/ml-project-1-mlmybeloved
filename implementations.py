# -*- coding: utf-8 -*-
"""ML functions to be used in the training"""
from helpers import *
import numpy as np

def standardize(x):
    """Standardizes matrix x.
    
    Args:
        x: numpy array of shape (N,D), D is the number of features.
        
    Returns:
        x: standardized x matrix.
        mean_x: array of column means of x.
        std_x: array of column standard deviations of x.
    """
    
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def add_x_bias(x):
    """Adds bias term in x.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        x: numpy array of shape (N,D), D is the number of features.
        
    Returns:
        tx: numpy array of shape (N,D+1), created by adding a column of 1s to x.
    """
    
    N = x.shape[0]
    tx = np.c_[np.ones(N), x]
    return tx

def replace_min_999_by_col_mean(x):
    """Replace invalid values -999 by column(feature) average.
    
    Args:
        x: numpy array of shape (N,D), D is the number of features.
        
    Returns:
        x with values -999 replaced by the mean of the column they are in. (mean computed excluding all -999)
    """
    
    mask_999 = np.where(x == -999, 1, 0) # 1 where -999 are, 0 otherwise
    for i in range(x.shape[1]):
        col = x[:, i] # Get column
        mask_col = mask_999[:, i] # Get corresponding mask column
        col_mean = np.ma.masked_array(col, mask_col).mean(axis=0) # Compute mean without the -999
        x[:, i] = np.where(col == -999, col_mean, col) # Replace -999 by mean or keep column
    return x

def compute_mse(y, tx, w):
    """Calculate the mse loss.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
        
    Returns:
        the value of the MSE loss (scalar), corresponding to the input parameters w.
    """

    N = np.size(y)
    error = y-tx@w # Error vector
    return (1/N)*np.sum(error**2)

def compute_mae(y, tx, w):
    """Calculate the mae loss.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
        
    Returns:
        the value of the MAE loss (scalar), corresponding to the input parameters w.
    """

    N = np.size(y)
    error = y-tx@w # Error vector
    return (1/N)*np.sum(np.abs(error))

def compute_mse_gradient(y, tx, w):
    """Computes the MSE gradient at w.
        
    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (D, ) containing the gradient of the loss at w.
    """
    
    N = np.size(y)
    error = y-tx@w # Error vector
    return (-1/N)*tx.T@(error)

def compute_mae_gradient(y, tx, w):
    """Computes the MAE gradient at w.
        
    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (D, ) containing the gradient of the loss at w.
    """
    
    N = np.size(y)
    error = y-tx@w # Error vector
    error = np.where(error>=0, 1, -1) # error := sign(error)
    return -(1/N)*tx.T@error

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm using MSE.
        
    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.
        
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, mean squared error."""
    
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma*gradient
    mse = compute_mse(y, tx, w)
    return w, mse

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