# -*- coding: utf-8 -*-

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

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    
    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    
    return (w, loss)

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