# -*- coding: utf-8 -*-
"""ML functions to be used in the training"""
import numpy as np

def load_data(train=True):
    """Loads data from csv files.
    
    Args:
        train: boolean indicating if we are loading the train set or the test set.
        
    Returns:
        res: numpy array of shape (N,), labels.
        values: numpy array of shape (N,D), D is the number of features.
    """
    
    path_dataset = "train.csv" if train else "test.csv"
    values = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=range(2, 32))
    res = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={1: lambda x: 0 if b"b" in x else 1})
    return res, values

def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing.
    
    Args:
        x: numpy array of shape (N,D), to split in test and train.
        y: numpy array of shape (N,), to split in test and train.
        ratio: scalar in [0,1], split ratio.
        seed: integer, seed for randomizing the split.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    np.random.seed(seed)
    split_idx = int(ratio*np.shape(x)[0])
    shuffler = np.random.permutation(np.shape(x)[0])
    x = x[shuffler]
    y = y[shuffler]
    x_tr = x[:split_idx]
    x_te = x[split_idx:]
    y_tr = y[:split_idx]
    y_te = y[split_idx:]
    return x_tr, x_te, y_tr, y_te

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

def build_poly(x, degree):
    """Builds a polynomial expansion on x of degree degree, this also adds bias.
       THIS DOES NOT ADD CROSS PRODUCTS
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,degree+1)
    """    

    expanded_X = np.ones(x.shape[0])
    for idx in range(1,degree+1):
        expanded_X = np.c_[expanded_X, x**idx]
    return expanded_X

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
        mse: scalar, mean squared error.
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma*gradient
    mse = compute_mse(y, tx, w)
    return w, mse

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) using MSE (batch size 1).
            
    Args:
        y: numpy array of shape=(N, ).
        tx: numpy array of shape=(N,D).
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters.
        max_iters: a scalar denoting the total number of iterations of SGD.
        gamma: a scalar denoting the stepsize.
        
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, mean squared error.
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        minibatch_y, minibatch_tx = next(batch_iter(y, tx, 1)) # Batch size of 1
        gradient = compute_mse_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma*gradient
    mse = compute_mse(y, tx, w)
    return w, mse

def least_squares(y, tx):
    """Computes the least squares solution.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, mean squared error.
    """

    w = np.linalg.solve(tx.T@tx, tx.T@y)
    mse = compute_mse(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """Computes ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, mean squared error.
    """
    
    w = np.linalg.inv(tx.T@tx + 2*np.size(y)*lambda_*np.eye(tx.shape[1]))@tx.T@y
    mse = compute_mse(y, tx, w)
    return w, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return w, loss