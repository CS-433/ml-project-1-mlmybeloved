# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def load_data(train=True):
    """Load data."""
    path_dataset = "train.csv" if train else "test.csv"
    values = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=range(2, 32))
    res = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={1: lambda x: 0 if b"b" in x else 1})
    
    return res, values

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
