#!/usr/bin/env python

import numpy as np
from scipy.linalg import norm
from numpy.matlib import repmat

import sparseqr


def rbf(x, y, sigma=5.):
    X = repmat(x, y.shape[0], 1)
    Y = y.repeat(x.shape[0], axis=0)

    k = np.exp(-norm(X - Y, axis=1)**2 /
               2/sigma**2)

    dk = (X - Y)/2/sigma**2*k[:, np.newaxis]

    return k, dk