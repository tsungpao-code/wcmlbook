
import numpy as np
import numpy.linalg as la
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
from scipy.linalg import cholesky, sqrtm

"""    Please write the code for Metropolis-Hastings random walk algorithm.
"""

def mhgd(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False,
         constellation_norm=None):


    return x_hat, mse