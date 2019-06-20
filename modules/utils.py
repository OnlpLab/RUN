import pickle
import numpy
import torch
import torch.nn as nn





def sample_weights(nrow, ncol):
    """
    This is form Bengio's 2010 paper
    """

    bound = (numpy.sqrt(6.0) / numpy.sqrt(nrow+ncol) ) * 1.0
    return nn.Parameter(torch.DoubleTensor(nrow, ncol).uniform_(-bound, bound))




