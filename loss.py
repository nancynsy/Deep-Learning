"""All the loss functions go here.

"""

from __future__ import division, print_function, absolute_import

import numpy as np


class Euclidean(object):
    """The Euclidean loss 'L = 1 / 2 || y_pred - y ||^2'.

    """

    def forward(self, y_pred, y):
        """Compute the Euclidean loss.

        Args:
            y_pred(np.array): the prediction.
            y(np.array): the ground truth.

        Return:
            The Euclidean loss.

        """
        L=0.5*np.sum((y-y_pred)**2) #sum of sample's lost
        return L

        # TODO: Forward code
        pass

    def backward(self, y_pred, y):
        """Compute the derivative of the Euclidean loss.

        Args:
            y_pred(np.array): the prediction.
            y(np.array): the ground truth.

        Returns:
            The derivative of the loss with respect to the y_pred.

        """
        d_ypred= y_pred-y
        return d_ypred

        # TODO: Backward code
        pass
