"""All the layer functions go here.
"""

from __future__ import division, print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape(tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.

    Attributes:
        W(np.array): the weights of the fully connected layer. An n-by-m matrix
            where m is the input size and n is the output size.
        b(np.array): the biases of the fully connected layer. A n-by-1 vector
            where n is the output size.

    """

    def __init__(self, shape):
#different initial setting    self.W = [[0]*shape[1]]*shape[0]
        self.W = np.random.randn(*shape)
        self.b = np.random.randn(shape[0], 1)

    def forward(self, x):
        
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """
        out_for = np.dot(self.W, x) + self.b
        
        return out_for

        # TODO: Forward code
        pass

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x(np.array): The derivative of the loss with respect to the
                input.
            dv_W(np.array): The derivative of the loss with respect to the
                weights.
            dv_b(np.array): The derivative of the loss with respect to the
                biases.

        """
        dv_x= np.dot(np.transpose(self.W), dv_y)
        dv_W= np.dot(dv_y, np.transpose(x))
        dv_b= dv_y

        # TODO: Backward code
        return dv_x, dv_W, dv_b
        pass


class Sigmoid(object):
    """Sigmoid function 'y = 1 / (1 + exp(-x))'

    """

    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """
        sig_for = 1/(1+np.exp(-x))
        return sig_for

        # TODO: Forward code
        pass

    def backward(self, x, dv_y):
        """Compute the gradient with respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            The derivative of the loss with respect to the input.

        """

        # TODO: Backward code
        sig_back=np.multiply(dv_y, np.exp(-x)/(1+np.exp(-x))**2)
        return sig_back
        pass
