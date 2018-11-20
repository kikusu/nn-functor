import numpy

from nn_functor import error
from nn_functor.functions import ab, node


def sigmoid(x):
    """

    Parameters
    ----------
    x : numpy.array

    Returns
    -------
    numpy.array
    """
    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_derivative(x):
    """

    Parameters
    ----------
    x : numpy.array

    Returns
    -------
    numpy.array
    """
    y = sigmoid(x)
    return (1.0 - y) * y


class SigmoidFunction(ab.Para):
    def __init__(self):
        super().__init__(None)

    def implement(self, a, p=None):
        del p
        x = a[0]
        return sigmoid(x)

    def update(self, a, b, p=None):
        del p, a, b
        raise error.NoUpdate()

    def request(self, a, b, p=None):
        x = a[0]
        return x - (sigmoid(x) - b) * sigmoid_derivative(x),


class SigmoidNode(node.Node):

    def __init__(self):
        super().__init__(SigmoidFunction())
