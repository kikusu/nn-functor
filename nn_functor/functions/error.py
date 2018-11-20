import numpy

from nn_functor.functions import ab, node


class MeanSquaredError(ab.ErrorFunction):
    def __init__(self):
        super().__init__()

    def implement(self, a, c):
        return 0.5 * numpy.square(a - c).mean()

    def request(self, a, c):
        return c


class MeanSquaredErrorNode(node.ErrorNode):

    def __init__(self):
        super().__init__(MeanSquaredError())
