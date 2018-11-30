import numpy

from nn_functor import error
from nn_functor.functions import ab, node

class NormMulFunction(ab.Learn):
    """R -> R"""

    def implement(self, a, p):
        return a[0] * p[0]

    def update(self, a, b, p):
        return p[0] - self.eps * (self.implement(a, p) - b) * a[0],

    def request(self, a, b, p):
        return p[0] - self.eps * (self.implement(a, p) - b) * a[0],


