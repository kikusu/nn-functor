import numpy

from nn_functor import functions


class Linear(functions.Learn):

    def implement(self, a, p):
        x = a[0]
        w, b0 = p

        return w.dot(x) + b0

    def update(self, a, b, p):
        i = self.implement(a, p)

        x = a[0]
        w, b0 = p

        u_w = w - self.eps * (i - b)[:, None].dot(x[None, :])
        u_b0 = b0 - self.eps * (i - b)
        return u_w, u_b0

    def request(self, a, b, p):
        i = self.implement(a, p)

        x = a[0]
        w, b0 = p

        return x - w.T.dot(i - b),


class LinearNode(functions.Node):

    def __init__(self, in_size, out_size, eps):
        super().__init__(Linear(eps))

        self.param_name = [
            "w", "b"
        ]

        self.w = numpy.random.randn(out_size, in_size)
        self.b = numpy.random.randn(out_size)
