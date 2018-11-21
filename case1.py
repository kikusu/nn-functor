"""
p.17-18
"""
import itertools
import random

import numpy
import numpy.random

import nn_functor.functions
import nn_functor.functions.error
import nn_functor.functions.sigmoid
import nn_functor.var


class L1Para(nn_functor.functions.Learn):
    def implement(self, a, p):
        """(p, a) -> b

        Parameters
        ----------
        a : tuple[numpy.array]
        p : tuple[numpy.array]

        Returns
        -------
        numpy.array
        """
        a = a[0]
        p00, p01, p10, b00, b01, q0, q1, b1 = p

        return numpy.array([
            nn_functor.functions.sigmoid.sigmoid(
                nn_functor.functions.sigmoid.sigmoid(
                    q0 * nn_functor.functions.sigmoid.sigmoid(p00 * a[0] + p10 * a[1] + b00)
                )
                + nn_functor.functions.sigmoid.sigmoid(
                    q1 * nn_functor.functions.sigmoid.sigmoid(p01 * a[0] + b01)
                ) + b1
            )
        ]).reshape(1)

    def update(self, a, b, p):
        """(p, a, b) -> p

        Parameters
        ----------
        a : tuple[numpy.array]
        b : numpy.array
        p : tuple[numpy.array]

        Returns
        -------
        tuple[numpy.array]
        """
        i = self.implement(a, p)

        a = a[0]
        p00, p01, p10, b00, b01, q0, q1, b1 = p

        beta = [
            nn_functor.functions.sigmoid.sigmoid_derivative(p00 * a[0] + p10 * a[1] + b00),
            nn_functor.functions.sigmoid.sigmoid_derivative(p01 * a[0] + b01)
        ]
        gamma = nn_functor.functions.sigmoid.sigmoid_derivative(
            nn_functor.functions.sigmoid.sigmoid(
                q0 * nn_functor.functions.sigmoid.sigmoid(p00 * a[0] + p10 * a[1] + b00)
            )
            + nn_functor.functions.sigmoid.sigmoid(
                q1 * nn_functor.functions.sigmoid.sigmoid(p01 * a[0] + b01)
            ) + b1
        )

        return (
            p00 - self.eps * (i - b) * gamma * q0 * beta[0] * a[0],
            p01 - self.eps * (i - b) * gamma * q0 * beta[0] * a[1],
            p10 - self.eps * (i - b) * gamma * q1 * beta[1] * a[0],
            b00 - self.eps * (i - b) * gamma * q1 * beta[0],
            b01 - self.eps * (i - b) * gamma * q1 * beta[1],
            q0 - self.eps * (i - b) * gamma * beta[0],
            q1 - self.eps * (i - b) * gamma * beta[1],
            b1 - self.eps * (i - b) * gamma
        )

    def request(self, a, b, p):
        """(p, a, b) -> a

        Parameters
        ----------
        a: tuple[numpy.array]
        b: numpy.array
        p: tuple[numpy.array]

        Returns
        -------
        tuple[numpy.array]
        """
        i = self.implement(a, p)

        a = a[0]
        p00, p01, p10, b00, b01, q0, q1, b1 = p

        beta = [
            nn_functor.functions.sigmoid.sigmoid_derivative(p00 * a[0] + p10 * a[1] + b00),
            nn_functor.functions.sigmoid.sigmoid_derivative(p01 * a[0] + b01)
        ]

        gamma = nn_functor.functions.sigmoid.sigmoid_derivative(
            nn_functor.functions.sigmoid.sigmoid(
                q0 * nn_functor.functions.sigmoid.sigmoid(p00 * a[0] + p10 * a[1] + b00)
            )
            + nn_functor.functions.sigmoid.sigmoid(
                q1 * nn_functor.functions.sigmoid.sigmoid(p01 * a[0] + b01)
            ) + b1
        )

        return numpy.array([
            a[0] - (i - b) * gamma * (q0 * beta[0] * p00 + q1 * beta[1] * p10),
            a[1] - (i - b) * gamma * (q0 * beta[0] * p01)
        ]).reshape(2),


class L1Node(nn_functor.functions.Node):
    def __init__(self, eps):
        super().__init__(L1Para(eps))

        self.param_name = [
            "p00", "p01", "p10", "b00", "b01", "q0", "q1", "b1"
        ]

        self.p00 = numpy.random.randn(1)
        self.p01 = numpy.random.randn(1)
        self.p10 = numpy.random.randn(1)
        self.b00 = numpy.random.randn(1)
        self.b01 = numpy.random.randn(1)

        self.q0 = numpy.random.randn(1)
        self.q1 = numpy.random.randn(1)
        self.b1 = numpy.random.randn(1)


if __name__ == '__main__':
    random.seed(0)

    def f(src):
        return src[0] * src[1]


    xy = [numpy.array(i) for i in
          itertools.product(numpy.arange(0, 1, 0.01), numpy.arange(0, 1, 0.01))]

    l1 = L1Node(0.01)
    err_f = nn_functor.functions.error.MeanSquaredErrorNode()

    err_hist = []
    count = 0
    report = 1000

    for i in range(1000000):
        random.shuffle(xy)

        for src in xy:
            var_src = nn_functor.var.Var(src, has_link_info=False)
            var_dst = nn_functor.var.Var(f(src), has_link_info=False)
            v = l1(var_src)
            err = err_f(v, var_dst)
            err_f.backward_chain()
            err_f.update_chain()

            err_hist.append(err.data)
            count += 1
            if count % report == 0:
                print(
                    f"i:{count}\tsrc:{var_src}\ttrue:{var_dst}\tpred:{v}, "
                    f"max_err:{max(err_hist)}, mean_err:{sum(err_hist) / report}")
                err_hist = []
        #     break
        # break

    print("result")
    for src in xy[:5]:
        var_src = nn_functor.var.Var(src, has_link_info=False)
        v = l1(var_src)
        print(f"{src} -> {f(src)}, predict:{v.data}")
