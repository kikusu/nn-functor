import typing

from nn_functor import error, var


class Node(object):
    """Paraの入出力情報をキャッシュしたりVarの連結情報生成する"""

    def __init__(self, para_func):
        """init

        Parameters
        ----------
        para_func : nn_functor.functions.ab.Para
        """
        self.para_func = para_func

        # パラメータ変数名
        self.param_name = []

        self._a: typing.Tuple[var.Var] = None
        self._b: var.Var = None
        self.__upstream_request = None
        self._request = None
        self.reset()

    @property
    def params(self):
        """パラメータのタプルを返す

        Returns
        -------
        tuple[numpy.array]
        """
        return tuple(getattr(self, i) for i in self.param_name)

    @params.setter
    def params(self, ps):
        """パラメータを更新する

        Parameters
        ----------
        ps : tuple[numpy.array]
        """
        for name, p in zip(self.param_name, ps):
            setattr(self, name, p)

    def reset(self):
        self._a = None
        self._b = None
        self.__upstream_request = None
        self._request = None

    def __call__(self, *a):
        return self.implement(*a)

    def implement(self, *a):
        """

        Parameters
        ----------
        a : tuple[nn_functor.var.Var]

        Returns
        -------
        tuple[nn_functor.var.Var]
        """
        for i in a:
            i.add_destination(self)
        self._a = a

        v_a = tuple(i.data for i in a)

        ret = self.para_func.implement(v_a, self.params)

        self._b = var.Var(ret, origin=self)

        return self._b

    def _upstream_request(self):
        if self.__upstream_request is None:
            self.__upstream_request = sum(
                [i.request(self._b).data for i in self._b.destinations()]
            ) # / len(self._b.destinations())

        return self.__upstream_request

    def request(self, a):
        """

        Parameters
        ----------
        a: nn_functor.var.Var
        Returns
        -------
        nn_functor.var.Var
        """
        assert hasattr(self, "_a") and hasattr(self, "_b")
        if self._request is None:
            self._request = [var.Var(i, has_link_info=False)
                             for i in self.para_func.request(
                    tuple(i.data for i in self._a),
                    self._upstream_request(),
                    self.params
                )]

        return self._request[self._a.index(a)]

    def update(self):
        try:
            self.params = self.para_func.update(
                tuple(i.data for i in self._a),
                self._upstream_request(),
                self.params
            )

        except error.NoUpdate:
            pass

    def backward_chain(self):
        for i in self._a:
            if i.origin:
                i.origin.backward_chain()

    def update_chain(self):
        self.update()
        for i in self._a:
            if i.origin:
                i.origin.update_chain()
        self.reset()


class ErrorNode(object):

    def __init__(self, error_func):
        """init

        Parameters
        ----------
        error_func : nn_functor.functions.ab.ErrorFunction
        """
        self.error_func = error_func

        self.param_name = []

        self._a = None
        self._c = None
        self.__upstream_request = None
        self._request = None
        self.reset()

    @property
    def params(self):
        """

        Returns
        -------
        tuple[numpy.array]
        """
        return set(getattr(self, i) for i in self.param_name)

    @params.setter
    def params(self, ps):
        for name, p in zip(self.param_name, ps):
            setattr(self, name, p)

    def reset(self):
        self._a = None
        self._c = None
        self.__upstream_request = None
        self._request = None

    def __call__(self, a, c):
        return self.implement(a, c)

    def implement(self, a, c):
        """

        Parameters
        ----------
        a : nn_functor.var.Var
        c : nn_functor.var.Var

        Returns
        -------
        tuple[nn_functor.var.Var]
        """
        a.add_destination(self)
        self._a = a
        self._c = c

        v_a = a.data

        ret = self.error_func.implement(v_a, c.data)

        ret = var.Var(ret, origin=self)

        return ret

    def request(self, a):
        """

        Parameters
        ----------
        a: nn_functor.var.Var
        Returns
        -------
        nn_functor.var.Var
        """
        assert a == self._a and hasattr(self, "_a") and hasattr(self, "_c")
        if self._request is None:
            self._request = var.Var(self.error_func.request(self._a.data, self._c.data),
                                    has_link_info=False)

        return self._request

    def update(self):
        pass

    def backward_chain(self):
        if self._a.origin:
            self._a.origin.backward_chain()

    def update_chain(self):
        if self._a.origin:
            self._a.origin.update_chain()
        self.reset()
