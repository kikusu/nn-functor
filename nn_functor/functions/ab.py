class Para(object):

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError


class Learn(Para):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps


class ErrorFunction(object):
    def implement(self, a, c):
        """(p, a) -> b

        Parameters
        ----------
        a : numpy.array
        c : numpy.array
            true

        Returns
        -------
        numpy.array
        """
        raise NotImplementedError

    def request(self, a, c):
        """(p, a, b) -> a

        Parameters
        ----------
        a: numpy.array
        c: numpy.array

        Returns
        -------
        tuple[numpy.array]
        """
        raise NotImplementedError
