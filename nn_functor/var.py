import numpy
import collections.abc

class Var(object):

    def __init__(self, data, origin=None, has_link_info=True):
        """

        Parameters
        ----------
        data: numpy.array
        origin: nn_functor.para.Node
        has_link_info: bool
            接続情報を保持するかどうか
        """
        self.data = numpy.asarray(data)
        if not self.data.shape:
            self.data = self.data.reshape(1)

        self._parent = None
        self._destinations = []
        self._has_link_info = has_link_info

        self.origin = origin

    @property
    def has_link_info(self):
        return self._has_link_info

    @has_link_info.setter
    def has_link_info(self, v):
        self._has_link_info = v

        if not v:
            self._destinations = []
            self._parent = None

    @property
    def origin(self):
        """

        Returns
        -------
        {None, nn_functor.para.ParaNode}
        """
        return self._parent

    @origin.setter
    def origin(self, node):
        """

        Parameters
        ----------
        node: nn_functor.para.Node
        """
        if self.has_link_info:
            self._parent = node

    def destinations(self):
        """

        Returns
        -------
        list[nn_functor.para.Node]
        """
        return self._destinations

    def add_destination(self, node):
        """

        Parameters
        ----------
        node: nn_functor.para.Node
        """
        if self.has_link_info:
            self._destinations.append(node)

    def __repr__(self):
        return f"Var({self.data})"
