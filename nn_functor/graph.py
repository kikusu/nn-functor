import contextlib


class Graph(object):
    _name_stack = []

    @classmethod
    @contextlib.contextmanager
    def name_scope(cls, name: str):
        old = cls._name_stack.copy()

        cls._name_stack.append(name)
        try:
            yield
        finally:
            cls._name_stack = old

    @classmethod
    def name(cls):
        return "/".join(cls._name_stack)
