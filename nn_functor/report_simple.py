import contextlib

import collections
import bidict
import numpy
import enum
import re
import itertools

from matplotlib import pylab


class AggregateStore(object):

    def __init__(self):
        self.datas = []
        self.counter = 0

    def add_data(self, v):
        self.datas.append(v)
        self.counter += 1

    def summary(self):
        def report(data_list):
            diff = numpy.diff(data_list, axis=0)

            if diff.shape == (0,):
                diff = numpy.array(0)
                print("aaa", diff, data_list)

            return {
                "mean": numpy.mean(data_list),
                "std": numpy.std(data_list),
                "max": numpy.max(data_list),
                "min": numpy.min(data_list),
                "diff_mean": numpy.mean(diff),
                "diff_std": numpy.std(diff),
                "diff_max": numpy.max(diff),
                "diff_min": numpy.min(diff),
            }

        ret = {
            "count": self.counter,
            "data": report(self.datas)
        }

        self.datas.clear()

        return ret


class Reporter(object):
    def __init__(self, store: AggregateStore, interval: int):
        self.store = store
        self.interval = interval

    def report(self, summary):
        raise NotImplementedError()

    def run(self):
        if self.store.counter % self.interval == 0:
            self.report(self.store.summary())


class PrintReporter(Reporter):
    def report(self, summary):
        print(summary)


class PlotType(enum.Enum):
    origin = enum.auto
    diff = enum.auto

    @classmethod
    def key_func(cls, target):
        if target == cls.diff:
            return lambda x: f"diff_{x}"
        else:
            return lambda x: x


class PlotTarget(enum.Enum):
    weight = "weight"
    request = "request"


class PylabReporter(Reporter):
    col = 3

    def __init__(self, collector, interval, mode=PlotType.diff):
        super().__init__(collector, interval)

        self.mode = mode

        self.history = []

    def _store_summary(self, summary):
        counter = summary["count"]
        data = summary["data"]
        key_func = PlotType.key_func(self.mode)

        mean = data["mean"]
        std = data["std"]
        max = data["max"]
        min = data["min"]
        self.history.append((counter, mean, std, max, min))

    def _print_summary(self):
        # self.fig.clf()
        try:
            from IPython.display import clear_output
            clear_output(True)
        finally:
            pass
        pylab.figure(figsize=(20, 20))
        x = numpy.array([i[0] for i in self.history])
        mean = numpy.array([i[1] for i in self.history])
        std = numpy.array([i[2] for i in self.history])
        max = numpy.array([i[3] for i in self.history])
        min = numpy.array([i[4] for i in self.history])

        pylab.fill_between(x, min, max, facecolor='r', alpha=0.1)
        pylab.fill_between(x, mean - std, mean + std, facecolor='b', alpha=0.2)
        pylab.plot(x, mean, "+-", label="mean",)
        pylab.legend()
        pylab.show()

        print(f"mean:{mean[-1]}, std:{std[-1]}, max:{max[-1]}, min:{min[-1]}")

    def report(self, summary):
        self._store_summary(summary)
        self._print_summary()
