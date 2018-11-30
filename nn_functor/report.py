import contextlib

import collections
import bidict
import numpy
import enum
import re
import itertools

from matplotlib import pylab


class Collector(object):

    def __init__(self):
        self.nodes = {}
        self.node_names = set()
        self.data_weight = collections.defaultdict(list)
        self.data_request = collections.defaultdict(list)

        self.counter = 0

    def add_node(self, node):
        node_name = node.node_name()
        if node_name in self.node_names:
            # 登録済みなら _数字 をつける
            i = 0
            while True:
                tmp_node_name = f"{node_name}_{i}"
                if tmp_node_name not in self.node_names:
                    node_name = tmp_node_name
                    break

        self.node_names.add(node_name)
        self.nodes[node_name] = node

    def collect(self):
        for k, v in self.nodes.items():
            for p_name in v.param_name:
                self.data_weight[f"{k}/{p_name}"].append(getattr(v, p_name).copy())

            if v._upstream_request() is not None:
                self.data_request[f"{k}"].append(v._upstream_request() - v._b.data)

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
            "weight": {k: report(v) for k, v in self.data_weight.items()},
            "request": {k: report(v) for k, v in self.data_request.items()}
        }

        self.data_request.clear()
        self.data_weight.clear()

        return ret


class Reporter(object):
    def __init__(self, collector, interval):
        self.collector = collector
        self.interval = interval

    def report(self, summary):
        raise NotImplementedError()

    def run(self):
        self.collector.collect()

        if self.collector.counter % self.interval == 0:
            self.report(self.collector.summary())


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

    def __init__(self, collector, interval, re_filter=None,
                 target=PlotTarget.request, mode=PlotType.diff):
        super().__init__(collector, interval)

        if not re_filter:
            re_filter = [".*"]

        self.filter = [re.compile(i) for i in re_filter]
        self.mode = mode
        self.target = target

        self.history = collections.defaultdict(list)


    def _store_summary(self, summary):
        counter = summary["count"]
        data = summary[self.target.value]
        filtered_key = sorted([
            key
            for key in data.keys()
            if next(filter(lambda x: x.match(key), self.filter), None) is not None
        ])

        key_func = PlotType.key_func(self.mode)

        for i, key in enumerate(filtered_key):
            mean = data[key][key_func("mean")]
            std = data[key][key_func("std")]
            max = data[key][key_func("max")]
            min = data[key][key_func("min")]
            self.history[key].append((counter, mean, std, max, min))

    def _print_summary(self):
        #self.fig.clf()
        try:
            from IPython.display import clear_output
            clear_output(True)
        finally:
            pass
        pylab.figure(figsize=(20, 20))
        for i, key in enumerate(sorted(self.history.keys())):
            pylab.subplot((len(self.history) % self.col) + 1, self.col, i + 1)

            x = numpy.array([i[0] for i in self.history[key]])
            mean = numpy.array([i[1] for i in self.history[key]])
            std = numpy.array([i[2] for i in self.history[key]])
            max = numpy.array([i[3] for i in self.history[key]])
            min = numpy.array([i[4] for i in self.history[key]])

            pylab.fill_between(x, min, max, facecolor='r', alpha=0.1)
            pylab.fill_between(x, mean - std, mean + std, facecolor='b', alpha=0.2)
            pylab.plot(x, mean, label="mean")
            pylab.title(key)
        pylab.legend()
        pylab.show()

    def report(self, summary):
        self._store_summary(summary)
        self._print_summary()
