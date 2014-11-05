import numpy
import logging
from abc import ABCMeta, abstractmethod

from theano import tensor

logger = logging.getLogger(__name__)


class GroundhogIterator(object):
    """A base class for Groundhog compatible iterator.

    Has mock implementations of all required methods except `next`.

    """

    __metaclass__ = ABCMeta

    def start(self, offset):
        pass

    @abstractmethod
    def next(self):
        pass

    @property
    def next_offset(self):
        return 0


class GroundhogModel(object):
    """Wraps a model into a Groundhog compatible interface."""

    def __init__(self, model):
        self.model = model

    @property
    def params(self):
        # Caching for speed
        if not hasattr(self, "_params"):
            self._params = self.model.get_params().values()
        return self._params

    @property
    def inputs(self):
        return self.model.input_variables

    @property
    def properties(self):
        return []

    @property
    def updates(self):
        return []

    @property
    def param_grads(self):
        # Caching to simplify the computation graph
        if not hasattr(self, "_grads"):
            self._grads = [tensor.grad(self.model.cost(), p)
                           for p in self.params]
        return self._grads

    @property
    def params_grad_scale(self):
        return [1.0] * len(self.params)

    @property
    def train_cost(self):
        # Caching to simplify the computation graph
        if not hasattr(self, "_train_cost"):
            self._train_cost = self.model.cost()
        return self._train_cost

    @property
    def valid_costs(self):
        return []

    @property
    def exclude_params_for_norm(self):
        return []

    def perturb(self, **kwargs):
        return kwargs

    def get_schedules(self):
        return []

    def save(self, path):
        params = self.model.get_params()
        # numpy.savez is vulnerable to slashes in names
        param_values = {name.replace("/", "-"): param.get_value()
                        for name, param in params.items()}
        numpy.savez(path, **param_values)

    def load(self, path):
        param_values = {name.replace("-", "/"): value
                        for name, value in numpy.load(path).items()}
        for name, value in param_values.items():
            selected = self.model.select(name)
            if len(selected) == 0:
                logger.error("Unknown parameter {}".format(name))
            assert len(selected) == 1
            selected = selected[0]

            assert selected.get_value().shape == value.shape
            selected.set_value(value)

        params = self.model.get_params()
        for name in params.keys():
            if name not in param_values:
                logger.error("No value is provided for the parameter {}"
                             .format(name))


class GroundhogState(object):
    """Good default values for groundhog state."""

    def __init__(self, prefix, batch_size, learning_rate, **kwargs):
        self.prefix = prefix
        self.bs = batch_size
        self.lr = learning_rate
        self.seed = 1
        # TODO: what does it mean?
        self.patience = 1
        self.minerr = -1
        self.timeStop = 10 ** 9
        self.minlr = 0
        self.overwrite = True
        self.hookFreq = -1
        self.saveFreq = 30
        self.validFreq = 10 ** 9
        self.trainFreq = 1
        self.loopIters = 10 ** 6

        self.cutoff_rescale_length = 0.0

    def as_dict(self):
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}
