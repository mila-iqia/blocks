import logging
from abc import ABCMeta, abstractmethod

from theano import tensor
from theano import Variable

from blocks.graph import Cost
from blocks.select import Selector
from blocks.serialization import save_params, load_params

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

    def __init__(self, bricks, cost):
        if not isinstance(bricks, Selector):
            bricks = Selector(bricks)
        if isinstance(cost, Variable):
            cost = Cost(cost)
        assert isinstance(cost, Cost)
        self.bricks = bricks
        self.cost = cost

    @property
    def params(self):
        # Caching for speed
        if not hasattr(self, "_params"):
            self._params = self.bricks.get_params().values()
        return self._params

    @property
    def train_cost(self):
        # Caching to simplify the computation graph
        return self.cost.actual_cost()

    @property
    def param_grads(self):
        # Caching to simplify the computation graph
        if not hasattr(self, "_grads"):
            self._grads = [tensor.grad(self.train_cost, p)
                           for p in self.params]
        return self._grads

    @property
    def inputs(self):
        return self.cost.inputs

    @property
    def properties(self):
        return []

    @property
    def updates(self):
        return []

    @property
    def params_grad_scale(self):
        return [1.0] * len(self.params)

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
        save_params(self.bricks, path)

    def load(self, path):
        load_params(self.bricks, path)


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
