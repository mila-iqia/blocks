from collections import OrderedDict

import theano
from theano import tensor
import pylearn2.costs.cost
import pylearn2.models

from blocks.select import Selector
from blocks.utils import pack


class Pylearn2Model(pylearn2.models.Model):
    supervised = False

    def __init__(self, brick, data_specs, **kwargs):
        self.brick = brick
        self.data_specs = data_specs
        super(Pylearn2Model, self).__init__(**kwargs)

    def get_params(self):
        return Selector(self.brick).get_params().values()


class Pylearn2Cost(pylearn2.costs.cost.Cost):
    """A Pylearn2 Cost instance.

    Parameters
    ----------
    cost : Theano variable
        A theano variable corresponding to the end of the cost
        computation graph.
    inputs : list of Theano variables
        The input variables of the cost computation graph. The order
        must correspond to the one of the iterator which is used
        for training.

    """
    def __init__(self, cost, inputs):
        self.cost = cost
        self.inputs = inputs

    def expr(self, model, data, **kwargs):
        assert not model.supervised
        data = pack(data)
        data = [tensor.unbroadcast(var, *range(var.ndim))
                for var in data]
        return theano.clone(self.cost,
                            replace=dict(zip(self.inputs, data)))

    def get_gradients(self, model, data, **kwargs):
        if not hasattr(self, "_grads"):
            self._grads = [tensor.grad(self.expr(model, data), p)
                           for p in model.get_params()]
        return OrderedDict(zip(model.get_params(), self._grads)), OrderedDict()

    def get_monitoring_channels(self, model, data, **kwargs):
        return OrderedDict()

    def get_data_specs(self, model):
        return model.data_specs
