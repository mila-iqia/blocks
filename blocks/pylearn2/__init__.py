from collections import OrderedDict

from pylearn2.costs.cost import Cost as Pylearn2Cost
from pylearn2.models import Model as Pylearn2Model
from theano import tensor

from blocks.select import Selector


class BlocksModel(Pylearn2Model):
    supervised = False

    def __init__(self, brick, data_specs, application_method='apply'):
        self.application_method = application_method
        self.brick = brick
        self.data_specs = data_specs

    def get_params(self):
        return Selector(self.brick).get_params().values()


class BlocksCost(Pylearn2Cost):
    """A Pylearn2 Cost instance."""
    def __init__(self, cost, application_method='apply'):
        self.cost = cost
        self.application_method = application_method

    def expr(self, model, data, **kwargs):
        if model.supervised:
            x, y = data
        else:
            x = y = data
        return getattr(self.cost, self.application_method)(
            y, getattr(model.brick, model.application_method)(x))

    def get_gradients(self, model, data, **kwargs):
        if not hasattr(self, "_grads"):
            self._grads = [tensor.grad(self.expr(model, data), p)
                           for p in model.get_params()]
        return OrderedDict(zip(model.get_params(), self._grads)), OrderedDict()

    def get_monitoring_channels(self, model, data, **kwargs):
        return OrderedDict()

    def get_data_specs(self, model):
        return model.data_specs
