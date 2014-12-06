from collections import OrderedDict

import theano
from theano import tensor
import pylearn2.costs.cost
import pylearn2.models
import pylearn2.train
from pylearn2.space import CompositeSpace

from blocks.select import Selector
from blocks.utils import pack
from blocks.graph import Cost


class Pylearn2Model(pylearn2.models.Model):
    supervised = False

    def __init__(self, brick, **kwargs):
        """Wraps a brick to support the Pylearn2 model interface.

        Parameters
        ----------
        brick : Brick
            The brick to wrap.

        """
        self.brick = brick
        super(Pylearn2Model, self).__init__(**kwargs)

    def get_params(self):
        return Selector(self.brick).get_params().values()


class Pylearn2Cost(pylearn2.costs.cost.Cost):
    """Wraps a Theano cost to support the PyLearn2 Cost interface.

    Parameters
    ----------
    cost : Theano variable
        A theano variable corresponding to the end of the cost
        computation graph.

    """
    def __init__(self, cost):
        self.cost = cost
        self.inputs = Cost(self.cost).dict_of_inputs()

    def expr(self, model, data, **kwargs):
        """Substitutes the user's input variables with the PyLearn2 ones."""
        assert not model.supervised
        data = pack(data)
        data = [tensor.unbroadcast(var, *range(var.ndim))
                for var in data]
        return theano.clone(
            self.cost, replace=dict(zip(self.inputs.values(), data)))

    def get_gradients(self, model, data, **kwargs):
        if not hasattr(self, "_grads"):
            self._grads = [tensor.grad(self.expr(model, data), p)
                           for p in model.get_params()]
        return OrderedDict(zip(model.get_params(), self._grads)), OrderedDict()

    def get_monitoring_channels(self, model, data, **kwargs):
        return OrderedDict()

    def get_data_specs(self, model):
        return model.data_specs


class Pylearn2Train(pylearn2.train.Train):
    """Convinience wrapper over the PyLearn2 main loop.

    Sets `model.data_specs` using `dataset.data_specs`
    and the names of the input variables.

    """
    def __init__(self, dataset, model, algorithm, *args, **kwargs):
        spaces, sources = dataset.data_specs
        if isinstance(spaces, CompositeSpace):
            spaces = spaces.components
        else:
            spaces = (spaces,)
            sources = (sources,)
        input_names = list(algorithm.cost.inputs.keys())
        spaces = [spaces[sources.index(source)] for source in input_names]
        if len(spaces) > 1:
            spaces = CompositeSpace(spaces)
            sources = input_names
        else:
            spaces = spaces[0]
            sources = input_names[0]
        model.data_specs = (spaces, sources)
        super(Pylearn2Train, self).__init__(dataset, model, algorithm,
                                            *args, **kwargs)
