import numpy
import logging

from theano import tensor

from blocks.model import Model

logger = logging.getLogger(__name__)


class GroundhogModel(Model):
    """Gives Model an interface required for Groundhog."""

    @property
    def params(self):
        return self.get_params().values()

    @property
    def inputs(self):
        return self.input_variables

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
            self._grads = [tensor.grad(self.cost(), p) for p in self.params]
        return self._grads

    @property
    def params_grad_scale(self):
        return [1.0] * len(self.params)

    @property
    def train_cost(self):
        # Caching to simplify the computation graph
        if not hasattr(self, "_train_cost"):
            self._train_cost = self.cost()
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
        params = self.get_params()
        param_values = {name.replace("/", "-"): param.get_value()
                        for name, param in params.items()}
        numpy.savez(path, **param_values)

    def load(self, path):
        param_values = {name.replace("-", "/"): value
                        for name, value in numpy.load(path).items()}
        for name, value in param_values.items():
            selected = self.select(name)
            if len(selected) == 0:
                logger.error("Unknown parameter {}".format(name))
            assert len(selected) == 1
            selected = selected[0]

            assert selected.get_value().shape == value.shape
            selected.set_value(value)

        params = self.get_params()
        for name in params.keys():
            if not name in param_values:
                logger.error("No value is provided for the parameter {}"
                             .format(name))
