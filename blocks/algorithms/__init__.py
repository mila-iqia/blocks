"""Training algorithms."""
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import theano
from six import add_metaclass
from theano import tensor

from blocks.graph import ComputationGraph


@add_metaclass(ABCMeta)
class TrainingAlgorithm(object):
    """Base class for training algorithms.

    A training algorithm object has a simple life-cycle.
    First it is initialized by calling its :meth:`initialize` method.
    At this stage, for instance, Theano functions can be compiled.
    After that the :meth:`process_batch` method is repeatedly
    called with a batch of training data as a parameter.

    """
    @abstractmethod
    def initialize(self):
        """Initialize the training algorithm."""
        pass

    @abstractmethod
    def process_batch(self, batch):
        """Process a batch of training data.

        Attributes
        ----------
        batch : dict
            A dictionary of (source name, data) pairs.

        """
        pass


class DifferentiableCostMinimizer(TrainingAlgorithm):
    """Minimizes a differentiable cost given as a Theano expression.

    Very often the goal of training is to minimize the expected value of a
    Theano expression. Batch processing in this cases typically consists of
    running a (or a few) Theano functions.
    :class:`DifferentiableCostMinimizer` is the base class for such
    algorithms.

    Parameters
    ----------
    cost : Theano variable
        The objective to be minimized.
    params : list of Theano shared variables, optional
        The parameters to be tuned. If ``None``, all shared variables of
        `cost` computation graph will be considered parameters.

    Attributes
    ----------
    updates : list of (shared variable, Theano expression) tuples
        Updates to be done for every batch. It is required that the
        updates are done using the old values of optimized parameters.
    cost : Theano variable
        The objective to be minimized.
    params : list of Theano shared variables
        The parameters to be tuned.

    Notes
    -----
        Changing `updates` attribute or calling `add_updates` after
        the `initialize` method is called will have no effect.

    .. todo::

        Some shared variables are not parameters (e.g. those created by
        random streams).

    .. todo::

        Due to a rather premature status of the :class:`ComputationGraph`
        class the parameter used only inside scans are not fetched
        currently.

    """
    def __init__(self, cost, params=None):
        self.cost = cost
        self.params = (params if params
                       else ComputationGraph(cost).get_shared_variables())
        self._cost_computation_graph = ComputationGraph(self.cost)
        self._updates = []

    @property
    def inputs(self):
        """Return inputs of the cost computation graph.

        Returns
        -------
            list of Theano variables

        """
        return self._cost_computation_graph.inputs

    @property
    def updates(self):
        return self._updates

    @updates.setter
    def updates(self, value):
        self._updates = value

    def add_updates(self, updates):
        """Add updates to the training process.

        The updates will be done _before_ the parameters are changed.

        Parameters
        ----------
        updates : list of tuples or :class:`OrderedDict`
            The updates to add.

        """
        if isinstance(updates, OrderedDict):
            updates = list(updates.items())
        assert isinstance(updates, list)
        self.updates.extend(updates)


class GradientDescent(DifferentiableCostMinimizer):
    """A base class for all gradient descent algorithms.

    By "gradient descent" we mean a training algorithm of the following
    form:

    .. code-block::  python

        for batch in data:
            for param in params:
                param += step.compute(grad_wr_param_on_batch)

    Parameters
    ----------
    step_rule : instance of :class:`StepRule`, optional
        An object incapsulating most of the algorithm's logic. Its
        `compute_step` method is called to get a Theano expression
        for the actual step to take for each parameter. Note, that
        the step rule might have a state, e.g. to remember a weighed
        sum of gradients from previous steps like it is done in
        gradient descent with momentum. If ``None``, an instance of
        :class:`SteepestDescent` is created.
    gradients : dict, optional
        A dictionary mapping a parameter to an expression for
        the cost's gradient with respect to the parameter. If ``None``,
        the gradient are taken automatically using `theano.tensor.grad`.

    Attributes
    ----------
    gradients : dict
        The gradient dictionary.
    step_rule : instance of :class:`StepRule`
        The step rule.

    """
    def __init__(self, step_rule=None, gradients=None, **kwargs):
        super(GradientDescent, self).__init__(**kwargs)
        self.gradients = (
            gradients if gradients
            else dict(
                zip(self.params, tensor.grad(self.cost, self.params))))
        self.step_rule = step_rule if step_rule else SteepestDescent()

    def initialize(self):
        all_updates = self.updates
        for param in self.params:
            all_updates.append((param,
                                param + self.step_rule.compute_step(
                                    param,
                                    self.gradients[param])))
        self._function = theano.function(self.inputs, [], updates=all_updates)

    def process_batch(self, batch):
        assert set(batch.keys()) == set([v.name for v in self.inputs])
        ordered_batch = [batch[v.name] for v in self.inputs]
        self._function(*ordered_batch)


@add_metaclass(ABCMeta)
class StepRule(object):
    """A rule to compute a step for a gradient descent algorithm."""
    @abstractmethod
    def compute_step(self, param, grad_wr_param):
        """Build a Theano expression for the step for a parameter.

        Parameters
        ----------
        param : Theano shared variable
            The parameter.
        grad_wr_param : Theano variable
            The expression for the gradient of the cost with respect to
            the parameter.

        Returns
        -------
            A Theano expression for the descent step.

        """
        raise NotImplemented()

    def additional_updates(self):
        """Return updates to be done in addition to parameter modification.

        Returns
        -------
            list of (Theano shared variable, Theano) expression tuples.

        """
        return []


class SteepestDescent(StepRule):
    """A step in the direction opposite to the gradient.

    Parameters
    ----------
    learning_rate : float
        The learning rate by which the gradient is multiplied to produce
        the descent step.

    Attributes
    ----------
    learning_rate : float
        The learning rate.

    """
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def compute_step(self, param, grad_wr_param):
        return -self.learning_rate * grad_wr_param
