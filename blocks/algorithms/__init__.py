"""Training algorithms."""
import logging
import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import theano
from six import add_metaclass
from theano import tensor

from blocks.graph import ComputationGraph
from blocks.utils import named_copy, shared_floatx
from blocks.theano_expressions import l2_norm

logger = logging.getLogger(__name__)


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
    cost : :class:`~tensor.TensorVariable`
        The objective to be minimized.
    params : list of :class:`~tensor.TensorSharedVariable`, optional
        The parameters to be tuned. If ``None``, all shared variables of
        `cost` computation graph will be considered parameters.

    Attributes
    ----------
    updates : list of :class:`~tensor.TensorSharedVariable` updates
        Updates to be done for every batch. It is required that the
        updates are done using the old values of optimized parameters.
    cost : :class:`~tensor.TensorVariable`
        The objective to be minimized.
    params : list of :class:`~tensor.TensorSharedVariable`
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
                       else ComputationGraph(cost).shared_variables)
        self._cost_computation_graph = ComputationGraph(self.cost)
        self._updates = []

    @property
    def inputs(self):
        """Return inputs of the cost computation graph.

        Returns
        -------
        inputs : list of :class:`~tensor.TensorVariable`
            Inputs to this graph.

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
        updates : list of tuples or :class:`~collections.OrderedDict`
            The updates to add.

        """
        if isinstance(updates, OrderedDict):
            updates = list(updates.items())
        if not isinstance(updates, list):
            raise ValueError
        self.updates.extend(updates)


variable_mismatch_error = """

Blocks tried to match the sources ({sources}) of the training dataset to \
the names of the Theano variables ({variables}), but failed to do so. \
If you want to train on a subset of the sources that your dataset provides, \
pass the `sources` keyword argument to its constructor. """


class GradientDescent(DifferentiableCostMinimizer):
    """A base class for all gradient descent algorithms.

    By "gradient descent" we mean a training algorithm of the following
    form:

    .. code-block::  python

        for batch in data:
            steps = step_rule.compute_steps(params, gradients_wr_params)
            for param in params:
                param += steps[param]

    Parameters
    ----------
    step_rule : instance of :class:`StepRule`, optional
        An object encapsulating most of the algorithm's logic. Its
        `compute_steps` method is called to get Theano expression for
        steps. Note, that the step rule
        might have a state, e.g. to remember a weighted sum of gradients
        from previous steps like it is done in gradient descent with
        momentum. If ``None``, an instance of :class:`SteepestDescent` is
        created.
    gradients : dict, optional
        A dictionary mapping a parameter to an expression for the cost's
        gradient with respect to the parameter. If ``None``, the gradient
        are taken automatically using :func:`theano.gradient.grad`.

    Attributes
    ----------
    gradients : dict
        The gradient dictionary.
    step_rule : instance of :class:`StepRule`
        The step rule.

    """
    def __init__(self, step_rule=None, gradients=None, **kwargs):
        super(GradientDescent, self).__init__(**kwargs)
        self.gradients = gradients
        if not self.gradients:
            logger.info("Taking the cost gradient")
            self.gradients = dict(
                zip(self.params, tensor.grad(self.cost, self.params)))
            logger.info("The cost gradient computation graph is built")
        self.step_rule = step_rule if step_rule else SteepestDescent()

        self.total_gradient_norm = named_copy(l2_norm(self.gradients.values()),
                                              "total_gradient_norm")
        self.steps, self.step_rule_updates = (
            self.step_rule.compute_steps(self.gradients))
        self.total_step_norm = named_copy(l2_norm(self.steps.values()),
                                          "total_step_norm")

    def initialize(self):
        logger.info("Initializing the training algorithm")
        all_updates = self.updates
        # Note: the gradients are computed in the same order in which
        # the parameters were given. Keep it like that to ensure
        # reproducibility.
        for param in self.params:
            all_updates.append((param, param + self.steps[param]))
        all_updates += self.step_rule_updates
        self._function = theano.function(self.inputs, [], updates=all_updates)
        logger.info("The training algorithm is initialized")

    def process_batch(self, batch):
        if not set(batch.keys()) == set([v.name for v in self.inputs]):
            raise ValueError("mismatch of variable names and data sources" +
                             variable_mismatch_error.format(
                                sources=batch.keys(),
                                variables=[v.name for v in self.inputs]))
        ordered_batch = [batch[v.name] for v in self.inputs]
        self._function(*ordered_batch)


@add_metaclass(ABCMeta)
class StepRule(object):
    """A rule to compute steps for a gradient descent algorithm."""
    def compute_step(self, param, gradient):
        """Build a Theano expression for the step for a parameter.

        This method is called by default implementation of
        :meth:`compute_steps`, it relieves from writing a loop each time.

        Parameters
        ----------
        param : :class:`~tensor.TensorSharedVariable`
            The parameter.
        gradient : :class:`~tensor.TensorVariable`
            The gradient of the cost with respect to the parameter.

        Returns
        -------
        step : :class:`~theano.Variable`
            Theano variable for the step to take.
        updates : list
            A list of tuples representing updates to be performed.

        """
        raise NotImplementedError

    def compute_steps(self, gradients):
        """Build a Theano expression for steps for all parameters.

        Override this method if you want to process the gradient
        with respect to all parameters as a whole, not parameter-wise.

        Parameters
        ----------
        gradients : OrderedDict
            An :class:`~OrderedDict` of
            (:class:`~tensor.TensorSharedVariable`
            :class:`~tensor.TensorVariable`) pairs. The keys are the
            parameters being trained, the values are the expressions for
            the gradients of the cost with respect to the parameters.

        Returns
        -------
        steps : OrderedDict
            A dictionary of the proposed steps in the same form as
            `gradients`.
        updates : list
            A list of tuples representing updates to be performed.

        """
        parameter_wise = [self.compute_step(param, gradients[param])
                          for param in gradients]
        (steps, updates) = zip(*parameter_wise)
        steps = OrderedDict((param, step) for param, step
                            in zip(gradients.keys(), steps))
        updates = list(itertools.chain(*updates))
        return steps, updates


class SteepestDescent(StepRule):
    """A step in the direction opposite to the gradient.

    Parameters
    ----------
    learning_rate : float
        The learning rate by which the gradient is multiplied to produce
        the descent step.

    Attributes
    ----------
    learning_rate : :class:`~tensor.TensorSharedVariable`
        The shared variable storing the learning rate used.

    """
    def __init__(self, learning_rate=1.0):
        self.learning_rate = shared_floatx(learning_rate)

    def compute_step(self, param, gradient):
        return -self.learning_rate * gradient, []


class Momentum(StepRule):
    """Accumulates the step with exponential discount.

    Parameters
    ----------
    momentum : float, optional
        The momentum coefficient.

    """
    def __init__(self, momentum=0.):
        self.momentum = shared_floatx(momentum)

    def compute_step(self, param, gradient):
        velocity = shared_floatx(param.get_value() * 0.)
        step = self.momentum * velocity + gradient
        updates = [(velocity, step)]
        return step, updates


class AdaDelta(StepRule):
    """Adapts the step size over time using only first order information.

    Parameters
    ----------
    decay_rate : float, optional
        Decay rate in [0, 1]. Defaults to 0.
    epsilon : float, optional
        Stabilizing constant for RMS. Defaults to 1e-7.

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.

    """
    def __init__(self, decay_rate=0., epsilon=1e-7):
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay rate needs to be in [0, 1]")
        self.decay_rate = shared_floatx(decay_rate)
        self.epsilon = shared_floatx(epsilon)

    def compute_step(self, param, gradient):
        mean_square_grad_tm1 = shared_floatx(param.get_value() * 0.)
        mean_square_delta_x_tm1 = shared_floatx(param.get_value() * 0.)

        mean_square_grad_t = (
            self.decay_rate * mean_square_grad_tm1 +
            (1 - self.decay_rate) * tensor.sqr(gradient)
        )

        rms_delta_x_tm1 = tensor.sqrt(mean_square_delta_x_tm1 + self.epsilon)
        rms_grad_t = tensor.sqrt(mean_square_grad_t + self.epsilon)
        delta_x_t = - rms_delta_x_tm1 / rms_grad_t * gradient

        mean_square_delta_x_t = (
            self.decay_rate * mean_square_delta_x_tm1 +
            (1 - self.decay_rate) * tensor.sqr(delta_x_t)
        )

        step = delta_x_t
        updates = [(mean_square_grad_tm1, mean_square_grad_t),
                   (mean_square_delta_x_tm1, mean_square_delta_x_t)]
        return step, updates


class GradientClipping(StepRule):
    """Clips the total gradient to make it not exceed a threshold.

    Parameters
    ----------
    threshold : float, optional
        The maximum permitted L2 norm for the gradient. The gradient
        will be rescaled to be not higher than this quanity.
        If ``None``, no rescaling will be applied.

    Attributes
    ----------
    threshold : :class:`.tensor.TensorSharedVariable`
        The shared variable storing the clipping threshold used.

    """
    def __init__(self, threshold=None):
        if threshold:
            self.threshold = theano.shared(threshold)

    def compute_steps(self, gradients):
        if not hasattr(self, 'threshold'):
            return gradients
        norm = l2_norm(gradients.values())
        multiplier = tensor.switch(norm < self.threshold,
                                   1, self.threshold / norm)
        steps = OrderedDict(
            (param, gradient * multiplier)
            for param, gradient in gradients.items())
        return steps, []


class CompositeRule(StepRule):
    """Chains several step rules.

    Parameters
    ----------
    components : list of :class:`StepRule`
        The learning rules to be chained. The rules will be applied in the
        order as given.

    """
    def __init__(self, components):
        self.components = components

    def compute_steps(self, gradients):
        result = gradients
        updates = []
        for rule in self.components:
            result, more_updates = rule.compute_steps(result)
            updates += more_updates
        return result, updates
