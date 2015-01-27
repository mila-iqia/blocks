"""Evaluate Theano expressions on auxiliary data and during training."""
import logging
from abc import ABCMeta, abstractmethod

from six import add_metaclass
from theano import tensor

from blocks.utils import shared_like

logger = logging.getLogger(__name__)


@add_metaclass(ABCMeta)
class AggregationScheme(object):
    """Specify how incrementally evaluate a Theano variable on a dataset.

    An AggregationScheme allocates :class:`VariableAggregator`s
    that can incrementally compute the value of a Theano variable on a
    full datset by aggregating partial results computed on multiple
    batches.

    The AggregationScheme should be attached via the tag
    `aggregation_scheme` to a Theano variable which computes the desired
    value on a single batch.

    Parameters
    ----------
    expression: Theano variable
        expression that computes the desired value on a single batch.

    """
    @abstractmethod
    def get_aggregator(self):
        """Return a new Aggregator for this variable."""
        pass


class Aggregator(object):
    """An Aggregator incrementally evaluates a Theano variable on a dataset.

    .. warning::
        The Aggregators should never be created directly. Instead use the
        :meth:`AggregationScheme.get_aggregator` method.

    Example usages are:

    * compute the mean of some value over examples, sequence lengths etc.
    * track a parameter of a model
    * monitor a penalty

    The Aggregator maintains a set of Theano sharer values called
    accumulators and specifies how they shoud be initialized, and
    updated with incremental calculations. Finally, it
    provides a Theano expression that reads the accumulators
    and computes the final value.

    Parameters
    ----------
    aggregation_scheme : :class:`AggregationScheme`
        The aggregation scheme that constructed this Aggregator
    initialization_updates : list of Theano updates
        Updates that specify how to initialize shared variables of
        this Aggregator. *Can only refer to shared variables and
        constants.*
    accumulation_updates : list of Theano updates
        Updates that specify how a new batch of data gets processed
        by this Aggregator. *Can refer to model inputs.*
    readout_expression : Theano variable
        Theano variable that computes the final value based on accumulated
        partial results. *readout_expression must only consist of shared
        variables and constants.*

    Attributes
    ----------
    All constructor parameters are accessible as attributes.

    """
    def __init__(self, aggregation_scheme, initialization_updates=None,
                 accumulation_updates=None, readout_expression=None):
        self.aggregation_scheme = aggregation_scheme
        self.readout_expression = readout_expression

        if initialization_updates is None:
            initialization_updates = []
        if accumulation_updates is None:
            accumulation_updates = []
        self.initialization_updates = initialization_updates
        self.accumulation_updates = accumulation_updates


class Mean(AggregationScheme):
    """Aggregation scheme which computes the mean.

    Parameters
    ----------
    numerator : Theano variable
        Theano expression for the numerator e.g. the likelihood
    denominator : Theano variable
        Theano expression for the denominator e.g. the batch size

    """
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def get_aggregator(self):
        numerator_acc = shared_like(self.numerator)
        denominator_acc = shared_like(self.denominator)
        initialization_updates = [(numerator_acc, 0.0),
                                  (denominator_acc, 0.0)]
        accumulation_updates = [(numerator_acc,
                                 numerator_acc + self.numerator),
                                (denominator_acc,
                                 denominator_acc + self.denominator)]
        aggregator = Aggregator(aggregation_scheme=self,
                                initialization_updates=initialization_updates,
                                accumulation_updates=accumulation_updates,
                                readout_expression=(numerator_acc /
                                                    denominator_acc))
        return aggregator


def mean(numerator, denominator=1.0):
    """Mean of quantity (numerator) over a number (denominator) values."""
    expression = numerator / denominator
    expression.tag.aggregation_scheme = Mean(numerator, denominator)
    expression.name = numerator.name
    return expression


class _DataIndependent(AggregationScheme):
    """Dummy aggregation scheme for values that don't depend on data."""
    def __init__(self, variable):
        self.variable = variable

    def get_aggregator(self):
        return Aggregator(aggregation_scheme=self,
                          initialization_updates=[],
                          accumulation_updates=[],
                          readout_expression=self.variable)


class TakeLast(AggregationScheme):
    """Aggregation scheme which remembers only the last value."""
    def __init__(self, variable):
        self.variable = variable

    def get_aggregator(self):
        self.storage = shared_like(self.variable)
        return Aggregator(aggregation_scheme=self,
                          initialization_updates=[
                              (self.storage, tensor.zeros_like(self.storage))],
                          accumulation_updates=[(self.storage, self.variable)],
                          readout_expression=self.storage)
