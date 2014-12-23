"""Support for monitoring of values computed by bricks.

Each monitored Theano expression can be tagged with an aggregation scheme
which denotes how the value should be aggregated over multiple mini-batch
evaluations.
"""

from abc import ABCMeta, abstractmethod

from collections import OrderedDict

import logging

import theano

from blocks.utils import shared_for_expression, dict_subset, graph_inputs
from blocks.bricks import VariableRole

logger = logging.getLogger(__name__)


DEFAULT_MONITORED_ROLES = [VariableRole.MONITOR, VariableRole.COST,
                           VariableRole.ADDITIONAL_COST]


class AggregationScheme(object):
    """Specify how to incrementally compute a statistic on a dataset.

    An AggregationScheme allocates Aggregators that can incrementally
    compute a statistic on a full datset by aggregating partial results
    computed on multiple batches.

    The AggragationScheme should be attached via the tag `aggregation_scheme`
    to a theano expression which computes the desired statistic for one
    mini-batch.

    Parameters
    ----------
        extpession: theano variable
            expression that computes the desired statstic for a full batch.

    """
    __metaclass__ = ABCMeta

    def __init__(self, expression, **kwargs):
        super(AggregationScheme, self).__init__(**kwargs)
        self.expression = expression

    @abstractmethod
    def get_aggregator(self):
        """Return a new Aggregator for this variable.

        """
        pass


class Aggregator(object):
    """An Aggregator incrementally computes a statistic on a dataset.

    The Aggregators are typically created by the
    :meth:`AggragationScheme.get_aggregator` method of an AgragationScheme.

    Example usages are:
    - compute the mean of some value over examples, sequence lengths etc.
    - track a parameter of a model
    - monitor a penalty

    The Aggregator maintains a set of accumulators (Theano shared
    variables) and lists Theano updates to (re-)initialize the
    accumulators and to accummulate statistics over a batch. Finally it
    provides a Theano expression that reads the accumulators
    and computes the statistic.

    Parameters:
    -----------
        aggregation_scheme : :class:`AggregationScheme`
            The aggregation scheme that constructed this Aggregator

        initialization_updates : list of theano updates
            Updates that specify how to initialize shared variables of
            this Aggregator.

            Can only refer to shared variables and constants.

        accumulation_updates : list of theano updates
            Updates that specify how a new batch of data gets processed
            by this Aggregator.

            Can refer to model inputs.

        readout_expression : theano variables
            Theano variable that computes the final value based on accumulated
            partial results.

            Can only refer to shared variables and constants.

    """
    def __init__(self, aggregation_scheme,
                 initialization_updates=[],
                 accumulation_updates=[],
                 readout_expression=None,
                 **kwargs):
        super(Aggregator, self).__init__(**kwargs)
        self.aggregation_scheme = aggregation_scheme
        self.initialization_updates = initialization_updates
        self.accumulation_updates = accumulation_updates
        self.readout_expression = readout_expression

    @property
    def name(self):
        """Get the name of tha associated expression.
        """
        return self.scheme.expression.name


class Frac(AggregationScheme):
    """Aggregation scheme which computes a fraction numerator/denominator.

    Parameters
    ----------
        numerator : theano expression for the numerator

        denominator : theano expression for the denominator

    """
    def __init__(self, numerator, denominator, **kwargs):
        super(Frac, self).__init__(**kwargs)
        self.numerator = numerator
        self.denominator = denominator

    def get_aggregator(self):
        numerator_acc = shared_for_expression(self.numerator)
        denominator_acc = shared_for_expression(self.denominator)
        initialization_updates = [(numerator_acc, 0.0),
                                  (denominator_acc, 0.0)]
        accumulation_updates = [(numerator_acc,
                                 numerator_acc + self.numerator),
                                (denominator_acc,
                                 denominator_acc + self.denominator)]
        aggregator = Aggregator(aggregation_scheme=self,
                         initialization_updates=initialization_updates,
                         accumulation_updates=accumulation_updates,
                         readout_expression=numerator_acc / denominator_acc)
        aggregator._numerator_acc = numerator_acc
        aggregator._denominator_acc = denominator_acc
        return aggregator


def frac(numerator, denominator, name=None):
    """Compute numerator/denominator and tag it with an aggregation scheme

    """
    expression = numerator / denominator
    expression.tag.aggregation_scheme = Frac(numerator, denominator,
                                             expression=expression)
    if name is not None:
        expression.name = name
    return expression


class ModelProperty(AggregationScheme):
    """Dummy AggregationScheme for values that don't depend on data.

    """
    def __init__(self, **kwargs):
        super(ModelProperty, self).__init__(**kwargs)

    def get_aggregator(self):
        return Aggregator(aggregation_scheme=self,
                          initialization_updates=[],
                          accumulation_updates=[],
                          readout_expression=self.expression
                          )


def model_property(expression, name):
    """Copy the given expression and tag with ModelProperty AggregationScheme.
    """
    expression = expression.copy()
    expression.name = name
    expression.tag.aggregation_scheme = ModelProperty(expression)
    return expression


class Validator(object):
    """A Validator automates computing several theano variables on a dataset.

    Parameters
    ----------
    monitored_variables : list or dict
        A list of monitored variables. Or a dict from monitoring channels
        keys to variables. If a list is given, keys will be set to the
        ``name``s attribute of the variables.

        Each variable can be tagged with an :class:`~AggregationScheme`
        that specifies how the value can be computed for a data set by
        aggregating minibatches.

    """

    def __init__(self, monitored_variables):
        if isinstance(monitored_variables, dict):
            self.monitored_variables = monitored_variables
        else:
            keyed_vars = ((v.name, v) for v in monitored_variables)
            self.monitored_variables = OrderedDict(keyed_vars)
        self.inputs = graph_inputs(self.monitored_variables.values())
        self._compile()

    def _compile(self):
        initialize_updates = []
        accumulate_updates = []
        readout = OrderedDict()

        for k, v in self.monitored_variables.iteritems():
            logger.debug('Monitoring: %s', v.name)
            if not hasattr(v.tag, 'aggregation_scheme'):
                if graph_inputs([v]) == []:
                    logger.debug('Using ModelProperty aggregation scheme'
                                 ' for %s since it does not depend on'
                                 ' the data', k)
                    v.tag.aggregation_scheme = ModelProperty(expression=v)
                else:
                    logger.debug('Using the default (average over minibatches)'
                                 ' aggregation scheme for %s', k)
                    v.tag.aggregation_scheme = Frac(v, 1.0, expression=v)

            aggregator = v.tag.aggregation_scheme.get_aggregator()
            initialize_updates.extend(aggregator.initialization_updates)
            accumulate_updates.extend(aggregator.accumulation_updates)
            readout[k] = aggregator.readout_expression

        if initialize_updates:
            self._initialize_fun = theano.function([], [],
                                                   updates=initialize_updates)
        else:
            self._initialize_fun = None

        if accumulate_updates:
            self._accumulate_fun = theano.function(self.inputs,
                                                   [],
                                                   updates=accumulate_updates)
        else:
            self._accumulate_fun = None

        readout_th_fun = theano.function([], readout.values())

        def readout_fun():
            ret_vals = readout_th_fun()
            return dict(zip(readout.keys(), ret_vals))
        self._readout_fun = readout_fun

    def validate(self, data_set_view):
        """Compute the monitored statistics over an iterable data set

        Parameters
        ----------
        data_set_view: an interable over batches
            each batch must be a dict from input names to ndarrays

        Returns
        -------
        a dict from variable names (or from the keys of the
        monitored_variables argument to __init__) to the calues computed
        on the provided dataset.

        """
        if self._initialize_fun is not None:
            self._initialize_fun()

        if self._accumulate_fun is not None:
            input_names = [v.name for v in self.inputs]
            for batch in data_set_view:
                batch = dict_subset(batch, input_names)
                self._accumulate_fun(**batch)
        else:
            logger.debug('Only constant monitors are used, will not'
                         'iterate the over data!')

        return self._readout_fun()
