"""Evaluate Theano expressions on auxiliary data and during training."""
import logging

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import theano

from blocks.utils import (shared_like, dict_subset,
                          graph_inputs, update_instance)

logger = logging.getLogger(__name__)


class VariableAggregationScheme(object):
    """Specify how incrementally evaluate a theano variable on a dataset.

    An VariableAggregationScheme allocates :class:`VariableAggregator`s
    that can incrementally compute the value of a theano variable on a
    full datset by aggregating partial results computed on multiple
    batches.

    The VariableAggregationScheme should be attached via the tag
    `aggregation_scheme` to a theano variable which computes the desired
    value on a single batch.

    Parameters
    ----------
    expression: theano variable
        expression that computes the desired value on a single batch.

    """
    __metaclass__ = ABCMeta

    def __init__(self, expression, **kwargs):
        super(VariableAggregationScheme, self).__init__(**kwargs)
        self.expression = expression

    @abstractmethod
    def get_aggregator(self):
        """Return a new Aggregator for this variable.

        """
        pass


class Aggregator(object):
    """An Aggregator incrementally evaluates a theano variable on a dataset.

    .. warning::
        The Aggregators should never be created directly. Instead use the
        :meth:`VariableAggregationScheme.get_aggregator` method.

    Example usages are:

    * compute the mean of some value over examples, sequence lengths etc.
    * track a parameter of a model
    * monitor a penalty

    The Aggregator maintains a set of theano sharer values called
    accumulators and specifies how they shoud be initialized, and
    updated with incremental calculations. Finally, it
    provides a Theano expression that reads the accumulators
    and computes the final value.

    Parameters
    ----------
    aggregation_scheme : :class:`VariableAggregationScheme`
        The aggregation scheme that constructed this Aggregator
    initialization_updates : list of theano updates
        Updates that specify how to initialize shared variables of
        this Aggregator. *Can only refer to shared variables and
        constants.*
    accumulation_updates : list of theano updates
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
                 accumulation_updates=None, readout_expression=None, **kwargs):
        super(Aggregator, self).__init__(**kwargs)
        if initialization_updates is None:
            initialization_updates = []
        if accumulation_updates is None:
            accumulation_updates = []
        update_instance(self, locals())

    @property
    def name(self):
        """Get the name of tha associated expression.
        """
        return self.scheme.expression.name


class Mean(VariableAggregationScheme):
    """Aggregation scheme which computes the division numerator/denominator.

    Parameters
    ----------
    numerator : theano expression for the numerator
    denominator : theano expression for the denominator

    """
    def __init__(self, numerator, denominator, **kwargs):
        super(Mean, self).__init__(**kwargs)
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
        aggregator._numerator_acc = numerator_acc
        aggregator._denominator_acc = denominator_acc
        return aggregator


def mean(numerator, denominator, name=None):
    """Mean of quantity (numerator) over a number (denominator) values.

    """
    expression = numerator / denominator
    expression.tag.aggregation_scheme = Mean(numerator, denominator,
                                             expression=expression)
    if name is not None:
        expression.name = name
    else:
        expression.name = 'mean{{}, {}}'.format(numerator.name,
                                                + denominator.name)
    return expression


class _DataIndependent(VariableAggregationScheme):
    """Dummy VariableAggregationScheme for values that don't depend on data.

    """
    def __init__(self, **kwargs):
        super(_DataIndependent, self).__init__(**kwargs)

    def get_aggregator(self):
        return Aggregator(aggregation_scheme=self,
                          initialization_updates=[],
                          accumulation_updates=[],
                          readout_expression=self.expression
                          )


class DatasetEvaluator(object):
    """A DatasetEvaluator evaluates many theano expressions on a dataset.

    The DatasetEvaluator provides a do-it-all method,
    :meth:`evaluate`, which computes values of ``expressions``
    on a dataset.

    Alternatively, methods :meth:`_initialize_computation`,
    :meth:`_process_batch`, :meth:`_readout_expressions` can be used
    with a custom loop over data.

    The values computed on subsets of the given dataset
    are aggregated using the VariableAggregationSchemes provided in
    `aggregation_scheme` tags. If no tag is given, the value is **averaged
    over minibatches**. However, care is taken to ensure that variables
    which do not depend on data are not unnecessarily recomputed.

    Parameters
    ----------
    expressions : list or dict
        A list of monitored variables. Or a dict from keys to variables.
        If a list is given, keys will be set to the variables themselves.

        Each variable can be tagged with an
        :class:`VariableAggregationScheme` that specifies how the value can
        be computed for a data set by aggregating minibatches.

    """
    def __init__(self, channel_variables):
        if isinstance(channel_variables, dict):
            self.channel_variables = channel_variables
        else:
            keyed_vars = ((v, v) for v in channel_variables)
            self.channel_variables = OrderedDict(keyed_vars)
        self.inputs = graph_inputs(self.channel_variables.values())
        self._compile()

    def _compile(self):
        initialize_updates = []
        accumulate_updates = []
        readout = OrderedDict()

        for k, v in self.channel_variables.iteritems():
            logger.debug('Monitoring: %s', v.name)
            if not hasattr(v.tag, 'aggregation_scheme'):
                if graph_inputs([v]) == []:
                    logger.debug('Using _DataIndependent aggregation scheme'
                                 ' for %s since it does not depend on'
                                 ' the data', k)
                    v.tag.aggregation_scheme = _DataIndependent(expression=v)
                else:
                    logger.debug('Using the default (average over minibatches)'
                                 ' aggregation scheme for %s', k)
                    v.tag.aggregation_scheme = Mean(v, 1.0, expression=v)

            aggregator = v.tag.aggregation_scheme.get_aggregator()
            initialize_updates.extend(aggregator.initialization_updates)
            accumulate_updates.extend(aggregator.accumulation_updates)
            readout[k] = aggregator.readout_expression

        if initialize_updates:
            self._initialize_fun = theano.function([], [],
                                                   updates=initialize_updates)
        else:
            self._initialize_fun = None

        self._initialized = False
        self._input_names = [v.name for v in self.inputs]

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

    def _initialize_computation(self):
        """Initialize the aggragators to process a dataset.

        """
        self._initialized = True
        if self._initialize_fun is not None:
            self._initialize_fun()

    def _process_batch(self, batch):
        if not self._initialized:
            self._initialize_computation()
        batch = dict_subset(batch, self._input_names)
        if self._accumulate_fun is not None:
            self._accumulate_fun(**batch)

    def _readout_expressions(self):
        if not self._initialized:
            raise Exception("To readout you must first initialize, then"
                            "process batches!")
        self._initialized = False
        return self._readout_fun()

    def evaluate(self, data_set_view):
        """Compute the expressions over an iterable data set

        Parameters
        ----------
        data_set_view: an iterable over batches
            each batch must be a dict from input names to ndarrays

        Returns
        -------
        dict from variables (or from the keys of the
            monitored_variables argument to __init__) to the values
            computed on the provided dataset.

        """
        self._initialize_computation()

        if self._accumulate_fun is not None:
            for batch in data_set_view:
                self._process_batch(batch)
        else:
            logger.debug('Only constant monitors are used, will not'
                         'iterate the over data!')

        return self._readout_expressions()


class MinibatchEvaluator(object):
    """Helper evaluating several theano variables using updates.

    The MinibatchEvaluator allocates storage for each of the variables
    given to its constructor. It then provides:

    - a list of updates which should be called by the training function
      on every minibatch. These updates store computed values in the
      shared variables.

    - a function which reads the shared variables and returns a dict from
        names (or channel keys) their values.

    Parameters
    ----------
    expressions : list or dict
        A list of monitored variables. Or a dict from keys to variables.
        If a list is given, keys will be set to the variables themselves.

    """
    def __init__(self, monitored_variables):
        if isinstance(monitored_variables, dict):
            monitored_variables = monitored_variables
        else:
            keyed_vars = ((v, v) for v in monitored_variables)
            monitored_variables = OrderedDict(keyed_vars)

        self._updates = []
        self._storage = OrderedDict()
        for k, v in monitored_variables.iteritems():
            shared_v = shared_like(v)
            self._storage[k] = shared_v
            self._updates.append((shared_v, v))

    @property
    def updates(self):
        """Updates that have to be called for each minibatch.

        """
        return self._updates

    def read_expressions(self):
        """Read values of the expressions computed on the last minibatch.

        Returns
        -------
        dict from variables (or from the keys of the
            monitored_variables argument to __init__) to the values
            computed on the provided dataset.

        """
        values = OrderedDict()
        for k, sv in self._storage.iteritems():
            values[k] = sv.get_value(borrow=False)
        return values
