from collections import OrderedDict
import logging

import theano

from blocks.utils import graph_inputs, dict_subset
from blocks.monitoring.aggregation import _DataIndependent, Mean
from blocks.graph import ComputationGraph

logger = logging.getLogger()


class DatasetEvaluator(object):
    """A DatasetEvaluator evaluates many Theano expressions on a dataset.

    The DatasetEvaluator provides a do-it-all method, :meth:`evaluate`,
    which computes values of ``expressions`` on a dataset.

    Alternatively, methods :meth:`_initialize_computation`,
    :meth:`_process_batch`, :meth:`_readout_expressions` can be used with a
    custom loop over data.

    The values computed on subsets of the given dataset are aggregated
    using the :class:`AggregationScheme`s provided in the
    `aggregation_scheme` tags. If no tag is given, the value is **averaged
    over minibatches**. However, care is taken to ensure that variables
    which do not depend on data are not unnecessarily recomputed.

    Parameters
    ----------
    expressions : dict or list
        If a list of Theano variables. The variable names are used as
        expression names. All the variables names must be different.

        Each variable can be tagged with an :class:`AggregationScheme` that
        specifies how the value can be computed for a data set by
        aggregating minibatches.

    """
    def __init__(self, expressions):
        self.expressions = OrderedDict(
            [(var.name, var) for var in expressions])
        if len(self.expressions) < len(expressions):
            raise ValueError(
                "Expression variables should have different names")

        self.inputs = ComputationGraph(
            list(self.expressions.values())).inputs
        self._compile()

    def _compile(self):
        """Compiles Theano functions.

        .. todo::

            The current compilation method does not account for updates
            attached to `ComputationGraph` elements. Compiling should
            be out-sourced to `ComputationGraph` to deal with it.

        """
        initialize_updates = []
        accumulate_updates = []
        readout = OrderedDict()

        for k, v in self.expressions.items():
            logger.debug('Expression to evaluate: %s', v.name)
            if not hasattr(v.tag, 'aggregation_scheme'):
                if graph_inputs([v]) == []:
                    logger.debug('Using _DataIndependent aggregation scheme'
                                 ' for %s since it does not depend on'
                                 ' the data', k)
                    v.tag.aggregation_scheme = _DataIndependent(variable=v)
                else:
                    logger.debug('Using the default (average over minibatches)'
                                 ' aggregation scheme for %s', k)
                    v.tag.aggregation_scheme = Mean(v, 1.0)

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

        self._readout_fun = theano.function([], list(readout.values()))

    def _initialize_computation(self):
        """Initialize the aggragators to process a dataset."""
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
        ret_vals = self._readout_fun()
        return dict(zip(self.expressions.keys(), ret_vals))

    def evaluate(self, data_stream):
        """Compute the expressions over a data stream.

        Parameters
        ----------
        data_stream : instance of :class:`DataStream`
            The data stream. Only the first epoch of data is used.

        Returns
        -------
        A mapping from expression names to the values computed on the
        provided dataset.

        """
        self._initialize_computation()

        if self._accumulate_fun is not None:
            for batch in data_stream.get_epoch_iterator(as_dict=True):
                self._process_batch(batch)
        else:
            logger.debug('Only constant monitors are used, will not'
                         'iterate the over data!')

        return self._readout_expressions()
