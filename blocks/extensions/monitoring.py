"""Extensions for monitoring the training process."""
from blocks.extensions import SimpleExtension
from blocks.algorithms import DifferentiableCostMinimizer
from blocks.monitoring.evaluators import AggregationBuffer, DatasetEvaluator

PREFIX_SEPARATOR = '_'


def _add_records(log, prefix, record_tuples):
    """Helper function to add monitoring records to the log."""
    for name, value in record_tuples:
        prefixed_name = prefix + PREFIX_SEPARATOR + name
        setattr(log.current_row, prefixed_name, value)


class DataStreamMonitoring(SimpleExtension):
    """Monitors values of Theano expressions on a data stream.

    By default monitoring is done before the first and after every epoch.

    Parameters
    ----------
    expressions : list of Theano variables
        The expressions to monitor. The variable names are used as
        expression names.
    data_stream : instance of :class:`DataStream`
        The data stream to monitor on. A data epoch is requsted
        each time monitoring is done.
    prefix : str
        A prefix to add to expression names when adding records to the
        log. An underscore will be used to separate the prefix.

    """
    PREFIX_SEPARATOR = '_'

    def __init__(self, expressions, data_stream, prefix, **kwargs):
        kwargs.setdefault("after_every_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(DataStreamMonitoring, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(expressions)
        self.data_stream = data_stream
        self.prefix = prefix

    def do(self, callback_name, *args):
        """Write the values of monitored expressions to the log."""
        value_dict = self._evaluator.evaluate(self.data_stream)
        _add_records(self.main_loop.log, self.prefix, value_dict.items())


class TrainingDataMonitoring(SimpleExtension):
    """Monitors values of Theano expressions on training batches.

    Parameters
    ----------
    expressions : list of Theano variables
        The expressions to monitor. The variable names are used as
        expression names.
    prefix : str
        A prefix to add to expression names when adding records to the
        log. An underscore will be used to separate the prefix.

    Notes
    -----
    Requires the training algorithm to be an instance of
    :class:`DifferentiableCostMinimizer`.

    """
    def __init__(self, expressions, prefix, **kwargs):
        kwargs.setdefault("before_training", True)
        super(TrainingDataMonitoring, self).__init__(**kwargs)
        self._buffer = AggregationBuffer(expressions, use_take_last=True)
        self._last_time_called = -1
        self.prefix = prefix

    def do(self, callback_name, *args):
        """Write the values of monitored expressions to the log."""
        if callback_name == self.before_training.__name__:
            if not isinstance(self.main_loop.algorithm,
                              DifferentiableCostMinimizer):
                raise ValueError
            self.main_loop.algorithm.add_updates(
                self._buffer.accumulation_updates)
            self._buffer.initialize_aggregators()
        else:
            if self.main_loop.status.iterations_done == self._last_time_called:
                raise Exception("TrainingDataMonitoring.do should be invoked"
                                " no more than once per iteration")
            _add_records(self.main_loop.log, self.prefix,
                         self._buffer.get_aggregated_values().items())
            self._buffer.initialize_aggregators()
