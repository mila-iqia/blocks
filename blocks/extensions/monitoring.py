"""Extensions for monitoring the training process."""
from blocks.extensions import SimpleExtension
from blocks.monitoring.evaluators import DatasetEvaluator


class DataStreamMonitoring(SimpleExtension):
    """Monitors values of Theano expressions on a data stream.

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
        for name, value in value_dict.items():
            prefixed_name = self.prefix + self.PREFIX_SEPARATOR + name
            setattr(self.main_loop.log.current_row, prefixed_name, value)
