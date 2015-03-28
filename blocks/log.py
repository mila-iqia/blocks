"""The event-based main loop of Blocks."""
from collections import defaultdict
from numbers import Integral

try:
    from pandas import DataFrame
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TrainingLog(defaultdict):
    """Base class for training logs.

    A training log stores the training timeline, statistics and other
    auxiliary information. Information is stored as a nested dictionary,
    ``log[time][key]``. An entry without stored data will return an empty
    dictionary (like ``defaultdict(dict)``).

    In addition to the set of records displaying training dynamics, a
    training log has a :attr:`status` attribute, which is a dictionary with
    data that is not bound to a particular time.

    Attributes
    ----------
    status : dict
        A dictionary with data representing the current state of training.
        By default it contains ``iterations_done``, ``epochs_done`` and
        ``_epoch_ends`` (a list of time stamps when epochs ended).

    """
    def __init__(self):
        super(TrainingLog, self).__init__(dict)
        self.status = {
            'iterations_done': 0,
            'epochs_done': 0,
            '_epoch_ends': []
        }

    def __reduce__(self):
        constructor, args, _, _, items = super(TrainingLog, self).__reduce__()
        return constructor, (), self.__dict__, _, items

    def __getitem__(self, time):
        self._check_time(time)
        return super(TrainingLog, self).__getitem__(time)

    def __setitem__(self, time, value):
        self._check_time(time)
        return super(TrainingLog, self).__setitem__(time, value)

    def _check_time(self, time):
        if not isinstance(time, Integral) or time < 0:
            raise ValueError("time must be a positive integer")

    @property
    def current_row(self):
        return self[self.status['iterations_done']]

    @property
    def previous_row(self):
        return self[self.status['iterations_done'] - 1]

    @property
    def last_epoch_row(self):
        return self[self.status['_epoch_ends'][-1]]

    def to_dataframe(self):
        """Convert a log into a :class:`.DataFrame`."""
        if not PANDAS_AVAILABLE:
            raise ImportError("The pandas library is not found. You can"
                              " install it with pip.")
        return DataFrame.from_dict(self, orient='index')
