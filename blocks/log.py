"""The event-based main loop of Blocks."""
from abc import ABCMeta, abstractmethod
from collections import defaultdict


class AbstractTrainingStatus(object):
    """The base class for objects that carry the training status.

    To support various backends (such as database tables) the
    descendants of this class are expected to override the
    `__getattr__` and `__setattr__` methods.

    Attributes
    ----------
    iterations_done : int
        The number of iterations done.
    epochs_done : int
        The number of epochs done.
    last_epoch_end : int
        The number of the last iteration of the last epoch. -1 if
        no epoch has ended so far.

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.iterations_done = 0
        self.epochs_done = 0
        self.last_epoch_end = -1


class TrainingLogRow(object):
    """A convenience interface for a row of the training log.

    Parameters
    ----------
    log : instance of :class:`AbstractTrainingLog`.
        The log to which the row belongs.
    time : int
        A time step of the row.

    """
    def __init__(self, log, time):
        self.log = log
        self.time = time

    def __getattr__(self, key):
        return self.log.fetch_record(self.time, key)

    def __setattr__(self, key, value):
        if key in ['log', 'time']:
            return super(TrainingLogRow, self).__setattr__(key, value)
        self.log.add_record(self.time, key, value)


class AbstractTrainingLog(object):
    """Base class for training logs.

    A training log stores the training timeline, statistics and
    other auxiliary information. Information is represented as a set of
    time-key-value triples. A default value can be set for a key that
    will be used when no other value is provided explicitly. The default
    default value is ''None''.

    In addition to the set of records displaying training dynamics, a
    training log has a status object whose attributes form the state of
    the training procedure.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_status(self):
        """Returns the training status.

        Returns
        -------
            An instance of :class:`AbstractTrainingStatus`, whose
            attributes contain the information regarding the status of
            the training process.

        """
        pass

    @property
    def status(self):
        """A convenient way to access the status."""
        return self.get_status()

    @abstractmethod
    def set_default_value(self, key, value):
        """Sets a default value for 'key'."""
        pass

    @abstractmethod
    def get_default_value(self, key):
        """Returns the default value set for the 'key'.

        Returns
        -------
        The default value for the `key`, or ``None`` if not set.

        """
        pass

    def add_record(self, time, key, value):
        """Adds a record to the log.

        If `value` equals to the default value for the `key`, nothing is
        done.

        """
        self._check_time(time)
        default_value = self.get_default_value(key)
        if value != default_value:
            self._add_record(time, key, value)

    @abstractmethod
    def _add_record(self, time, key, value):
        """Adds a record to the log.

        The implementation method to be overriden.

        """
        pass

    def fetch_record(self, time, key):
        """Fetches a record from the log.

        If no such 'key' for the `time` is found or if the value for the
        key is ``None``, the default value for the 'key' is returned.

        """
        self._check_time(time)
        value = self._fetch_record(time, key)
        if value is not None:
            return value
        return self.get_default_value(key)

    @abstractmethod
    def _fetch_record(self, time, key):
        """Fetches a record from the log.

        The implementation method to be overriden.

        """
        pass

    def __getitem__(self, time):
        self._check_time(time)
        return TrainingLogRow(self, time)

    @property
    def current_row(self):
        return self[self.status.iterations_done]

    @property
    def previous_row(self):
        return self[self.status.iterations_done - 1]

    @property
    def last_epoch_row(self):
        return self[self.status.last_epoch_end]

    @abstractmethod
    def __iter__(self):
        """Returns an iterator over time-key-value triples of the log."""
        pass

    def _check_time(self, time):
        if not isinstance(time, int) or time < 0:
            raise ValueError("time must be a positive integer")


class RAMStatus(AbstractTrainingStatus):
    """A simple training status."""
    pass


class RAMTrainingLog(AbstractTrainingLog):
    """A simple training log storing information in main memory."""
    def __init__(self):
        self._storage = defaultdict(dict)
        self._default_values = {}
        self._status = RAMStatus()

    def get_default_value(self, key):
        return self._default_values.get(key)

    def set_default_value(self, key, value):
        self._default_values[key] = value

    def _add_record(self, time, key, value):
        self._storage[time][key] = value

    def _fetch_record(self, time, key):
        slice_ = self._storage.get(time)
        if not slice_:
            return None
        return slice_.get(key)

    def __iter__(self):
        for time, records in self._storage.items():
            for key, value in records.items():
                yield time, key, value

    def get_status(self):
        return self._status
