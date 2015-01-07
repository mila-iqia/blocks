from abc import ABCMeta, abstractmethod

import six

from blocks.utils import update_instance


class Dataset(six.Iterator):
    """A dataset.

    Dataset classes implement the interface to a particular dataset.

    Notes
    -----
    Datasets should only implement the interface; they are not expected to
    perform the iteration over the actual data. As such, they are
    stateless, and can be shared by different parts of the library
    simultaneously.

    """
    __metaclass__ = ABCMeta
    sources = ('features', 'targets')
    supported_iteration_schemes = tuple()

    def __iter__(self):
        """Datasets can return a data stream here for default iteration."""
        if not hasattr(self, 'default_scheme'):
            raise NotImplementedError("Does not provide a default iterator")
        elif not hasattr(self, 'default_stream'):
            self.default_stream = DataStream(self, self.default_scheme)
        return iter(self.default_stream)

    def open(self):
        """Return the state of datasets which require one.

        Datasets which e.g. read files from disks require open file
        handlers, and this sort of stateful information should be handled
        by the data stream when iterating over data.

        Returns
        -------
        state : object
            An object representing the state of a dataset.

        """
        pass

    @abstractmethod
    def get_data(state=None, request=None):
        """Request data from the dataset.

        Parameters
        ----------
        state : object, optional
            The state as returned by the :meth:`open` method. The dataset
            can use this to e.g. interact with files when needed.
        request : object, optional
            If supported, the request for a particular part of the data
            e.g. the number of examples to return, or the indices of a
            particular minimbatch of examples.

        Returns
        -------
        tuple
            A tuple of data in the same order as the data sources defined
            in :attr:`sources`.

        """
        raise NotImplementedError


class DataStream(object):
    """A stream of data.

    A data stream implements the ``__iter__`` protocol, returning a new
    iterator over the data each time it gets called (in the case of finite
    data). A single pass over one of these iterators represents an *epoch*.

    The data stream also opens a given dataset and maintains the state
    (e.g. file handles) if necessary.

    """
    def __init__(self, dataset, iteration_scheme=None):
        dataset_state = dataset.open()
        update_instance(self, locals())

    def __iter__(self):
        return DataIterator(self.dataset, self.dataset_state,
                            iter(self.iteration_scheme)
                            if self.iteration_scheme else None)


class DataIterator(six.Iterator):
    """An iterator over data, representing a single epoch."""
    def __init__(self, dataset, dataset_state=None, iteration_scheme=None):
        update_instance(self, locals())

    def __iter__(self):
        return self

    def __next__(self):
        return self.dataset.get_data(self.dataset_state,
                                     next(self.iteration_scheme)
                                     if self.iteration_scheme else None)
