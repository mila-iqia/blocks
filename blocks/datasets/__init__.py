from abc import ABCMeta, abstractmethod

import numpy
import six

from blocks.utils import update_instance


class Dataset(object):
    """A dataset.

    Dataset classes implement the interface to a particular dataset. The
    interface consists of a number of routines to manipulate so called
    "state" objects, e.g. open, reset and close them.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset can provide.

    Notes
    -----
    Datasets should only implement the interface; they are not expected to
    perform the iteration over the actual data. As such, they are
    stateless, and can be shared by different parts of the library
    simultaneously.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def open(self):
        """Return the state if the dataset requires one.

        Datasets which e.g. read files from disks require open file
        handlers, and this sort of stateful information should be handled
        by the data stream.

        Returns
        -------
        state : object
            An object representing the state of a dataset.

        """
        pass

    def reset(self, state):
        """Resets the state.

        The default implementation closes the state and opens a new one.
        A more efficient implementation can override the default one
        in descendant classes.

        Returns
        -------
        state : object
            A reseted state.

        """
        self.close(state)
        return self.open()

    def next_epoch(self, state):
        """Switches the state to the next epoch.

        The default implementation simply resets the state.

        Returns
        -------
        state : object
            The state corresponding to the start of next epoch.

        """
        return self.reset(state)

    def close(self, state):
        """Cleanly close the dataset e.g. close file handles."""
        pass

    @abstractmethod
    def get_data(self, state=None, request=None, sources=None):
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
        sources : tuple of strings, optional
            The data sources to return. By default all data sources are
            returned.

        .. todo::

           A way for the dataset to communicate which kind of requests it
           accepts, and a way to communicate what kind of request is being
           sent when supporting multiple.

        Returns
        -------
        tuple
            A tuple of data.

        """
        raise NotImplementedError

    def open_stream(self):
        """Use the default iteration scheme to construct a data stream.

        """
        if not hasattr(self, 'default_scheme'):
            raise NotImplementedError("Does not provide a default iterator")
        return InitialDataStream(self, self.default_scheme)


class ContainerDataset(Dataset):
    """Equips a Python container with the dataset interface.

    Parameters
    ----------
    container : iterable
        The container to provide interface to. The container's
        `__iter__` method should return a new iterator over the
        container.

    .. todo::

        Multiple container, custom source names.

    """
    default_scheme = None
    sources = ("data")

    def __init__(self, container):
        self.container = container

    def open(self):
        return iter(self.container)

    def get_data(self, state, request, sources):
        assert request is None
        return next(state)


class DataStream(object):
    """A stream of data separated into epochs.

    A data stream is an iterable stream of minibatches. In fact it acts
    like a standard Python file, with an additional capability of switching
    to the next epoch. Its `reset` method replaces `file_.seek(0)` idiom
    returning a stream back to the same state it had right after creation.

    Parameters
    ----------
    data : object
        An instance of :class:`Dataset` or of :class:`DataStream`. A
        data stream is typically build from a dataset or by wrapping
        a data stream, and a reference to this object is necessary
        to get the default source names.
    iteration_scheme : :class:`IterationScheme`, optional
        The iteration scheme to use when retrieving data. Note that not all
        datasets support the same iteration schemes, some datasets require
        one, and others don't support any. In case when the data stream
        wraps another data stream, the choice of supported iteration
        schemes is typically even more limited.
    sources : tuple of strings, optional
        The sources of data to return. By default, all sources of the
        `data` object are returned.

    Attributes
    ----------
    iteration_scheme : object
        The iteration scheme used to retrieve data.
    request_iterator : iterable
        The iterator over data requested produced by the iteration scheme.


    """
    __metaclass__ = ABCMeta

    def __init__(self, data, iteration_scheme=None, sources=None):
        update_instance(self, locals())
        self.request_iterator = (iter(self.iteration_scheme)
                                 if self.iteration_scheme
                                 else None)

    @property
    def sources(self):
        if getattr(self, '_sources', None) is None:
            return self.data.sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_data(next(self.request_iterator)
                             if self.request_iterator else None,
                             self.sources)

    @abstractmethod
    def get_data(self, request, sources):
        """Request data from the dataset or the wrapped stream.

        Parameters
        ----------
        request : object
            A request fetched from the `request_iterator`.
        sources : object
            The data sources requested.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the data stream."""
        pass

    @abstractmethod
    def close(self):
        """Gracefully close the data stream, e.g. releasing file handles."""
        pass

    @abstractmethod
    def next_epoch(self):
        """Switch to the next epoch of data. """
        pass

    def epochs(self):
        """Allow iteration through all epochs."""
        while True:
            self.next_epoch()
            yield self


class InitialDataStream(DataStream):
    """A stream of data from a dataset."""
    def __init__(self, dataset, iteration_scheme=None, sources=None):
        super(InitialDataStream, self).__init__(dataset, iteration_scheme,
                                                sources)
        update_instance(self, locals())
        self.data_state = self.dataset.open()

    def close(self):
        self.data_state = self.dataset.close(self.data_state)

    def reset(self):
        self.data_state = self.dataset.reset(self.data_state)

    def next_epoch(self):
        self.data_state = self.dataset.next_epoch(self.data_state)

    def get_data(self, request, sources):
        """Get data from the dataset.

        """
        return self.data.get_data(self.data_state, request, sources)


class WrapperDataStream(DataStream):
    """A data stream that wraps another data stream."""
    def __init__(self, data_stream, iteration_scheme, sources):
        super(WrapperDataStream, self).__init__(data_stream, iteration_scheme,
                                                sources)
        update_instance(self, locals())

    def close(self):
        self.data_stream.close()

    def reset(self):
        self.data_stream.reset()

    def next_epoch(self):
        self.data_stream.next_epoch()

    def get_data(self, request, sources):
        """Get data from the wrapped data stream.

        Should be overriden by descendants for less trivial behaviour.

        .. todo::

            Do we actually need having sources specification for
            upper level data streams?

        """
        assert request is None
        return next(self.data_stream)


class MappingDataStream(WrapperDataStream):
    """Applies a mapping to the data of the wrapped data stream."""
    def __init__(self, data_stream, mapping):
        super(MappingDataStream, self).__init__(data_stream, None, None)
        self.mapping = mapping

    def get_data(self, request, sources):
        return self.mapping(next(self.data_stream))


class CachedDataStream(WrapperDataStream):
    """Cache examples when sequentially reading a dataset.

    Given a data stream which reads large chunks of data, this data
    stream caches these chunks and returns smaller batches from it until
    exhausted.

    Parameters
    ----------
    iteration_scheme : :class:`IterationScheme`
        Note that this iteration scheme must return either batch sizes
        (integers) or sequential batches (slices), which must necessarily
        be smaller than the child data stream i.e. the batches returned
        must be smaller than the cache size.

    """
    def __init__(self, data_stream, iteration_scheme, *args, **kwargs):
        super(CachedDataStream, self).__init__(data_stream, iteration_scheme,
                                               *args, **kwargs)
        self.cache = [[] for source in self.sources]

    def get_data(self, request=None, sources=None):
        if isinstance(request, six.integer_types):
            batch_size = request
        elif isinstance(request, slice):
            batch_size = request.stop - request.start
        if batch_size >= len(self.cache[0]):
            self._cache(sources)
        data = []
        for i, cache in enumerate(self.cache):
            data.append(numpy.asarray(cache[:batch_size]))
            self.cache[i] = cache[batch_size:]
        return tuple(data)

    def _cache(self, sources):
        for cache, data in zip(self.cache, next(self.data_stream)):
            cache.extend(data)
