from abc import ABCMeta, abstractmethod

import numpy
import six

from blocks.utils import update_instance


class Dataset(object):
    """A dataset.

    Dataset classes implement the interface to a particular dataset.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset can provide. By default, these are
        ``features`` and ``targets``.

    Notes
    -----
    Datasets should only implement the interface; they are not expected to
    perform the iteration over the actual data. As such, they are
    stateless, and can be shared by different parts of the library
    simultaneously.

    """
    __metaclass__ = ABCMeta
    sources = ('features', 'targets')

    def __iter__(self):
        """Use the default iteration scheme to construct a data stream.

        .. warning::

           A dataset only produces a single data stream using the default
           iteration scheme, so multiple iterations in parallel are not
           supported.

        """
        if not hasattr(self, 'default_scheme'):
            raise NotImplementedError("Does not provide a default iterator")
        elif not hasattr(self, 'default_stream'):
            self.default_stream = DataStream(self, self.default_scheme)
        return iter(self.default_stream)

    def open(self, state=None):
        """Return the state if the dataset requires one.

        Datasets which e.g. read files from disks require open file
        handlers, and this sort of stateful information should be handled
        by the data stream.

        Parameters
        ----------
        state : object
            The state of the dataset after the previous epoch.

        Returns
        -------
        state : object
            An object representing the state of a dataset.


        Notes
        -----
        The first time the dataset is opened this function always receives
        `None`. Subsequent calls might pass the state at the end of the
        previous operation. This allows for quicker opening e.g. by seeking
        to the beginning of the file instead of re-opening it.

        """
        pass

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


class DataStream(object):
    """A stream of data.

    A data stream implements the ``__iter__`` protocol, returning a new
    iterator over the data each time it gets called (in the case of finite
    data). A single pass over one of these iterators represents an *epoch*.

    The data stream also opens a given dataset and maintains the state
    (e.g. file handles) if necessary.

    A data stream can provide an iterator over a dataset, but it can also
    wrap another datastream e.g. to provide caching or some sort of
    pre-processing.

    Parameters
    ----------
    data : :class:`Dataset` or :class:`DataStream`
        An object that implements the :meth:`open` and :meth:`get_data`
        methods. Usually this is a :class:`Dataset` instance, but when
        chaining iterators to e.g. perform caching this can also be a
        :class:`DataStream` object.
    iteration_scheme : :class:`IterationScheme`, optional
        The iteration scheme to use when retrieving data. Note that not all
        datasets support the same iteration schemes, some data sets require
        one, and others don't support any.
    sources : tuple of strings, optional
        The sources of data to return. By default, all sources of the
        dataset are requested.

    """
    def __init__(self, data, iteration_scheme=None, sources=None):
        update_instance(self, locals())
        self.data_state = None

    def __iter__(self):
        self.data_state = self.data.open(self.data_state)
        # TODO Allow for multiple child datasets/data streams, so that they
        # can be combined
        if isinstance(self.data, DataStream):
            self.child_iterator = iter(self.data)
        return DataIterator(self, iter(self.iteration_scheme)
                            if self.iteration_scheme else None, self.sources)

    @property
    def sources(self):
        if getattr(self, '_sources', None) is None:
            return self.data.sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources

    def open(self, state=None):
        return self.data.open(state)

    def close(self, satte):
        self.data.close(self.data_state)

    def get_data(self, state=None, request=None, sources=None):
        """Get data from the dataset.

        Notes
        -----
        This is the default implementation which redirects the request for
        data directly to the dataset (or wrapped data stream). For more
        complex data streams, one could perform e.g. caching here.

        """
        return self.data.get_data(state, request, sources)


class CachedDataStream(DataStream):
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
    def __init__(self, iteration_scheme, *args, **kwargs):
        super(CachedDataStream, self).__init__(
            iteration_scheme=iteration_scheme, *args, **kwargs)
        self.cache = [[] for source in self.sources]

    def get_data(self, state=None, request=None, sources=None):
        if isinstance(request, six.integer_types):
            batch_size = request
        elif isinstance(request, slice):
            batch_size = request.stop - request.start
        if batch_size >= len(self.cache[0]):
            self._cache(state, sources)
        data = []
        for i, cache in enumerate(self.cache):
            data.append(numpy.asarray(cache[:batch_size]))
            self.cache[i] = cache[batch_size:]
        return tuple(data)

    def _cache(self, state, sources):
        for cache, data in zip(self.cache, next(self.child_iterator)):
            cache.extend(data)


class DataIterator(six.Iterator):
    """An iterator over data, representing a single epoch."""
    def __init__(self, data_stream, iterator=None, sources=None):
        update_instance(self, locals())

    def __iter__(self):
        return self

    def __next__(self):
        return self.data_stream.get_data(self.data_stream.data_state,
                                         next(self.iterator)
                                         if self.iterator else None,
                                         self.sources)
