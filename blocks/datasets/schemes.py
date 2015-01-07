import itertools
from abc import ABCMeta, abstractmethod


class IterationScheme(object):
    """An iteration scheme

    Iteration schemes provide a dataset-agnostic iteration scheme, such as
    sequential batches, shuffled batches, etc. for datasets that choose to
    support them.

    Notes
    -----
    Iteration schemes implement the ``__iter__`` method, returning
    generators or iterators.

    Stochastic iteration schemes should generally not be shared between
    different data schemes, because it would make experiments harder to
    reproduce.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class BatchSizeScheme(IterationScheme):
    """Iteration scheme that returns batch sizes.

    For infinite datasets it doesn't make sense to provide indices to
    examples, but the number of samples per batch can still be given.
    Hence BatchSizeScheme is the base class for iteration schemes
    that only provide the number of examples that should be in a batch.

    """
    pass


class BatchScheme(IterationScheme):
    """Iteration schemes that return slices or indices for batches.

    For datasets where the number of examples is known and easily
    accessible (as is the case for most datasets which are small enough
    to be kept in memory, like MNIST) we can provide slices or lists of
    labels to the dataset.

    """
    pass


class ConstantScheme(BatchSizeScheme):
    """Constant batch size iterator.

    This subset iterator simply returns the same constant batch size
    for a given number of times (or else infinitely).

    """
    def __init__(self, batch_size, times=None):
        self.batch_size = batch_size
        self.times = times

    def __iter__(self):
        if self.times is None:
            return itertools.repeat(self.batch_size)
        else:
            return itertools.repeat(self.batch_size, self.times)


class SequentialScheme(BatchScheme):
    """Sequential batches iterator.

    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    """
    def __init__(self, num_examples, batch_size):
        self.num_examples = num_examples
        self.batch_size = batch_size

    def __iter__(self):
        return (slice(x, min(self.num_examples, x + self.batch_size))
                for x in range(0, self.num_examples, self.batch_size))
