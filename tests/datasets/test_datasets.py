from collections import OrderedDict

import numpy
from six.moves import zip

from blocks.datasets import (CachedDataStream, ContainerDataset, DataStream,
                             DataStreamMapping)
from blocks.datasets.mnist import MNIST
from blocks.datasets.schemes import (BatchSizeScheme, ConstantScheme,
                                     SequentialScheme)


def test_dataset():
    data = [1, 2, 3]
    data_doubled = [2, 4, 6]

    # The default stream requests an example at a time
    stream = ContainerDataset([1, 2, 3]).get_default_stream()
    epoch = stream.get_epoch_iterator()
    assert list(epoch) == list(zip(data))

    # Check if iterating over multiple epochs works
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == list(zip(data))

    # Check whether the returning as a dictionary of sources works
    assert next(stream.get_epoch_iterator(as_dict=True)) == {"data": 1}

    # Check whether basic stream wrappers work
    wrapper = DataStreamMapping(
        stream, lambda d: (2 * d[0],))
    assert list(wrapper.get_epoch_iterator()) == list(zip(data_doubled))


def test_sources_selection():
    features = [5, 6, 7, 1]
    targets = [1, 0, 1, 1]
    stream = ContainerDataset(OrderedDict(
        [('features', features), ('targets', targets)])).get_default_stream()
    assert list(stream.get_epoch_iterator()) == list(zip(features, targets))

    stream = ContainerDataset({'features': features, 'targets': targets},
                              sources=('targets',)).get_default_stream()
    assert list(stream.get_epoch_iterator()) == list(zip(targets))


def test_data_driven_epochs():
    class TestDataset(ContainerDataset):
        sources = ('data',)
        default_scheme = ConstantScheme(1)

        def __init__(self):
            self.data = [[1, 2, 3, 4],
                         [5, 6, 7, 8]]

        def open(self):
            epoch_iter = iter(self.data)
            data_iter = iter(next(epoch_iter))
            return (epoch_iter, data_iter)

        def next_epoch(self, state):
            try:
                data_iter = iter(next(state[0]))
                return (state[0], data_iter)
            except StopIteration:
                return self.open()

        def get_data(self, state, request):
            data = []
            for i in range(request):
                data.append(next(state[1]))
            return (data,)

    epochs = []
    epochs.append([([1],), ([2],), ([3],), ([4],)])
    epochs.append([([5],), ([6],), ([7],), ([8],)])
    stream = TestDataset().get_default_stream()
    assert list(stream.get_epoch_iterator()) == epochs[0]
    assert list(stream.get_epoch_iterator()) == epochs[1]
    assert list(stream.get_epoch_iterator()) == epochs[0]

    stream.reset()
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == epochs[i]

    # test scheme reseting between epochs
    class TestScheme(BatchSizeScheme):

        def get_request_iterator(self):
            return iter([1, 2, 1, 3])

    epochs = []
    epochs.append([([1],), ([2, 3],), ([4],)])
    epochs.append([([5],), ([6, 7],), ([8],)])
    stream = DataStream(TestDataset(), iteration_scheme=TestScheme())
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == epochs[i]


def test_cache():
    mnist = MNIST('test')
    stream = DataStream(
        mnist, iteration_scheme=SequentialScheme(mnist.num_examples, 11))
    cached_stream = CachedDataStream(stream, ConstantScheme(7))
    epoch = cached_stream.get_epoch_iterator()

    # Make sure that cache is filled as expected
    for (features, targets), cache_size in zip(epoch, [4, 8, 1, 5, 9, 2,
                                                       6, 10, 3, 7, 0, 4]):
        assert len(cached_stream.cache[0]) == cache_size

    # Make sure that the epoch finishes correctly
    for features, targets in cached_stream.get_epoch_iterator():
        pass
    assert len(features) == mnist.num_examples % 7
    assert not cached_stream.cache[0]

    # Ensure that the epoch transition is correct
    cached_stream = CachedDataStream(stream, ConstantScheme(7, times=3))
    for _, epoch in zip(range(2), cached_stream.iterate_epochs()):
        cache_sizes = [4, 8, 1]
        for i, (features, targets) in enumerate(epoch):
            assert len(cached_stream.cache[0]) == cache_sizes[i]
            assert len(features) == 7
            assert numpy.all(mnist.features[i * 7:(i + 1) * 7] ==
                             features)
        assert i == 2
