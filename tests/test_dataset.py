from blocks.datasets import (ContainerDataset, InitialDataStream,
                             MappingDataStream)
from blocks.datasets.schemes import BatchSizeScheme, ConstantScheme
from collections import OrderedDict


def test_dataset():
    data = [1, 2, 3]
    data2 = [2, 4, 6]
    stream = ContainerDataset([1, 2, 3]).open_stream()
    assert list(stream) == list(zip(data))

    stream.reset()
    for i, epoch in zip(range(3), stream.epochs()):
        assert list(epoch) == list(zip(data))

    stream.reset()
    assert stream.next_as_dict() == {"data": 1}

    wrapper = MappingDataStream(
        stream, lambda d: (2 * d[0],))
    wrapper.reset()
    assert list(wrapper) == list(zip(data2))


def test_sources_selection():
    features = [5, 6, 7, 1]
    targets = [1, 0, 1, 1]
    stream = ContainerDataset(
        OrderedDict([("features", features),
                     ("targets", targets)])).open_stream()
    assert list(stream) == list(zip(features, targets))

    stream.sources = ("targets",)
    stream.reset()
    assert list(stream) == list(zip(targets))


def test_data_driven_epochs():

    class TestDataset(ContainerDataset):

        sources = "data"
        default_scheme = ConstantScheme(1)

        def __init__(self):
            self.data = [[1, 2, 3, 4],
                         [5, 6, 7, 8]]

        def open(self):
            epoch_iter = iter(self.data)
            data_iter = iter(next(epoch_iter))
            return (epoch_iter, data_iter)

        def next_epoch(self, state):
            data_iter = iter(next(state[0]))
            return (state[0], data_iter)

        def get_data(self, state, request, sources):
            data = []
            for i in range(request):
                data.append(next(state[1]))
            return (data,)

    epochs = []
    epochs.append([([1],), ([2],), ([3],), ([4],)])
    epochs.append([([5],), ([6],), ([7],), ([8],)])
    stream = TestDataset().open_stream()
    assert list(stream) == epochs[0]
    stream.next_epoch()
    assert list(stream) == epochs[1]
    stream.reset()
    for i, epoch in enumerate(stream.epochs()):
        assert list(epoch) == epochs[i]

    # test scheme reseting between epochs
    class TestScheme(BatchSizeScheme):

        def __iter__(self):
            return iter([1, 2, 1, 3])

    epochs = []
    epochs.append([([1],), ([2, 3],), ([4],)])
    epochs.append([([5],), ([6, 7],), ([8],)])
    stream = InitialDataStream(TestDataset(), TestScheme())
    for i, epoch in enumerate(stream.epochs()):
        assert list(epoch) == epochs[i]
