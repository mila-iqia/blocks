from blocks.datasets import ContainerDataset, MappingDataStream
from collections import OrderedDict


def test_dataset():
    data = [1, 2, 3]
    data2 = [2, 4, 6]
    stream = ContainerDataset([1, 2, 3]).open_stream()
    assert list(stream) == list(zip(data))

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
