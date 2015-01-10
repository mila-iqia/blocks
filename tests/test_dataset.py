from blocks.datasets import ContainerDataset, MappingDataStream


def test_dataset():
    data = [1, 2, 3]
    data2 = [2, 4, 6]
    stream = ContainerDataset([1, 2, 3]).open_stream()
    assert list(stream) == data

    for i, epoch in zip(range(3), stream.epochs()):
        assert list(epoch) == data

    wrapper = MappingDataStream(stream, lambda x: x * 2)
    wrapper.reset()
    assert list(wrapper) == data2
