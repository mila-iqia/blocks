from six.moves import zip

from blocks.datasets import ContainerDataset, DataStreamMapping


def test_dataset():
    data = [1, 2, 3]
    data_doubled = [2, 4, 6]

    # The default stream requests an example at a time
    stream = ContainerDataset([1, 2, 3]).get_default_stream()
    epoch = stream.get_epoch_iterator()
    assert list(epoch) == list(zip(data))

    # Check if iterating over multiple epochs works
    for i, epoch in zip(range(2), stream.epochs):
        assert list(epoch) == list(zip(data))
    for i, epoch in enumerate(stream.epochs):
        assert list(epoch) == list(zip(data))
        if i == 1:
            break

    # Check whether the returning as a dictionary of sources works
    assert next(stream.get_epoch_iterator(as_dict=True)) == {"data": 1}

    # Check whether basic stream wrappers work
    wrapper = DataStreamMapping(
        stream, lambda d: (2 * d[0],))
    assert list(wrapper.get_epoch_iterator()) == list(zip(data_doubled))


def test_sources_selection():
    features = [5, 6, 7, 1]
    targets = [1, 0, 1, 1]
    stream = ContainerDataset({'features': features,
                               'targets': targets}).get_default_stream()
    assert list(stream.get_epoch_iterator()) == list(zip(features, targets))

    stream = ContainerDataset({'features': features, 'targets': targets},
                              sources=('targets',)).get_default_stream()
    assert list(stream.get_epoch_iterator()) == list(zip(targets))
