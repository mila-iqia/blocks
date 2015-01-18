import os

import dill
import numpy

from blocks.datasets import DataStream
from blocks.datasets.mnist import MNIST
from blocks.datasets.schemes import SequentialScheme


def test_in_memory():
    # Load MNIST and get two batches
    mnist = MNIST('train')
    data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
        num_examples=mnist.num_examples, batch_size=256))
    epoch = data_stream.get_epoch_iterator()
    for i, (features, targets) in enumerate(epoch):
        if i == 1:
            break
    numpy.all(features == mnist.data['features'][256:512])

    # Pickle the epoch and make sure that the data wasn't dumped
    with open('epoch_test.pkl', 'wb') as f:
        dill.dump(epoch, f, protocol=dill.HIGHEST_PROTOCOL)
    assert os.path.getsize('epoch_test.pkl') < 1024 * 1024  # Less than 1MB

    # Reload the epoch and make sure that the state was maintained
    del epoch
    with open('epoch_test.pkl', 'rb') as f:
        epoch = dill.load(f)
    features, targets = next(epoch)
    assert numpy.all(features == mnist.data['features'][512:768])
