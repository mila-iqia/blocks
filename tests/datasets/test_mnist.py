import numpy
from numpy.testing import assert_raises

import theano

from blocks.datasets.mnist import MNIST


def test_mnist():
    mnist_train = MNIST('train', start=20000)
    assert len(mnist_train.features) == 40000
    assert len(mnist_train.targets) == 40000
    assert mnist_train.num_examples == 40000
    mnist_test = MNIST('test', sources=('targets',))
    assert len(mnist_test.features) == 10000
    assert len(mnist_test.targets) == 10000
    assert mnist_test.num_examples == 10000

    first_feature, first_target = mnist_train.get_data(request=[0])
    assert first_feature.shape == (1, 784)
    assert first_feature.dtype is numpy.dtype(theano.config.floatX)
    assert first_target.shape == (1, 1)
    assert first_target.dtype is numpy.dtype('uint8')

    first_target, = mnist_test.get_data(request=[0, 1])
    assert first_target.shape == (2, 1)

    binary_mnist = MNIST('test', binary=True, sources=('features',))
    first_feature, = binary_mnist.get_data(request=[0])
    assert first_feature.dtype.kind == 'b'
    assert_raises(ValueError, MNIST, 'valid')
