import logging
import os

import pylearn2

import blocks
from examples.mnist import main as mnist_test
from examples.pylearn2.markov_chain import main as pylearn2_test


def setup():
    # Silence Pylearn2's logger
    logger = logging.getLogger(pylearn2.__name__)
    logger.setLevel(logging.ERROR)

    # Silence Block's logger
    logger = logging.getLogger(blocks.__name__)
    logger.setLevel(logging.ERROR)


def test_mnist():
    mnist_test()

test_mnist.setup = setup


def test_pylearn2():
    filename = 'unittest_markov_chain'
    pylearn2_test('train', filename, 0, 3, False)
    os.remove(filename)

test_pylearn2.setup = setup
