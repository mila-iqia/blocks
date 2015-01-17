import logging
import os

from examples.mnist import main as mnist
from examples.pylearn2.markov_chain import main as pylearn2


def test_mnist():
    mnist()


def test_pylearn2():
    # Silence Pylearn2's logger
    logger = logging.getLogger(pylearn2.__name__)
    logger.setLevel(logging.ERROR)

    filename = 'unittest_markov_chain'
    pylearn2('train', filename, 0, 3, False)
    os.remove(filename)
