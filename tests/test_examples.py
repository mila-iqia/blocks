import logging

import dill

import blocks
from blocks.extensions.saveload import SAVED_TO
from examples.sqrt import main as sqrt_test
from examples.mnist import main as mnist_test
from examples.markov_chain.main import main as markov_chain_test
from tests import temporary_files, silence_printing


def setup():
    # Silence Block's logger
    logger = logging.getLogger(blocks.__name__)
    logger.setLevel(logging.ERROR)


@temporary_files('__sqrt')
@silence_printing
def test_sqrt():
    filename = '__sqrt'
    sqrt_test(filename, 7)
    main_loop = sqrt_test(filename, 14, continue_=True)
    assert main_loop.log[7][SAVED_TO] == filename


@temporary_files('mnist.pkl')
@silence_printing
def test_mnist():
    filename = 'mnist.pkl'
    mnist_test(filename, 1)
    with open(filename, "rb") as source:
        main_loop = dill.load(source)
    main_loop.find_extension("FinishAfter").invoke_after_n_epochs(2)
    main_loop.run()
    assert main_loop.log.status.epochs_done == 2

test_mnist.setup = setup


@temporary_files('chain.pkl')
@silence_printing
def test_markov_chain():
    filename = 'chain.pkl'
    markov_chain_test("train", filename, None, 10)

test_mnist.setup = setup
