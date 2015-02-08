import logging
import tempfile

import dill

import blocks
from blocks.extensions.saveload import SAVED_TO
from examples.sqrt import main as sqrt_test
from examples.mnist import main as mnist_test
from examples.markov_chain.main import main as markov_chain_test
from tests import silence_printing


def setup():
    # Silence Block's logger
    logger = logging.getLogger(blocks.__name__)
    logger.setLevel(logging.ERROR)


@silence_printing
def test_sqrt():
    save_path = tempfile.mkdtemp()
    sqrt_test(save_path, 7)
    main_loop = sqrt_test(save_path, 14, continue_=True)
    assert main_loop.log[7][SAVED_TO] == save_path


@silence_printing
def test_mnist():
    f = tempfile.NamedTemporaryFile(delete=False)
    filename = f.name
    mnist_test(filename, 1)
    with open(filename, "rb") as source:
        main_loop = dill.load(source)
    main_loop.find_extension("FinishAfter").set_conditions(after_n_epochs=2)
    main_loop.run()
    assert main_loop.log.status.epochs_done == 2

test_mnist.setup = setup


@silence_printing
def test_markov_chain():
    f = tempfile.NamedTemporaryFile(delete=False)
    filename = f.name
    markov_chain_test("train", filename, None, 10)

test_mnist.setup = setup
