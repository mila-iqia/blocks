import numpy
import theano
from fuel.datasets import IterableDataset
from fuel.schemes import (ConstantScheme,
                          SequentialExampleScheme,
                          SequentialScheme)
from fuel.streams import DataStream
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter, ProgressBar, Printing
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx


def setup_mainloop(extension, iteration_scheme=None):
    """Set up a simple main loop for progress bar tests.

    Create a MainLoop, register the given extension, supply it with a
    DataStream and a minimal model/cost to optimize.

    """
    # Since progressbar2 3.6.0, the `maxval` kwarg has been replaced by
    # `max_value`, which has a default value of 100. If we're still using
    # `maxval` by accident, this test should fail complaining that
    # the progress bar has received a value out of range.
    features = [numpy.array(f, dtype=theano.config.floatX)
                for f in [[1, 2]] * 101]
    dataset = IterableDataset(dict(features=features))
    data_stream = DataStream(dataset, iteration_scheme=iteration_scheme)

    W = shared_floatx([0, 0], name='W')
    x = tensor.vector('features')
    cost = tensor.sum((x-W)**2)
    cost.name = "cost"

    algorithm = GradientDescent(cost=cost, parameters=[W],
                                step_rule=Scale(1e-3))

    main_loop = MainLoop(
        model=None,
        data_stream=data_stream,
        algorithm=algorithm,
        extensions=[
            FinishAfter(after_n_epochs=1),
            extension])

    return main_loop


def test_progressbar():
    main_loop = setup_mainloop(ProgressBar())

    # We are happy if it does not crash or raise any exceptions
    main_loop.run()


def test_progressbar_iter_per_epoch_indices():
    iter_per_epoch = 100
    progress_bar = ProgressBar()
    main_loop = setup_mainloop(
        None, iteration_scheme=SequentialExampleScheme(iter_per_epoch))
    progress_bar.main_loop = main_loop

    assert progress_bar.get_iter_per_epoch() == iter_per_epoch


def test_progressbar_iter_per_epoch_batch_indices():
    num_examples = 1000
    batch_size = 10
    progress_bar = ProgressBar()
    main_loop = setup_mainloop(
        None, iteration_scheme=SequentialScheme(num_examples, batch_size))
    progress_bar.main_loop = main_loop

    assert progress_bar.get_iter_per_epoch() == num_examples // batch_size


def test_progressbar_iter_per_epoch_batch_examples():
    num_examples = 1000
    batch_size = 10
    progress_bar = ProgressBar()
    main_loop = setup_mainloop(
        None, iteration_scheme=ConstantScheme(batch_size, num_examples))
    progress_bar.main_loop = main_loop

    assert progress_bar.get_iter_per_epoch() == num_examples // batch_size


def test_printing():
    main_loop = setup_mainloop(Printing())

    # We are happy if it does not crash or raise any exceptions
    main_loop.run()
