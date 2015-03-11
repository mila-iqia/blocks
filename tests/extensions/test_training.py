import numpy
from numpy.testing import assert_allclose

import theano
from fuel.datasets import IterableDataset
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter
from blocks.extensions.training import SharedVariableModifier, TrackTheBest
from blocks.log import TrainingLog
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx

floatX = theano.config.floatX


def test_shared_variable_modifier():
    weights = numpy.array([-1, 1], dtype=floatX)
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    targets = [(weights * f).sum() for f in features]
    n_batches = 3
    dataset = IterableDataset(dict(features=features, targets=targets))

    x = tensor.vector('features')
    y = tensor.scalar('targets')
    W = shared_floatx([0, 0], name='W')
    cost = ((x * W).sum() - y) ** 2
    cost.name = 'cost'

    step_rule = Scale(0.001)
    sgd = GradientDescent(cost=cost, params=[W],
                          step_rule=step_rule)
    main_loop = MainLoop(
        model=None, data_stream=dataset.get_example_stream(),
        algorithm=sgd,
        extensions=[
            FinishAfter(after_n_epochs=1),
            SharedVariableModifier(step_rule.learning_rate,
                                   lambda n: numpy.cast[floatX](10. / n))
            ])

    main_loop.run()

    assert_allclose(step_rule.learning_rate.get_value(),
                    numpy.cast[floatX](10. / n_batches))


def test_shared_variable_modifier_two_params():
    weights = numpy.array([-1, 1], dtype=floatX)
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    targets = [(weights * f).sum() for f in features]
    n_batches = 3
    dataset = IterableDataset(dict(features=features, targets=targets))

    x = tensor.vector('features')
    y = tensor.scalar('targets')
    W = shared_floatx([0, 0], name='W')
    cost = ((x * W).sum() - y) ** 2
    cost.name = 'cost'

    step_rule = Scale(0.001)
    sgd = GradientDescent(cost=cost, params=[W],
                          step_rule=step_rule)
    modifier = SharedVariableModifier(
        step_rule.learning_rate,
        lambda _, val: numpy.cast[floatX](val * 0.2))
    main_loop = MainLoop(
        model=None, data_stream=dataset.get_example_stream(),
        algorithm=sgd,
        extensions=[FinishAfter(after_n_epochs=1), modifier])

    main_loop.run()

    new_value = step_rule.learning_rate.get_value()
    assert_allclose(new_value,
                    0.001 * 0.2 ** n_batches,
                    atol=1e-5)


def test_track_the_best():
    class FakeMainLoop(object):

        def __init__(self):
            self.log = TrainingLog()

        @property
        def status(self):
            return self.log.status

    main_loop = FakeMainLoop()
    extension = TrackTheBest("cost")
    extension.main_loop = main_loop

    main_loop.status.iterations_done += 1
    main_loop.log.current_row.cost = 5
    extension.dispatch('after_batch')
    assert main_loop.status.best_cost == 5
    assert main_loop.log.current_row['cost_is_best_so_far'] == True

    main_loop.status.iterations_done += 1
    main_loop.log.current_row.cost = 6
    extension.dispatch('after_batch')
    assert main_loop.status.best_cost == 5
    assert main_loop.log.current_row['cost_is_best_so_far'] is None

    main_loop.status.iterations_done += 1
    main_loop.log.current_row.cost = 5
    extension.dispatch('after_batch')
    assert main_loop.status.best_cost == 5
    assert main_loop.log.current_row['cost_is_best_so_far'] is None

    main_loop.status.iterations_done += 1
    main_loop.log.current_row.cost = 4
    extension.dispatch('after_batch')
    assert main_loop.status.best_cost == 4
    assert main_loop.log.current_row['cost_is_best_so_far'] is True
