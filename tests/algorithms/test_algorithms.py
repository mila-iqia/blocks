import numpy
from numpy.testing import assert_allclose

from theano import tensor

from blocks.algorithms import GradientDescent
from blocks.utils import shared_floatx


def test_gradient_descent():
    W = shared_floatx(numpy.array([[1, 2], [3, 4]]))
    W_start_value = W.get_value()
    cost = tensor.sum(W ** 2)

    algorithm = GradientDescent(cost=cost, params=[W])
    algorithm.step_rule.learning_rate = 0.75
    algorithm.initialize()
    algorithm.process_batch(dict())
    assert_allclose(W.get_value(), -0.5 * W_start_value)
