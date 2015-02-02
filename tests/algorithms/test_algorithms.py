import numpy
from numpy.testing import assert_allclose

from theano import tensor

from blocks.algorithms import GradientDescent, GradientClipping
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


def test_gradient_clipping():
    rule1 = GradientClipping(4)
    rule2 = GradientClipping(5)

    gradients = {0: shared_floatx(3.0), 1: shared_floatx(4.0)}
    clipped1 = rule1.compute_steps(gradients)
    assert_allclose(clipped1[0].eval(), 12 / 5.0)
    assert_allclose(clipped1[1].eval(), 16 / 5.0)
    clipped2 = rule2.compute_steps(gradients)
    assert_allclose(clipped2[0].eval(), 3.0)
    assert_allclose(clipped2[1].eval(), 4.0)
