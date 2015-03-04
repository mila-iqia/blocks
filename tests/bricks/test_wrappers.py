import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor

from blocks.bricks import Linear
from blocks.bricks.wrappers import As2D
from blocks.initialization import Constant


def test_as2d_ndim_gt_2():
    X = tensor.tensor4('X')
    brick = Linear(input_dim=3, output_dim=4, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = As2D(brick.apply)
    wrapper.initialize()
    f = theano.function([X], wrapper.apply(X))
    assert_allclose(
        f(numpy.ones(shape=(2, 2, 2, 3), dtype=theano.config.floatX)),
        3 * numpy.ones(shape=(2, 2, 2, 4), dtype=theano.config.floatX))


def test_as2d_ndim_leq_2():
    X = tensor.matrix('X')
    brick = Linear(input_dim=3, output_dim=4, weights_init=Constant(1),
                   biases_init=Constant(0))
    wrapper = As2D(brick.apply)
    wrapper.initialize()
    f = theano.function([X], wrapper.apply(X))
    assert_allclose(
        f(numpy.ones(shape=(2, 3), dtype=theano.config.floatX)),
        3 * numpy.ones(shape=(2, 4), dtype=theano.config.floatX))
