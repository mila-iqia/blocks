from blocks.bricks import BatchNormalization, Sequence, Tanh, MLP
from blocks.graph import (ComputationGraph, batch_normalization,
                          apply_batch_normalization)
from blocks.initialization import Constant

import numpy
from numpy.testing import assert_allclose
import theano
from theano import tensor


def test_batch_normalization_simple():
    x = tensor.matrix()
    eps = 1e-4
    bn = BatchNormalization(input_dim=4, epsilon=eps)
    bn.initialize()
    with batch_normalization(bn):
        y = bn.apply(x)
    rng = numpy.random.RandomState((2016, 1, 18))
    x_ = rng.uniform(size=(5, 4)).astype(theano.config.floatX)
    y_ = y.eval({x: x_})
    y_expected = (x_ - x_.mean(axis=0)) / numpy.sqrt(x_.var(axis=0) + eps)
    assert_allclose(y_, y_expected, rtol=1e-4)


def test_batch_normalization_nested():
    x = tensor.tensor4()
    eps = 1e-4
    r_dims = (0, 2, 3)
    batch_dims = (5, 4, 3, 2)
    bn = BatchNormalization(input_dim=batch_dims[1:],
                            broadcastable=(False, True, True),
                            epsilon=eps)
    seq = Sequence([bn.apply, Tanh().apply])
    seq.initialize()
    with batch_normalization(seq):
        y = seq.apply(x)
    rng = numpy.random.RandomState((2016, 1, 18))
    x_ = rng.uniform(size=batch_dims).astype(theano.config.floatX)
    y_ = y.eval({x: x_})
    y_expected = numpy.tanh((x_ - x_.mean(axis=r_dims, keepdims=True)) /
                            numpy.sqrt(x_.var(axis=r_dims, keepdims=True) +
                                       eps))
    assert_allclose(y_, y_expected, rtol=1e-4)


def test_apply_batch_normalization_nested():
    x = tensor.matrix()
    eps = 1e-8
    batch_dims = (3, 9)
    bn = BatchNormalization(input_dim=5, epsilon=eps)
    mlp = MLP([Sequence([bn.apply, Tanh().apply])], [9, 5],
              weights_init=Constant(0.4), biases_init=Constant(1))
    mlp.initialize()
    y = mlp.apply(x)
    cg, _ = apply_batch_normalization(ComputationGraph([y]))
    y_bn = cg.outputs[0]
    rng = numpy.random.RandomState((2016, 1, 18))
    x_ = rng.uniform(size=batch_dims).astype(theano.config.floatX)
    y_ = y_bn.eval({x: x_})
    W_, b_ = map(lambda s: (getattr(mlp.linear_transformations[0], s)
                            .get_value(borrow=True)), ['W', 'b'])
    z_ = numpy.dot(x_, W_) + b_
    y_expected = numpy.tanh((z_ - z_.mean(axis=0)) /
                            numpy.sqrt(z_.var(axis=0) + eps))
    assert_allclose(y_, y_expected, rtol=1e-3)
