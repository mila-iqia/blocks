import collections
import theano
from theano import tensor
import numpy
from numpy.testing import assert_raises, assert_allclose, assert_equal
from blocks.bricks import Tanh, Sequence
from blocks.bricks import (BatchNormalization, SpatialBatchNormalization,
                           BatchNormalizedMLP)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                MaxPooling, AveragePooling)
from blocks.initialization import Constant
from blocks.graph import ComputationGraph, batch_normalize


def random_unif(rng, dim, low=1, high=10):
    return (rng.uniform(low, high, size=dim)
            .astype(theano.config.floatX))


def test_batch_normalization_allocation_initialization():
    """Sanity check allocation & initialization of BN bricks."""
    def check(input_dim, expected_shape, broadcastable=None, save_memory=True):
        bn = BatchNormalization(input_dim=input_dim,
                                broadcastable=broadcastable,
                                save_memory=save_memory)
        if broadcastable is None:
            if not isinstance(input_dim, collections.Sequence):
                b_input_dim = (input_dim,)
            else:
                b_input_dim = input_dim
            input_broadcastable = tuple(False for _ in range(len(b_input_dim)))
        else:
            input_broadcastable = broadcastable
        bn.allocate()
        assert save_memory == bn.save_memory
        assert input_dim == bn.input_dim
        assert bn.broadcastable == broadcastable
        real_broadcastable = (True,) + input_broadcastable
        assert bn.W.broadcastable == real_broadcastable
        assert bn.b.broadcastable == real_broadcastable
        assert bn.population_mean.broadcastable == real_broadcastable
        assert bn.population_stdev.broadcastable == real_broadcastable
        assert_allclose(bn.population_mean.get_value(borrow=True), 0.)
        assert_allclose(bn.population_stdev.get_value(borrow=True), 1.)
        assert_equal(bn.W.get_value(borrow=True).shape, expected_shape)
        assert_equal(bn.b.get_value(borrow=True).shape, expected_shape)
        assert_equal(bn.population_mean.get_value(borrow=True).shape,
                     expected_shape)
        assert_equal(bn.population_stdev.get_value(borrow=True).shape,
                     expected_shape)
        assert numpy.isnan(bn.b.get_value(borrow=True)).all()
        assert numpy.isnan(bn.W.get_value(borrow=True)).all()
        bn.initialize()
        assert_allclose(bn.b.get_value(borrow=True), 0.)
        assert_allclose(bn.W.get_value(borrow=True), 1.)

    yield check, 5, (1, 5)
    yield check, (6, 7, 9), (1, 6, 7, 9), (False, False, False)
    yield check, (7, 4, 3), (1, 1, 4, 3), (True, False, False)
    yield check, (9, 3, 6), (1, 9, 1, 1), (False, True, True)
    yield check, (7, 4, 5), (1, 7, 1, 5), (False, True, False), False


def apply_setup(input_dim, broadcastable, save_memory):
    bn = BatchNormalization(input_dim, broadcastable, save_memory)
    bn.initialize()
    b_len = (len(input_dim) if isinstance(input_dim, collections.Sequence)
             else 1)
    x = tensor.TensorType(theano.config.floatX,
                          [False] * (b_len + 1))()
    y = bn.apply(x)
    return bn, x, y


def test_batch_normalization_inference_apply():
    def check(input_dim, variable_dim, broadcastable=None, save_memory=True):
        bn, x, y = apply_setup(input_dim, broadcastable, save_memory)
        rng = numpy.random.RandomState((2015, 12, 16))
        input_ = random_unif(rng,
                             (9,) +
                             (input_dim
                              if isinstance(input_dim, collections.Sequence)
                              else (input_dim,)))

        # Upon initialization, should be just the identity function.
        assert_allclose(y.eval({x: input_}), input_, rtol=1e-4)

        # Test population mean gets subtracted.
        pop_mean = random_unif(rng, variable_dim)
        bn.population_mean.set_value(pop_mean)
        assert_allclose(y.eval({x: input_}), input_ - pop_mean, rtol=1e-4)

        # Test population stdev is divided out.
        pop_stdev = random_unif(rng, variable_dim)
        bn.population_stdev.set_value(pop_stdev)
        assert_allclose(y.eval({x: input_}), (input_ - pop_mean) / pop_stdev,
                        rtol=1e-4)

        # Test learned scale is applied.
        gamma = random_unif(rng, variable_dim)
        bn.W.set_value(gamma)
        assert_allclose(y.eval({x: input_}),
                        (input_ - pop_mean) * (gamma / pop_stdev),
                        rtol=1e-4)

        # Test learned offset is applied.
        beta = random_unif(rng, variable_dim)
        bn.b.set_value(beta)
        assert_allclose(y.eval({x: input_}),
                        (input_ - pop_mean) * (gamma / pop_stdev) + beta,
                        rtol=1e-4)

    yield check, 9, (1, 9)
    yield check, (5, 4), (1, 5, 4), None, False
    yield check, (2, 9, 7), (1, 2, 1, 1), (False, True, True)


def test_batch_normalization_train_apply():
    def check(input_dim, variable_dim, broadcastable=None, save_memory=True):
        epsilon = numpy.cast[theano.config.floatX](1e-4)
        bn, x, y = apply_setup(input_dim, broadcastable, save_memory)
        cg = ComputationGraph([y])
        new_cg, _ = batch_normalize(cg, epsilon=epsilon)
        y_hat = new_cg.outputs[0]

        rng = numpy.random.RandomState((2015, 12, 16))
        input_ = random_unif(rng, (9,) +
                             (input_dim
                              if isinstance(input_dim, collections.Sequence)
                              else (input_dim,)))
        axes = tuple(i for i, b in
                     enumerate(bn.population_mean.broadcastable) if b)

        # NumPy implementation of the batch-normalization transform.
        def normalize(x):
            return ((x - x.mean(axis=axes, keepdims=True,
                                dtype=theano.config.floatX)) /
                    numpy.sqrt(numpy.var(x, axis=axes, keepdims=True,
                                         dtype=theano.config.floatX) +
                               epsilon))

        # Check that batch norm is doing what it should be.
        assert_allclose(y_hat.eval({x: input_}), normalize(input_),
                        atol=(1e-3 if theano.config.floatX == 'float32'
                              else 1e-7))

        # Check that the scale parameters are still getting applied.
        gamma = random_unif(rng, variable_dim)
        bn.W.set_value(gamma)
        assert_allclose(y_hat.eval({x: input_}), normalize(input_) * gamma,
                        atol=(1e-3 if theano.config.floatX == 'float32'
                              else 1e-7))

        beta = random_unif(rng, variable_dim)
        bn.b.set_value(beta)
        # Check that the shift parameters are still getting applied.
        assert_allclose(y_hat.eval({x: input_}),
                        normalize(input_) * gamma + beta,
                        atol=(1e-3 if theano.config.floatX == 'float32'
                              else 1e-7))

        # Double check that setting the population parameters doesn't
        # affect anything.
        bn.population_mean.set_value(numpy.nan *
                                     bn.population_mean.get_value())
        bn.population_stdev.set_value(numpy.nan *
                                      bn.population_mean.get_value())
        assert_allclose(y_hat.eval({x: input_}),
                        normalize(input_) * gamma + beta,
                        atol=(1e-3 if theano.config.floatX == 'float32'
                              else 1e-7))

    yield check, 9, (1, 9)
    yield check, (5, 4), (1, 5, 4), None, False
    yield check, (2, 9, 7), (1, 2, 1, 1), (False, True, True)


def test_batch_normalization_image_size_setter():
    bn = BatchNormalization()
    bn.image_size = (5, 4)
    assert bn.input_dim == (None, 5, 4)
    bn.image_size = (4, 5)
    assert bn.input_dim == (None, 4, 5)


def test_spatial_batch_normalization():
    def check(*input_dim):
        sbn = SpatialBatchNormalization(input_dim)
        sbn.initialize()
        x = theano.tensor.TensorType(theano.config.floatX,
                                     [False] * (len(input_dim) + 1))()
        y = sbn.apply(x)
        rng = numpy.random.RandomState((2015, 12, 17))
        input_ = random_unif(rng, (11,) + input_dim)
        assert_equal(y.eval({x: input_}), input_)

    # Work around a stupid bug in nose2 by passing as *args.
    yield check, 2, 3, 5
    yield check, 5, 3, 2, 3
    yield check, 1, 11


def test_raise_exception_spatial():
    # Work around a stupid bug in nose2 that unpacks the tuple into
    # separate arguments.
    yield assert_raises, (ValueError, SpatialBatchNormalization, (5,))
    yield assert_raises, (ValueError, SpatialBatchNormalization, 3)

    def do_not_fail(*input_dim):
        try:
            SpatialBatchNormalization(input_dim)
        except ValueError:
            assert False

    # Work around a stupid bug in nose2 by passing as *args.
    yield do_not_fail, 5, 4, 3
    yield do_not_fail, 7, 6
    yield do_not_fail, 3, 9, 2, 3


def test_batch_normalization_inside_convolutional_sequence():
    conv_seq = ConvolutionalSequence(
        [Convolutional(filter_size=(3, 3), num_filters=4),
         BatchNormalization(broadcastable=(False, True, True)),
         AveragePooling(pooling_size=(2, 2)),
         BatchNormalization(broadcastable=(False, False, False)),
         MaxPooling(pooling_size=(2, 2), step=(1, 1))],
        weights_init=Constant(1.),
        biases_init=Constant(2.),
        image_size=(10, 8), num_channels=9)

    conv_seq_no_bn = ConvolutionalSequence(
        [Convolutional(filter_size=(3, 3), num_filters=4),
         AveragePooling(pooling_size=(2, 2)),
         MaxPooling(pooling_size=(2, 2), step=(1, 1))],
        weights_init=Constant(1.),
        biases_init=Constant(2.),
        image_size=(10, 8), num_channels=9)

    conv_seq.initialize()
    conv_seq_no_bn.initialize()
    rng = numpy.random.RandomState((2015, 12, 17))
    input_ = random_unif(rng, (2, 9, 10, 8))

    x = theano.tensor.tensor4()
    ybn = conv_seq.apply(x)
    y = conv_seq_no_bn.apply(x)
    yield (assert_equal, ybn.eval({x: input_}), y.eval({x: input_}))

    std = conv_seq.children[-2].population_stdev
    std.set_value(3 * std.get_value(borrow=True))
    yield (assert_equal, ybn.eval({x: input_}), y.eval({x: input_}) / 3.)


def test_batch_normalized_mlp_construction():
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9])
    assert all(isinstance(a, Sequence) for a in mlp.activations)
    assert all(isinstance(a.children[0], BatchNormalization)
               for a in mlp.activations)
    assert all(isinstance(a.children[1], Tanh)
               for a in mlp.activations)


def test_batch_normalized_mlp_allocation():
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9])
    mlp.allocate()
    assert mlp.activations[0].children[0].input_dim == 7
    assert mlp.activations[1].children[0].input_dim == 9
    assert not any(l.use_bias for l in mlp.linear_transformations)


def test_batch_normalized_mlp_initialization():
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9])
    mlp.allocate()
    assert mlp.activations[0].children[0].input_dim == 7
    assert mlp.activations[1].children[0].input_dim == 9
    assert not any(l.use_bias for l in mlp.linear_transformations)
