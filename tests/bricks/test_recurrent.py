import itertools
import unittest

import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent, Recurrent, Bidirectional
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal


floatX = theano.config.floatX


class TestRecurrent(unittest.TestCase):
    def setUp(self):
        self.simple = Recurrent(dim=3, weights_init=Constant(2),
                                activation=Tanh())
        self.simple.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        mask = tensor.vector('mask')
        h1 = self.simple.apply(x, h0, mask=mask, iterate=False)
        next_h = theano.function(inputs=[h0, x, mask], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=floatX)
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]],
                                  dtype=floatX)
        mask_val = numpy.array([1, 0]).astype(floatX)
        h1_val = numpy.tanh(h0_val.dot(2 * numpy.ones((3, 3))) + x_val)
        h1_val = mask_val[:, None] * h1_val + (1 - mask_val[:, None]) * h0_val
        assert_allclose(h1_val, next_h(h0_val, x_val, mask_val)[0])

    def test_many_steps(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        h = self.simple.apply(x, mask=mask, iterate=True)
        calc_h = theano.function(inputs=[x, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype=floatX)
        x_val = numpy.ones((24, 4, 3),
                           dtype=floatX) * x_val[..., None]
        mask_val = numpy.ones((24, 4), dtype=floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=floatX)
        for i in range(1, 25):
            h_val[i] = numpy.tanh(h_val[i - 1].dot(
                2 * numpy.ones((3, 3))) + x_val[i - 1])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        assert_allclose(h_val, calc_h(x_val, mask_val)[0], rtol=1e-04)


class TestGatedRecurrent(unittest.TestCase):
    def setUp(self):
        self.gated = GatedRecurrent(
            dim=3, weights_init=Constant(2),
            activation=Tanh(), gate_activation=Tanh())
        self.gated.initialize()
        self.reset_only = GatedRecurrent(
            dim=3, weights_init=IsotropicGaussian(),
            activation=Tanh(), gate_activation=Tanh(),
            use_update_gate=False, rng=numpy.random.RandomState(1))
        self.reset_only.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        z = tensor.matrix('z')
        r = tensor.matrix('r')
        h1 = self.gated.apply(x, z, r, h0, iterate=False)
        next_h = theano.function(inputs=[h0, x, z, r], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=floatX)
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]],
                                  dtype=floatX)
        zi_val = (h0_val + x_val) / 2
        ri_val = -x_val
        W_val = 2 * numpy.ones((3, 3), dtype=floatX)

        z_val = numpy.tanh(h0_val.dot(W_val) + zi_val)
        r_val = numpy.tanh(h0_val.dot(W_val) + ri_val)
        h1_val = (z_val * numpy.tanh((r_val * h0_val).dot(W_val) + x_val)
                  + (1 - z_val) * h0_val)
        assert_allclose(h1_val, next_h(h0_val, x_val, zi_val, ri_val)[0],
                        rtol=1e-6)

    def test_reset_only_many_steps(self):
        x = tensor.tensor3('x')
        ri = tensor.tensor3('ri')
        mask = tensor.matrix('mask')
        h = self.reset_only.apply(x, reset_inputs=ri, mask=mask)
        calc_h = theano.function(inputs=[x, ri, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype=floatX)
        x_val = numpy.ones((24, 4, 3), dtype=floatX) * x_val[..., None]
        ri_val = 0.3 - x_val
        mask_val = numpy.ones((24, 4), dtype=floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=floatX)
        W = self.reset_only.state_to_state.get_value()
        U = self.reset_only.state_to_reset.get_value()

        for i in range(1, 25):
            r_val = numpy.tanh(h_val[i - 1].dot(U) + ri_val[i - 1])
            h_val[i] = numpy.tanh((r_val * h_val[i - 1]).dot(W)
                                  + x_val[i - 1])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        # TODO Figure out why this tolerance needs to be so big
        assert_allclose(h_val, calc_h(x_val, ri_val,  mask_val)[0], 1e-03)


class TestBidirectional(unittest.TestCase):
    def setUp(self):
        self.bidir = Bidirectional(weights_init=Orthogonal(),
                                   prototype=Recurrent(
                                       dim=3, activation=Tanh()))
        self.simple = Recurrent(dim=3, weights_init=Orthogonal(),
                                activation=Tanh())
        self.bidir.initialize()
        self.simple.initialize()
        self.x_val = 0.1 * numpy.asarray(
            list(itertools.permutations(range(4))),
            dtype=floatX)
        self.x_val = (numpy.ones((24, 4, 3), dtype=floatX)
                      * self.x_val[..., None])
        self.mask_val = numpy.ones((24, 4), dtype=floatX)
        self.mask_val[12:24, 3] = 0

    def test(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [self.bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        assert_allclose(h_simple, h_bidir[..., :3], rtol=1e-04)
        assert_allclose(h_simple_rev, h_bidir[::-1, ...,  3:], rtol=1e-04)
