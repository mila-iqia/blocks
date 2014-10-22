import unittest
import itertools
import numpy
from numpy.testing import assert_almost_equal
import theano
from theano import tensor
from blocks.bricks import Recurrent, Tanh
from blocks.initialization import Constant


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class TestRecurrent(unittest.TestCase):

    def setUp(self):
        self.simple = Recurrent(dim=3, weights_init=Constant(2),
                                activation=Tanh())
        self.simple.initialize()

    def test_one_step(self):
        h0 = tensor.fmatrix('h0')
        x = tensor.fmatrix('x')
        mask = tensor.fvector('mask')
        h1 = self.simple.apply(x, h0, mask=mask, one_step=True)
        next_h = theano.function(inputs=[h0, x, mask], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]], dtype="float32")
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]], dtype="float32")
        mask_val = numpy.array([1, 0]).astype("float32")
        h1_val = numpy.tanh(h0_val.dot(2 * numpy.ones((3, 3))) + x_val)
        h1_val = mask_val[:, None] * h1_val + (1 - mask_val[:, None]) * h0_val
        assert_almost_equal(h1_val, next_h(h0_val, x_val, mask_val)[0])

    def test_many_steps(self):
        x = tensor.ftensor3('x')
        mask = tensor.fmatrix('mask')
        h = self.simple.apply(x, mask=mask)
        calc_h = theano.function(inputs=[x, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype="float32")
        x_val = numpy.ones((24, 4, 3), dtype="float32") * x_val[..., None]
        mask_val = numpy.ones((24, 4), dtype="float32")
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype="float32")
        for i in range(1, 25):
            h_val[i] = numpy.tanh(h_val[i - 1].dot(
                2 * numpy.ones((3, 3))) + x_val[i - 1])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        assert_almost_equal(h_val, calc_h(x_val, mask_val)[0])
