import numpy

import theano
from theano import tensor

from blocks.bricks import Brick, GatedRecurrent, Tanh
from blocks.sequence_generators import (
    SequenceGenerator, TrivialEmitter, TrivialFeedback)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant

floatX = theano.config.floatX


class MSEEmitter(TrivialEmitter):

    @Brick.apply_method
    def cost(self, readouts, outputs):
        return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)


def test_sequence_generator():
    output_dim = 1
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = GatedRecurrent(
        name="transition", activation=Tanh(), dim=dim,
        weights_init=Orthogonal())

    generator = SequenceGenerator(
        transition, MSEEmitter(output_dim, name="emitter"),
        TrivialFeedback(output_dim, name="feedback"),
        null_output=tensor.zeros((output_dim,)),
        weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
    generator.readout.readout_dim = output_dim

    y = tensor.tensor3('y')
    mask = tensor.matrix('mask')
    costs = generator.cost(y, mask)
    assert costs.ndim == 2
    costs_val = theano.function([y, mask], [costs])(
        numpy.zeros((n_steps, batch_size, output_dim), dtype=floatX),
        numpy.ones((n_steps, batch_size), dtype=floatX))[0]
    assert costs_val.shape == (n_steps, batch_size)

    states, outputs, costs = [variable.eval() for variable in
                              generator.generate(
                                  iterate=True, batch_size=batch_size,
                                  n_steps=n_steps)]
    assert states.shape == (n_steps, batch_size, dim)
    assert outputs.shape == (n_steps, batch_size, output_dim)
    assert costs.shape == (n_steps, batch_size)
