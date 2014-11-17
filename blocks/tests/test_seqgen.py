import numpy

import theano
from theano import tensor

from blocks.bricks import Tanh, application
from blocks.recurrent import GatedRecurrent
from blocks.sequence_generators import (
    SequenceGenerator, LinearReadout, TrivialEmitter,
    SoftmaxEmitter, LookupFeedback)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant

floatX = theano.config.floatX


def test_sequence_generator():
    # Disclaimer: here we only check shapes, not values.

    output_dim = 1
    dim = 20
    batch_size = 30
    n_steps = 10

    class Emitter(TrivialEmitter):
        @application
        def cost(self, readouts, outputs):
            """Compute MSE."""
            return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)

    transition = GatedRecurrent(
        name="transition", activation=Tanh(), dim=dim,
        weights_init=Orthogonal())
    generator = SequenceGenerator(
        LinearReadout(readout_dim=output_dim, source_names=["states"],
                      emitter=Emitter(name="emitter"), name="readout"),
        transition,
        weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
        name="generator")
    generator.initialize()

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

def test_integer_sequence_generator():
    # Disclaimer: here we only check shapes, not values.

    readout_dim = 5
    feedback_dim = 3
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = GatedRecurrent(
        name="transition", activation=Tanh(), dim=dim,
        weights_init=Orthogonal())
    generator = SequenceGenerator(
        LinearReadout(readout_dim=readout_dim, source_names=["states"],
                      emitter=SoftmaxEmitter(name="emitter"),
                      feedbacker=LookupFeedback(readout_dim, feedback_dim),
                      name="readout"),
        transition,
        weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
        name="generator")
    generator.initialize()

    y = tensor.lmatrix('y')
    mask = tensor.matrix('mask')
    costs = generator.cost(y, mask)
    assert costs.ndim == 2
    costs_val = theano.function([y, mask], [costs])(
        numpy.zeros((n_steps, batch_size), dtype='int64'),
        numpy.ones((n_steps, batch_size), dtype=floatX))[0]
    assert costs_val.shape == (n_steps, batch_size)

    states, outputs, costs = generator.generate(
        iterate=True, batch_size=batch_size, n_steps=n_steps)
    states_val, outputs_val, costs_val = theano.function(
        [], [states, outputs, costs],
        updates=costs.owner.inputs[0].owner.tag.updates)()
    assert states_val.shape == (n_steps, batch_size, dim)
    assert outputs_val.shape == (n_steps, batch_size)
    assert outputs_val.dtype == 'int64'
    assert costs_val.shape == (n_steps, batch_size)
