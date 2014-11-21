import numpy

import theano
from theano import tensor

from blocks.bricks import Tanh, application
from blocks.parallel import Mixer
from blocks.recurrent import Recurrent, GatedRecurrent
from blocks.attention import SequenceContentAttention
from blocks.sequence_generators import (
    SequenceGenerator, LinearReadout, TrivialEmitter,
    SoftmaxEmitter, LookupFeedback, AttentionTransition)
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


class TestTransition(Recurrent):
    def __init__(self, attended_dim, **kwargs):
        super(TestTransition, self).__init__(**kwargs)
        self.attended_dim = attended_dim

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, *args, **kwargs):
        for context in TestTransition.apply.contexts:
            kwargs.pop(context)
        return super(TestTransition, self).apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return super(TestTransition, self).apply

    def get_dim(self, name):
        if name == 'attended':
            return self.attended_dim
        if name == 'attended_mask':
            return 0
        return super(TestTransition, self).get_dim(name)


def test_attention_transition():
    inp_dim = 2
    inp_len = 10
    attended_dim = 3
    attended_len = 11
    batch_size = 4

    transition = TestTransition(dim=inp_dim, attended_dim=attended_dim,
                                name="transition")
    attention = SequenceContentAttention(transition.apply.states,
                                         match_dim=inp_dim, name="attention")
    mixer = Mixer([name for name in transition.apply.sequences
                   if name != 'mask'],
                  attention.take_look.outputs[0],
                  name="mixer")
    att_trans = AttentionTransition(transition, attention, mixer,
                                    name="att_trans")
    att_trans.weights_init = IsotropicGaussian(0.01)
    att_trans.biases_init = Constant(0)
    att_trans.initialize()

    attended = tensor.tensor3("attended")
    attended_mask = tensor.matrix("attended_mask")
    inputs = tensor.tensor3("inputs")
    inputs_mask = tensor.matrix("inputs_mask")
    states, glimpses, weights = att_trans.apply(
        inp=inputs, mask=inputs_mask,
        attended=attended, attended_mask=attended_mask)
    assert states.ndim == 3
    assert glimpses.ndim == 3
    assert weights.ndim == 3

    input_values = numpy.zeros((inp_len, batch_size, inp_dim),
                                  dtype=floatX)
    input_mask_values = numpy.ones((inp_len, batch_size),
                                    dtype=floatX)
    attended_values = numpy.zeros((attended_len, batch_size, attended_dim),
                                   dtype=floatX)
    attended_mask_values = numpy.ones((attended_len, batch_size),
                                       dtype=floatX)

    func = theano.function([inputs, inputs_mask, attended, attended_mask],
                           [states, glimpses, weights])
    states_values, glimpses_values, weight_values = func(
        input_values, input_mask_values,
        attended_values, attended_mask_values)

    assert states_values.shape == input_values.shape
    assert glimpses_values.shape == (inp_len, batch_size, attended_dim)
    assert weight_values.shape == (inp_len, batch_size, attended_len)


