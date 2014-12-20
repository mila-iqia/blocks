import numpy

import theano
from theano import tensor

from blocks.bricks.attention import SequenceContentAttention

floatX = theano.config.floatX


def test_sequence_content_attention():
    # Disclaimer: only check dimensions, not values
    rng = numpy.random.RandomState([2014, 12, 2])

    seq_len = 5
    batch_size = 6
    state_dim = 2
    sequence_dim = 3
    match_dim = 4

    attention = SequenceContentAttention(
        state_names=["states"], state_dims={"states": state_dim},
        sequence_dim=sequence_dim, match_dim=match_dim)
    attention.allocate()

    sequences = tensor.tensor3('sequences')
    states = tensor.matrix('states')
    mask = tensor.matrix('mask')
    glimpses, weights = attention.take_look(sequences, states=states,
                                            mask=mask)
    assert glimpses.ndim == 2
    assert weights.ndim == 2

    seq_values = numpy.zeros((seq_len, batch_size, sequence_dim), dtype=floatX)
    states_values = numpy.zeros((batch_size, state_dim), dtype=floatX)
    mask_values = numpy.zeros((seq_len, batch_size), dtype=floatX)
    # randomly generate a sensible mask
    for sed_idx in range(batch_size):
        mask_values[:rng.randint(1, seq_len), sed_idx] = 1
    glimpses_values, weight_values = theano.function(
        [sequences, states, mask], [glimpses, weights])(
            seq_values, states_values, mask_values)
    assert glimpses_values.shape == (batch_size, sequence_dim)
    assert weight_values.shape == (batch_size, seq_len)
    assert numpy.all(weight_values >= 0)
    assert numpy.all(weight_values <= 1)
    assert numpy.all(weight_values.sum(axis=1) == 1)
    assert numpy.all((weight_values.T == 0) == (mask_values == 0))
