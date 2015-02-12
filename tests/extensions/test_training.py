import numpy
from numpy.testing import assert_allclose

import theano

from blocks.extensions.training import SharedVariableModifier


def test_shared_variable_modifier():
    parameter = theano.shared(0.)
    modifier = SharedVariableModifier(parameter, lambda n: 10. / n)

    dummy_batch = {0: numpy.zeros(5)}
    modifier.after_batch(dummy_batch)

    new_value = parameter.get_value()
    assert_allclose(new_value, 2.)

    parameter.set_value(10.)
    modifier = SharedVariableModifier(parameter, lambda n, val: val * 0.2)

    modifier.after_batch(dummy_batch)

    new_value = parameter.get_value()
    assert_allclose(new_value, 10. * 0.2)
