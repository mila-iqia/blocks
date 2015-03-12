from numpy.testing import assert_raises
from theano import tensor

from blocks.utils import (check_theano_variable, unpack, equizip,
                          IterableLengthMismatch)


def test_unpack():
    assert unpack((1, 2)) == [1, 2]
    assert unpack([1, 2]) == [1, 2]
    assert unpack([1]) == 1
    test = object()
    assert unpack(test) is test
    assert_raises(ValueError, unpack, [1, 2], True)


def test_check_theano_variable():
    check_theano_variable(None, 3, 'float')
    check_theano_variable([[1, 2]], 2, 'int')
    assert_raises(ValueError, check_theano_variable,
                  tensor.vector(), 2, 'float')
    assert_raises(ValueError, check_theano_variable,
                  tensor.vector(), 1, 'int')


def test_equizip():
    assert_raises(IterableLengthMismatch, list, equizip(range(4), range(5)))
    assert_raises(IterableLengthMismatch, list, equizip(range(4), range(3)))
    assert (list(equizip((i for i in range(2)), (i for i in range(2)))) ==
            [(0, 0), (1, 1)])
