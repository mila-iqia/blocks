from numpy.testing import assert_raises
from theano import tensor

from blocks.utils import check_theano_variable, unpack


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
