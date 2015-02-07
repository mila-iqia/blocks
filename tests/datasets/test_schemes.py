from numpy.testing import assert_raises

from blocks.datasets.schemes import ConstantScheme, SequentialScheme


def iterator_requester(scheme):
    def get_request_iterator(*args, **kwargs):
        scheme_obj = scheme(*args, **kwargs)
        return scheme_obj.get_request_iterator()
    return get_request_iterator


def test_constant_scheme():
    get_request_iterator = iterator_requester(ConstantScheme)
    assert list(get_request_iterator(3, num_examples=7)) == [3, 3, 1]
    assert list(get_request_iterator(3, num_examples=9)) == [3, 3, 3]
    assert list(get_request_iterator(3, num_examples=2)) == [2]
    assert list(get_request_iterator(2, times=3)) == [2, 2, 2]
    assert list(get_request_iterator(3, times=1)) == [3]
    it = get_request_iterator(3)
    assert [next(it) == 3 for _ in range(10)]
    assert_raises(ValueError, get_request_iterator, 10, 2, 2)


def test_sequential_scheme():
    get_request_iterator = iterator_requester(SequentialScheme)
    assert list(get_request_iterator(5, 3)) == [[0, 1, 2], [3, 4]]
    assert list(get_request_iterator(4, 2)) == [[0, 1], [2, 3]]
