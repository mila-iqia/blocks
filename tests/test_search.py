import numpy

from blocks.search import BeamSearch


def test_beam_search_smallest():
    a = numpy.array([[3, 6, 4], [1, 2, 7]])
    ind, mins = BeamSearch._smallest(a, 2)
    assert numpy.all(numpy.array(ind) == numpy.array([[1, 1], [0, 1]]))
    assert numpy.all(mins == [1, 2])
