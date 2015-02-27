import numpy

from blocks.search import BeamSearch


def test_top_probs():
    """Test top probabilities."""
    a = numpy.array([[[3, 6, 4], [1, 2, 7]]])
    ind, maxs = BeamSearch._top_probs(a, 2)
    assert numpy.all(numpy.array(ind) == (numpy.array([[1, 0], [2, 1]])))
    assert numpy.all(maxs == [7, 6])
