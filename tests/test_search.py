__author__ = 'serdyuk'

import numpy as np

from blocks.search import BeamSearch


def test_top_probs():
    """Test top probabilities"""
    a = np.array([[[3, 6, 4], [1, 2, 7]]])
    ind, maxs = BeamSearch._top_probs(a, 2)
    assert np.all(np.array(ind) == (np.array([[1, 0], [2, 1]])))
    assert np.all(maxs == [7, 6])

