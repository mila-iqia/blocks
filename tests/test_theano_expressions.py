from numpy.testing import assert_allclose

from blocks.theano_expressions import L2_norm


def test_l2_norm():
    assert_allclose(L2_norm([2]).eval(), 2.0)
    assert_allclose(L2_norm([3, 4]).eval(), 5.0)
    assert_allclose(L2_norm([3, [1, 2]]).eval(), 14.0 ** 0.5)
    assert_allclose(
        L2_norm([3, [1, 2], [[1, 2], [3, 4]]]).eval(), 44.0 ** 0.5)
