from blocks.utils.containers import OrderedSet


def test_ordered_set():
    c = OrderedSet(range(3))
    assert len(c) == 3
    assert 2 in c
    assert 3 not in c
    c.add(-1)
    assert list(c) == [0, 1, 2, -1]
    c.discard(1)
    assert 1 not in c
    assert list(c) == [0, 2, -1]
    assert list(reversed(c)) == [-1, 2, 0]
    assert -1 == c.pop()
    assert list(c) == [0, 2]
    assert str(c) == 'OrderedSet([0, 2])'
    assert not c == OrderedSet([2, 0])
    assert c == {0, 2}
    assert not c == 5
