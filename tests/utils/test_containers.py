from blocks.utils.containers import AnnotatingList


def test_annotating_list():
    l = AnnotatingList(range(10))
    assert repr(l) == repr(list(range(10)))
    assert l == list(range(10))
    assert l != list(range(9))
    assert l[0] == 0
    l[0] = 10
    del l[0]
    l.insert(0, 0)
    assert l == list(range(10))
