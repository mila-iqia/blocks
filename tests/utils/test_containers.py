from blocks.utils.containers import AnnotatingList


def test_annotating_list():
    lst = AnnotatingList(range(10))
    assert repr(lst) == repr(list(range(10)))
    assert lst == list(range(10))
    assert lst != list(range(9))
    assert lst[0] == 0
    lst[0] = 10
    del lst[0]
    lst.insert(0, 0)
    assert lst == list(range(10))
