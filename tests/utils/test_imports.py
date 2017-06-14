import sys


def test_no_theano_import():
    del sys.modules['theano']
    import blocks.utils
    assert 'theano' not in sys.modules
    from blocks.utils import dict_union
    assert 'theano' not in sys.modules


def test_imports():
    from blocks.utils import dict_union
    from blocks.utils import check_theano_variable
    from blocks.utils.utils import dict_union
    from blocks.utils.theano_utils import check_theano_variable
