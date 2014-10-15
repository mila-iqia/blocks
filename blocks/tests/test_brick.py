from blocks.bricks import Brick
from theano import tensor

class Identity(Brick):

    @Brick.apply_method
    def apply(self, a, b=1, **kwargs):
        if isinstance(a, list):
            a = a[0]
        return [a, b] + kwargs.values()

def test_apply_method():
    brick = Identity()
    x = tensor.vector('x')
    y = tensor.vector('y')
    z = tensor.vector('z')

    def check_output_variable(o):
        assert o.tag.owner == brick
        assert o.owner.inputs[0].tag.owner == brick

    # Case 1: both positional arguments are provided.
    u, v = brick.apply(x, y)
    for o in [u, v]:
        check_output_variable(o)

    # Case 2: `b` is given as a keyword argument.
    u, v = brick.apply(x, b=y)
    for o in [u, v]:
        check_output_variable(o)

    # Case 3: two positional and one keyword argument.
    u, v, w = brick.apply(x, y, c=z)
    for o in [u, v, w]:
        check_output_variable(o)

    # Case 4: one positional argument.
    u, v = brick.apply(x)
    check_output_variable(u)
    assert v == 1

    # Case 5: variable was wrapped in a list. We can not handle that.
    try:
        u, v = brick.apply([x])
        check_output_variable(u)
    except AttributeError:
        pass
    else:
        assert False

