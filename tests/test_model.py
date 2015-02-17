from theano import tensor

from blocks.bricks import MLP, Tanh
from blocks.model import DifferentiableCostModel


def test_differentiable_cost_model():
    x = tensor.matrix('x')
    mlp1 = MLP([Tanh(), Tanh()], [10, 20, 30], name="mlp1")
    mlp2 = MLP([Tanh()], [30, 40], name="mlp2")
    h1 = mlp1.apply(x)
    h2 = mlp2.apply(h1)

    model = DifferentiableCostModel(h2.sum())
    assert model.get_top_bricks() == [mlp1, mlp2]
    # The order of parameters returned is deterministic but
    # not sensible.
    assert list(model.get_params().items()) == [
        ('/mlp2/linear_0.b', mlp2.linear_transformations[0].b),
        ('/mlp1/linear_1.b', mlp1.linear_transformations[1].b),
        ('/mlp1/linear_0.b', mlp1.linear_transformations[0].b),
        ('/mlp1/linear_0.W', mlp1.linear_transformations[0].W),
        ('/mlp1/linear_1.W', mlp1.linear_transformations[1].W),
        ('/mlp2/linear_0.W', mlp2.linear_transformations[0].W)]
