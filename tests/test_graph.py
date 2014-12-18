from theano import tensor

from blocks.bricks import Brick

from blocks.graph import ComputationGraph

from tests.test_brick import TestBrick


def test_application_graph_auxiliary_vars():
    X = tensor.matrix('X')
    Brick.lazy = True
    brick = TestBrick()
    Y = brick.access_application_call(X)
    graph = ComputationGraph(outputs=[Y])
    test_val_found = False
    for var in graph.variables:
        if var.name == 'test_val':
            test_val_found = True
            break
    assert test_val_found
