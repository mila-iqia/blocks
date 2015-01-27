import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.bricks.base import Brick
from blocks.graph import apply_noise, ComputationGraph
from tests.bricks.test_bricks import TestBrick

floatX = theano.config.floatX


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


def test_computation_graph():
    x = tensor.matrix('x')
    y = tensor.matrix('y')
    z = x + y
    a = z.copy()
    a.name = 'a'
    b = z.copy()
    b.name = 'b'
    r = tensor.matrix('r')

    cg = ComputationGraph([a, b])
    assert set(cg.inputs) == {x, y}
    assert set(cg.outputs) == {a, b}
    assert set(cg.variables) == {x, y, z, a, b}
    assert cg.variables[2] is z
    assert ComputationGraph(a).inputs == cg.inputs

    cg2 = cg.replace({z: r})
    assert set(cg2.inputs) == {r}
    assert set([v.name for v in cg2.outputs]) == {'a', 'b'}

    W = theano.shared(numpy.zeros((3, 3), dtype=floatX))
    cg3 = ComputationGraph([z + W])
    assert set(cg3.shared_variables) == {W}


def test_apply_noise():
    x = tensor.scalar()
    y = tensor.scalar()
    z = x + y

    cg = ComputationGraph([z])
    rng = RandomStreams(1)
    noised_cg = apply_noise(cg, [y], 1, rng)
    assert_allclose(noised_cg.outputs[0].eval({x: 1., y: 1.}),
                    2 + RandomStreams(1).normal().eval())
