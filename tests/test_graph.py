from theano import tensor

from blocks.graph import ComputationGraph

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
    assert ComputationGraph(a).inputs == cg.inputs

    cg2 = cg.replace({z: r})
    assert set(cg2.inputs) == {r}
    assert set([v.name for v in cg2.outputs]) == {'a', 'b'}
