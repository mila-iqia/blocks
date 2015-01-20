import numpy

import theano

from blocks.graph import ComputationGraph
from blocks.bricks import VariableRole
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.datasets import ContainerDataset
from tests.monitoring.test_aggregation import TestBrick


def test_dataset_evaluators():
    X = theano.tensor.matrix('X')
    brick = TestBrick(name='test_brick')
    Y = brick.apply(X)
    graph = ComputationGraph([Y])
    monitor_variables = [
        v for v in graph.variables
        if getattr(v.tag, 'role', None) == VariableRole.MONITOR]
    validator = DatasetEvaluator({v.name: v for v in monitor_variables})

    data = [numpy.arange(1, 5).reshape(2, 2),
            numpy.arange(10, 16).reshape(3, 2)]
    data_stream = ContainerDataset(dict(X=data)).get_default_stream()

    values = validator.evaluate(data_stream)
    assert values['V_squared'] == 4
    numpy.testing.assert_allclose(
        values['mean_row_mean'], numpy.vstack(data).mean())
    per_batch_mean = numpy.mean([batch.mean() for batch in data])
    numpy.testing.assert_allclose(values['mean_batch_element'],
                                  per_batch_mean)
