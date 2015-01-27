import numpy

import theano

from blocks.graph import ComputationGraph
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.datasets import ContainerDataset
from tests.monitoring.test_aggregation import TestBrick

floatX = theano.config.floatX


def test_dataset_evaluators():
    X = theano.tensor.matrix('X')
    brick = TestBrick(name='test_brick')
    Y = brick.apply(X)
    graph = ComputationGraph([Y])
    monitor_variables = [v for v in graph.auxiliary_variables]
    validator = DatasetEvaluator(monitor_variables)

    data = [numpy.arange(1, 5, dtype=floatX).reshape(2, 2),
            numpy.arange(10, 16, dtype=floatX).reshape(3, 2)]
    data_stream = ContainerDataset(dict(X=data)).get_default_stream()

    values = validator.evaluate(data_stream)
    assert values['test_brick_apply_V_squared'] == 4
    numpy.testing.assert_allclose(
        values['test_brick_apply_mean_row_mean'], numpy.vstack(data).mean())
    per_batch_mean = numpy.mean([batch.mean() for batch in data])
    numpy.testing.assert_allclose(
        values['test_brick_apply_mean_batch_element'], per_batch_mean)
