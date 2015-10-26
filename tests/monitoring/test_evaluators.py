import numpy
import theano
from fuel.datasets import IterableDataset
from numpy.testing import assert_raises

from blocks.graph import ComputationGraph
from blocks.monitoring.evaluators import DatasetEvaluator
from tests.monitoring.test_aggregation import TestBrick


def test_dataset_evaluators():
    X = theano.tensor.matrix('X')
    brick = TestBrick(name='test_brick')
    Y = brick.apply(X)
    graph = ComputationGraph([Y])
    monitor_variables = [v for v in graph.auxiliary_variables]
    validator = DatasetEvaluator(monitor_variables)

    data = [numpy.arange(1, 5, dtype=theano.config.floatX).reshape(2, 2),
            numpy.arange(10, 16, dtype=theano.config.floatX).reshape(3, 2)]
    data_stream = IterableDataset(dict(X=data)).get_example_stream()

    values = validator.evaluate(data_stream)
    assert values['test_brick_apply_V_squared'] == 4
    numpy.testing.assert_allclose(
        values['test_brick_apply_mean_row_mean'], numpy.vstack(data).mean())
    per_batch_mean = numpy.mean([batch.mean() for batch in data])
    numpy.testing.assert_allclose(
        values['test_brick_apply_mean_batch_element'], per_batch_mean)

    with assert_raises(Exception) as ar:
        data_stream = IterableDataset(dict(X2=data)).get_example_stream()
        validator.evaluate(data_stream)
    assert "Not all data sources" in ar.exception.args[0]
