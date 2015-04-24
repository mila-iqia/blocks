import theano
import numpy
from numpy.testing import assert_allclose

from collections import OrderedDict
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from theano import tensor

from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.monitoring.aggregation import Mean

def test_mean_aggregator():
  num_examples = 4
  num_batches = 2
  features = numpy.array([[0, 3], 
                       [2, 9],
                       [2, 4],
                       [5, 1]], dtype = theano.config.floatX)

  dataset = IndexableDataset( OrderedDict([('features', features)]))

  data_stream = DataStream(dataset,
      iteration_scheme = SequentialScheme(num_examples, 
                                          num_examples/num_batches))

  x = tensor.matrix('features')
  y = (x**2).mean(axis = 0)
  y.name = 'y'
  z = y.sum()
  z.name = 'z'

  y.tag.aggregation_scheme = Mean(y, 1.0)
  z.tag.aggregation_scheme = Mean(z, 1.0)

  assert_allclose(DatasetEvaluator([y]).evaluate(data_stream)['y'],
                  numpy.array([8.25, 26.75],dtype = theano.config.floatX))
  assert_allclose(DatasetEvaluator([z]).evaluate(data_stream)['z'],
                  numpy.array([35],dtype = theano.config.floatX))