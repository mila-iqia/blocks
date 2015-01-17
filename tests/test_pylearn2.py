import logging
import os

import numpy
import pylearn2
from pylearn2.testing.datasets import random_dense_design_matrix
from pylearn2.training_algorithms.sgd import SGD
from theano import tensor

from blocks.bricks import Sigmoid, MLP
from blocks.bricks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from blocks.pylearn2 import Pylearn2Model, Pylearn2Cost, Pylearn2Train
from blocks.pylearn2.examples.markov_chain import main


def test_pylearn2_training():
    # Construct the model
    mlp = MLP(activations=[Sigmoid(), Sigmoid()], dims=[784, 100, 784],
              weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
    mlp.initialize()
    cost = SquaredError()

    # Load the data
    rng = numpy.random.RandomState(14)
    train_dataset = random_dense_design_matrix(rng, 1024, 784, 10)
    valid_dataset = random_dense_design_matrix(rng, 1024, 784, 10)

    x = tensor.matrix('features')
    block_cost = Pylearn2Cost(cost.apply(x, mlp.apply(x)))
    block_model = Pylearn2Model(mlp)

    # Silence Pylearn2's logger
    logger = logging.getLogger(pylearn2.__name__)
    logger.setLevel(logging.ERROR)

    # Training algorithm
    sgd = SGD(learning_rate=0.01, cost=block_cost, batch_size=128,
              monitoring_dataset=valid_dataset)
    train = Pylearn2Train(train_dataset, block_model, algorithm=sgd)
    train.main_loop(time_budget=3)


def test_markov_chain():
    # Silence Pylearn2's logger
    logger = logging.getLogger(pylearn2.__name__)
    logger.setLevel(logging.ERROR)

    filename = 'unittest_markov_chain'
    main('train', filename, 0, 3, False)
    os.remove(filename)
