import numpy
from pylearn2.space import VectorSpace
from pylearn2.testing.datasets import random_dense_design_matrix
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

from blocks.bricks import Sigmoid, MLP
from blocks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from blocks.pylearn2 import BlocksModel, BlocksCost


def test_pylearn2_trainin():
    # Construct the model
    mlp = MLP(activations=[Sigmoid(), Sigmoid()], dims=[784, 100, 784],
              weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
    mlp.initialize()
    cost = SquaredError()

    block_cost = BlocksCost(cost)
    block_model = BlocksModel(mlp, (VectorSpace(dim=784), 'features'))

    # Load the data
    rng = numpy.random.RandomState(14)
    train_dataset = random_dense_design_matrix(rng, 1024, 784, 10)
    valid_dataset = random_dense_design_matrix(rng, 1024, 784, 10)

    # Training algorithm
    sgd = SGD(learning_rate=0.01, cost=block_cost, batch_size=128,
              monitoring_dataset=valid_dataset)
    train = Train(train_dataset, block_model, algorithm=sgd)
    train.main_loop(time_budget=3)
