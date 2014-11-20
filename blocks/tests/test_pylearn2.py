from blocks.bricks import Sigmoid, MLP
from blocks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from blocks.pylearn2 import BlocksModel, BlocksCost

from pylearn2.datasets.mnist import MNIST
from pylearn2.space import VectorSpace
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD


def test_pylearn2_trainin():
    # Construct the model
    mlp = MLP(activations=[Sigmoid(), Sigmoid()], dims=[784, 100, 784],
              weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
    mlp.initialize()
    cost = SquaredError()

    block_cost = BlocksCost(cost)
    block_model = BlocksModel(mlp, (VectorSpace(dim=784), 'features'))

    # Load the data
    mnist_train = MNIST('train')
    mnist_test = MNIST('test')

    # Training algorithm
    sgd = SGD(learning_rate=0.01, cost=block_cost, batch_size=128,
              monitoring_dataset=mnist_test)
    train = Train(mnist_train, block_model, algorithm=sgd)
    train.main_loop(time_budget=5)
