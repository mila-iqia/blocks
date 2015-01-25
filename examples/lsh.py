import numpy
import theano
from theano import tensor

from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph, OUTPUT
from blocks.bricks.lookup import Hash, LookupTable
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_zeros

bits = 4
n_gram_order = 5
vocab_size = 1000
num_hidden = 256
embedding_size = 300
batch_size = 1

x = tensor.lmatrix('features')
y = tensor.lmatrix('targets')
hashes = shared_floatx_zeros(vocab_size, dtype='int64', name='hashes')

lookup = LookupTable(length=vocab_size, dim=embedding_size, name='lookup',
                     weights_init=IsotropicGaussian(0.01))
lookup.initialize()
embeddings = lookup.lookup(x)
embeddings = embeddings.reshape((embeddings.shape[0],
                                 embeddings.shape[1] * embeddings.shape[2]))
mlp = MLP(activations=[Rectifier(), Softmax()],
          dims=[lookup.dim * n_gram_order, num_hidden, vocab_size],
          weights_init=IsotropicGaussian(0.01), biases_init=Constant(0.001))
mlp.initialize()
y_hat = mlp.apply(embeddings)

cg = ComputationGraph(y_hat)

W = mlp.linear_transformations[1].params[0].T
hasher = Hash(bits, num_hidden)
W_hashes = hasher.apply(W)
hidden_state, = VariableFilter(
    roles=[OUTPUT], bricks=[mlp.linear_transformations[0]])(cg.variables)
hidden_state_hash = hasher.apply(hidden_state).sum()

neighbors = tensor.eq(hashes, hidden_state_hash).nonzero()[0]

# Initialize the hashes
f = theano.function([], [], updates=[(hashes, W_hashes)])
f()
# Function to calculate neighbors
g = theano.function([x], [neighbors])
# Function to update hashes after update
h = theano.function(
    [], [], updates=[(hashes,
                      tensor.set_subtensor(hashes[neighbors],
                                           hasher.apply(W, neighbors)))])
# Given neighbors + target, update
sample = tensor.lvector('sample')
y_hat_estimate = Softmax().apply(tensor.dot(hidden_state, W[:, sample]))
cost = CategoricalCrossEntropy().apply(y, y_hat_estimate)
print(g(numpy.random.randint(vocab_size, size=(batch_size, n_gram_order))))
