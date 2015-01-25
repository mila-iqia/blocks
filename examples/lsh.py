import theano
from theano import tensor
from theano.ifelse import ifelse

from blocks.bricks import MLP, Sigmoid, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.lookup import Hash, LookupTable
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_zeros

from blocks.datasets import BatchDataStream
from blocks.datasets.schemes import ConstantScheme
from blocks.datasets.text import OneBillionWord, NGramStream

from blocks.algorithms import GradientDescent, SteepestDescent
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)

bits = 4
n_gram_order = 5
vocab_size = 100000
num_hidden = 256
embedding_size = 300
batch_size = 1


def main():

    # The input and the targets
    x = tensor.lmatrix('features')
    y = tensor.lmatrix('targets')

    # Input -> hidden state
    lookup = LookupTable(length=vocab_size, dim=embedding_size, name='lookup',
                         weights_init=IsotropicGaussian(0.01))
    lookup.initialize()
    embeddings = lookup.lookup(x)
    embeddings = embeddings.reshape(
        (embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]))
    hidden = MLP(activations=[Sigmoid()],
                 dims=[lookup.dim * n_gram_order, num_hidden],
                 weights_init=IsotropicGaussian(0.01),
                 biases_init=Constant(0.001))
    hidden.initialize()
    hidden_state = hidden.apply(embeddings)

    # Initialize the weight matrix for the output layer
    output = MLP(activations=[Softmax()], dims=[num_hidden, vocab_size],
                 weights_init=IsotropicGaussian(0.001), use_bias=False)
    output.initialize()
    W = output.linear_transformations[0].params[0]

    real_cost = CategoricalCrossEntropy().apply(y.flatten(), output)

    # Create a hasher and hash the weight matrix vectors and the hidden state
    hasher = Hash(num_hidden, bits)
    W_hashes = hasher.apply(W)

    # We want to store the weight hashes somewhere
    hashes = shared_floatx_zeros(vocab_size, dtype='int64', name='hashes')
    f = theano.function([], [], updates=[(hashes, W_hashes)])
    f()

    # Calculate neighbors and add the target to the neighbours if not there
    hidden_state_hash = hasher.apply(hidden_state.T).sum()
    neighbors = tensor.eq(hashes, hidden_state_hash).nonzero()[0]
    neighbors = ifelse(tensor.eq(neighbors, y.flatten()[0]).sum(), neighbors,
                       tensor.concatenate([neighbors, y.flatten()]))

    # Calculate the cost of the softmax approximation
    y_hat_estimate = Softmax().apply(tensor.dot(hidden_state, W[:, neighbors]))
    y_estimate = tensor.eq(neighbors, y.flatten()[0]).nonzero()[0]
    cost = -(tensor.log(y_hat_estimate[0])[y_estimate[0]]).mean()
    cost.name = 'cost'
    # cost = CategoricalCrossEntropy().apply(y_estimate, y_hat_estimate)

    # Function to update hashes after update
    import cPickle
    with open('/data/lisa/datasets/1-billion-word/processed/'
              'one_billion_counter_full.pkl') as f:
        counter = cPickle.load(f)
    vocab = dict(counter.most_common(vocab_size - 3) +
                 [('<S>', 0), ('</S>', 1), ('<UNK>', 2)])
    vocab = dict((word, index) for word, index in vocab.items()
                 if index < vocab_size)
    train = OneBillionWord('training', range(1, 100), vocab)
    stream = train.get_default_stream()
    batch_stream = BatchDataStream(stream, ConstantScheme(1))
    n_gram_stream = NGramStream(n_gram_order, batch_stream,
                                iteration_scheme=ConstantScheme(1))

    test = OneBillionWord('test', range(1, 100), vocab)
    test_stream = test.get_default_stream()
    test_batch_stream = BatchDataStream(test_stream, ConstantScheme(1))
    test_n_gram_stream = NGramStream(n_gram_order, test_batch_stream,
                                     iteration_scheme=ConstantScheme(100))

    cg = ComputationGraph(cost)
    algorithm = GradientDescent(
        cost=cost, step_rule=SteepestDescent(learning_rate=0.0001),
        params=[var for var in cg.shared_variables if var not in [hashes] +
                VariableFilter(bricks=[hasher])(cg.shared_variables)])
    algorithm.add_updates(
        [(hashes,
         tensor.set_subtensor(hashes[neighbors], hasher.apply(W, neighbors)))])

    main_loop = MainLoop(
        model=None, data_stream=n_gram_stream, algorithm=algorithm,
        extensions=[FinishAfter(after_n_epochs=5),
                    DataStreamMonitoring([real_cost], test_n_gram_stream,
                                         prefix='test'),
                    TrainingDataMonitoring([cost], prefix='train',
                                           after_every_batch=True),
                    Printing(after_every_batch=True)])
    main_loop.run()

if __name__ == "__main__":
    main()
