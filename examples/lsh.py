import cPickle
import logging
from argparse import ArgumentParser
from itertools import izip, chain, count

from theano import tensor
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.bricks import MLP, Rectifier, Softmax, Linear
from blocks.bricks.lookup import Hash, LookupTable
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_zeros
from blocks.filter import VariableFilter

from fuel.transformers import Batch
from fuel.transformers.text import NGrams
from fuel.datasets import OneBillionWord
from fuel.schemes import ConstantScheme

from blocks.algorithms import GradientDescent, Scale
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)


def main(training_scheme, vocab_size, n_gram_order, embedding_size, bits,
         num_hidden):
    # The input and the targets
    x = tensor.lrow('features')
    y = tensor.lvector('targets')
    y = y.dimshuffle(0, 'x')

    # Define the model
    lookup = LookupTable(length=vocab_size, dim=embedding_size, name='lookup')
    hidden_layer = MLP(activations=[Rectifier()],
                       dims=[lookup.dim * n_gram_order, num_hidden])
    output_layer = Linear(input_dim=num_hidden, output_dim=vocab_size)

    # Create the Theano variables
    embeddings = lookup.apply(x)
    embeddings = embeddings.flatten(ndim=2)
    hidden = hidden_layer.apply(embeddings)
    output = output_layer.apply(hidden)

    # Hash the target word embeddings
    hasher = Hash(num_hidden, bits)
    W = output_layer.params[0]  # Target word embeddings
    W_hashes = hasher.apply(W)

    # Store the hashes in a shared variable
    hashes = shared_floatx_zeros(vocab_size, dtype='int64', name='hashes')
    hashes.set_value(W_hashes.eval())

    # Calculate neighbors and add the target to the neighbours if not there
    if training_scheme == 'random':
        rng = RandomStreams(1)
        neighbors = rng.choice(a=vocab_size, size=(vocab_size // 2 ** bits,))
    if training_scheme == 'mips':
        hidden_state_hash = hasher.apply(hidden.T).sum()
        neighbors = tensor.eq(hashes, hidden_state_hash).nonzero()[0]
    if training_scheme != 'full':
        # Add the target if it wasn't in the same hash bin
        neighbors = ifelse(tensor.eq(neighbors, y.flatten()[0]).sum(),
                           neighbors,
                           tensor.concatenate([neighbors, y.flatten()]))

    # Calculate the cost of the full softmax
    real_cost = Softmax().categorical_cross_entropy(y.flatten(), output)
    real_cost.name = 'real_cost'

    # And the cost of softmax approximation, if applicable
    if training_scheme == 'full':
        cost = real_cost
    else:
        # Find the index of the target in the softmax subset
        y_estimate = tensor.eq(neighbors, y.flatten()[0]).nonzero()[0]
        # Calculate the softmax over the neighbours only
        z = tensor.dot(hidden, W[:, neighbors])
        cost = Softmax().categorical_cross_entropy(y_estimate, z)
    cost.name = 'cost'

    cost_difference = tensor.abs_(real_cost - cost)
    cost_difference.name = 'cost_difference'

    # Initialization
    lookup.weights_init = IsotropicGaussian(0.001)
    hidden_layer.weights_init = IsotropicGaussian(0.01)
    hidden_layer.biases_init = Constant(0.001)
    output_layer.weights_init = IsotropicGaussian(0.001)
    output_layer.biases_init = Constant(0.001)

    lookup.initialize()
    hidden_layer.initialize()
    output_layer.initialize()

    # Creat the vocabulary
    with open('/data/lisa/datasets/1-billion-word/processed/'
              'one_billion_counter_full.pkl') as f:
        counter = cPickle.load(f)
    freq_words = list(izip(*counter.most_common(vocab_size - 3)))[0]
    vocab = dict(izip(chain(['<UNK>', '<S>', '</S>'], freq_words), count()))

    # Load the training and test data
    train = OneBillionWord('training', range(1, 100), vocab)
    stream = train.get_example_stream()
    n_gram_stream = NGrams(n_gram_order, stream)
    train_stream = Batch(n_gram_stream, iteration_scheme=ConstantScheme(1))

    valid = OneBillionWord('heldout', range(1), vocab)
    stream = valid.get_example_stream()
    n_gram_stream = NGrams(n_gram_order, stream)
    valid_stream = Batch(n_gram_stream, iteration_scheme=ConstantScheme(1))

    # Training
    cg = ComputationGraph(cost)
    # Exclude the hash table from the parameters, we don't want to train it
    params = VariableFilter(bricks=[Linear, LookupTable])(cg.parameters)
    algorithm = GradientDescent(
        cost=cost, step_rule=Scale(learning_rate=0.001),
        params=params)
    algorithm.add_updates(
        [(hashes,
         tensor.set_subtensor(hashes[neighbors], hasher.apply(W, neighbors)))])

    main_loop = MainLoop(
        model=None, data_stream=train_stream, algorithm=algorithm,
        extensions=[DataStreamMonitoring([real_cost], valid_stream,
                                         prefix='valid', every_n_batches=5000),
                    TrainingDataMonitoring([cost, real_cost, cost_difference],
                                           prefix='train', after_batch=True),
                    Printing(every_n_batches=1),
                    Checkpoint('lsh.pkl', every_n_batches=500)])
    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Language model using LSH-MIPS")
    parser.add_argument('training_scheme', choices=['full', 'mips', 'random'])
    parser.add_argument('--vocab', type=int, default=50000)
    parser.add_argument('--ngram', type=int, default=5)
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--bits', type=int, default=4)
    args = parser.parse_args()
    main(args.training_scheme, args.vocab, args.ngram, args.dim, args.bits,
         args.hidden)
