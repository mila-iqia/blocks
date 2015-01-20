from __future__ import print_function

import argparse
import logging
import os
import pprint

import numpy
import six
import theano
from pylearn2.datasets.dataset import Dataset
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import IndexSpace
from pylearn2.training_algorithms.sgd import SGD
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.pylearn2 import (
    Pylearn2Model, Pylearn2Cost, Pylearn2Train, Pylearn2LearningRule,
    SGDLearningRule)
from blocks.select import Selector

floatX = theano.config.floatX

logger = logging.getLogger(__name__)


class ChainDataset(Dataset):
    """Training data generator.

    Supports the PyLearn2 dataset interface.

    """
    num_states = 3
    trans_prob = numpy.array([[0.1, 0.5, 0.4],
                              [0.1, 0.9, 0.0],
                              [0.3, 0.3, 0.4]])
    values, vectors = numpy.linalg.eig(trans_prob.T)
    equilibrium = vectors[:, values.argmax()]
    equilibrium = equilibrium / equilibrium.sum()
    trans_entropy = trans_prob * numpy.log(trans_prob + 1e-6)
    entropy = equilibrium.dot(trans_entropy).sum()

    data_specs = (SequenceDataSpace(IndexSpace(max_labels=num_states,
                                               dim=1)),
                  'x')

    def __init__(self, rng, seq_len):
        self.rng = rng
        self.seq_len = seq_len

    def iterator(self, batch_size, data_specs,
                 return_tuple, mode, num_batches, rng=None):
        """Returns a PyLearn2 compatible iterator."""
        assert return_tuple

        dataset = self

        class Iterator(six.Iterator):
            # This is not true, but let PyLearn2 think that this
            # iterator is not stochastic.
            # Makes life easier for now.
            stochastic = False
            num_examples = num_batches * batch_size

            def __init__(self, **kwargs):
                self.batches_retrieved = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.batches_retrieved < num_batches:
                    self.batches_retrieved += 1
                    return (dataset._next_batch(batch_size)[..., None],)
                raise StopIteration()
        return Iterator()

    def get_num_examples(self):
        """Part of the PyLearn2 Dataset interface."""
        return float('inf')

    def _next_single(self):
        states = [0]
        while len(states) != self.seq_len:
            states.append(numpy.random.multinomial(
                1, self.trans_prob[states[-1]]).argmax())
        return states

    def _next_batch(self, batch_size):
        """Generate random sequences from the family."""
        x = numpy.zeros((self.seq_len, batch_size), dtype='int64')
        for i in range(batch_size):
            x[:, i] = self._next_single()
        return x


def main(mode, save_path, steps, time_budget, reset):

    num_states = ChainDataset.num_states

    if mode == "train":
        # Experiment configuration
        rng = numpy.random.RandomState(1)
        batch_size = 50
        seq_len = 100
        dim = 10
        feedback_dim = 8

        # Build the bricks and initialize them
        transition = GatedRecurrent(name="transition", activation=Tanh(),
                                    dim=dim)
        generator = SequenceGenerator(
            LinearReadout(readout_dim=num_states, source_names=["states"],
                          emitter=SoftmaxEmitter(name="emitter"),
                          feedbacker=LookupFeedback(
                              num_states, feedback_dim, name='feedback'),
                          name="readout"),
            transition,
            weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
            name="generator")
        generator.push_initialization_config()
        transition.weights_init = Orthogonal()
        generator.initialize()

        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in Selector(generator).get_params().items()],
                        width=120))
        logger.info("Markov chain entropy: {}".format(
            ChainDataset.entropy))
        logger.info("Expected min error: {}".format(
            -ChainDataset.entropy * seq_len * batch_size))

        if os.path.isfile(save_path) and not reset:
            model = Pylearn2Model.load(save_path)
        else:
            model = Pylearn2Model(generator)

        # Build the cost computation graph.
        # Note: would be probably nicer to make cost part of the model.
        x = tensor.ltensor3('x')
        cost = Pylearn2Cost(model.brick.cost(x[:, :, 0]).sum())

        dataset = ChainDataset(rng, seq_len)
        sgd = SGD(learning_rate=0.0001, cost=cost,
                  batch_size=batch_size, batches_per_iter=10,
                  monitoring_dataset=dataset,
                  monitoring_batch_size=batch_size,
                  monitoring_batches=1,
                  learning_rule=Pylearn2LearningRule(
                      SGDLearningRule(),
                      dict(training_objective=cost.cost)))
        train = Pylearn2Train(dataset, model, algorithm=sgd,
                              save_path=save_path, save_freq=10)
        train.main_loop(time_budget=time_budget)
    elif mode == "sample":
        model = Pylearn2Model.load(save_path)
        generator = model.brick

        sample = ComputationGraph(generator.generate(
            n_steps=steps, batch_size=1, iterate=True)).function()

        states, outputs, costs = [data[:, 0] for data in sample()]

        numpy.set_printoptions(precision=3, suppress=True)
        print("Generation cost:\n{}".format(costs.sum()))

        freqs = numpy.bincount(outputs).astype(floatX)
        freqs /= freqs.sum()
        print("Frequencies:\n {} vs {}".format(freqs,
                                               ChainDataset.equilibrium))

        trans_freqs = numpy.zeros((num_states, num_states), dtype=floatX)
        for a, b in zip(outputs, outputs[1:]):
            trans_freqs[a, b] += 1
        trans_freqs /= trans_freqs.sum(axis=1)[:, None]
        print("Transition frequencies:\n{}\nvs\n{}".format(
            trans_freqs, ChainDataset.trans_prob))
    else:
        assert False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Case study of generating a Markov chain with RNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "sample"],
        help="The mode to run. Use `train` to train a new model"
             " and `sample` to sample a sequence generated by an"
             " existing one.")
    parser.add_argument(
        "save_path", default="sine",
        help="The part to save PyLearn2 model")
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of steps to plot")
    parser.add_argument(
        "--reset", action="store_true", default=False,
        help="Start training from scratch")
    parser.add_argument(
        "--time-budget", default=None, type=float,
        help="Train for this many seconds")
    args = parser.parse_args()
    main(**vars(args))
