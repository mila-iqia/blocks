from blocks.utils import dict_union

__author__ = "Dmitry Serdyuk <serdyuk.dmitriy@gmail.com>"

from abc import ABCMeta, abstractmethod

import numpy as np

from theano import function
from theano import config
from theano import tensor as tt

from blocks.bricks.sequence_generators import BaseSequenceGenerator


class Search(object):
    """Abstract search class

    Parameters
    ----------
        :param sequence_generator : sequence generator to be used
    """
    __metaclass__ = ABCMeta

    def __init__(self, sequence_generator):
        if not isinstance(sequence_generator, BaseSequenceGenerator):
            raise ValueError("The input should be BaseSequenceGenerator")

        self.generator = sequence_generator
        self.compiled = False

    @abstractmethod
    def compile(self):
        self.compiled = True

    @abstractmethod
    def search(self, beam_size, **kwargs):
        """Performs search

        Parameters
        ----------
            **kwargs : Arguments needed by sequence generator

        outputs : Generated sequence
        """
        if not self.compiled:
            self.compile()


class GreedySearch(Search):

    def __init__(self, sequence_generator):
        super(GreedySearch, self).__init__(sequence_generator)


class BeamSearch(Search):
    """Beam search

    Parameters
    ----------
    :param beam_size : int, size of beam
    :param batch_size : int, size of batch, should be dividable by `beam_size`
    :param sequence_generator : a sequence generator brick
    """
    def __init__(self, beam_size, batch_size, sequence_generator):
        super(BeamSearch, self).__init__(sequence_generator)
        assert batch_size % beam_size == 0
        self.beam_size = beam_size
        self.batch_size = batch_size

    def compile(self, *args, **kwargs):
        """Compiles functions for beamsearch

        Parameters
        ----------
        :param args:
        :param kwargs : States, contexts, and glimpses are expected as
               keyword arguments
        :return:
        """
        states = {name: kwargs[name] for name
                  in self.generator.state_names}
        contexts = {name: kwargs[name] for name
                    in self.generator.context_names}
        glimpses = {name: kwargs[name] for name
                    in self.generator.glimpse_names}
        input_dict = dict_union(states, contexts, glimpses)

        next_glimpses = self.generator.transition.take_look(
            return_dict=True, **input_dict)

        self.next_glimpse_computer = function(input_dict.values(),
                                              next_glimpses.values())

        self.outputs = tt.TensorType('int64', (False,) *
                                     self.generator.get_dim("outputs"))()
        next_readouts = self.generator.readout.readout(
            feedback=self.readout.feedback(self.outputs),
            **input_dict)

        self.next_readouts_computer = next_readouts.eval()

        next_outputs, next_states, next_costs = \
            self.generator.compute_next_states(next_readouts,
                                               next_glimpses,
                                               **kwargs)

        self.next_states_computers = [var.eval() for var in next_states]
        self.next_outputs_computer = next_outputs.eval()
        self.next_costs_computer = next_costs.eval()

        next_probs = self.generator.readout.emit_probs(next_readouts)
        self.next_probs_computer = next_probs.eval()

        super(BeamSearch, self).compile(*args, **kwargs)

    @classmethod
    def _chunks(cls, list, n):
        """ Yields successive n-sized chunks from l.

        :param list: list to be divided into chunks
        :param n: chunk size
        :return: a list of lists with dimension (l / n, n), where l is
                 length of the `list`
        """
        for i in xrange(0, len(list), n):
            yield list[i:i + n]

    @classmethod
    def _top_probs(cls, probs, beam_size):
        """
        Returns indexes of elements with highest probabilities

        :param probs: a 3d array of probabilities (time, batch, readout_dim)
        :param beam_size: beam size, number of top probs to return
        :return: tuple of (indexes, top probabilities)
        """
        args = np.argpartition(-probs.flatten(), beam_size)
        # convert args back
        indexes = np.unravel_index(args, probs.shape)
        return indexes, probs[indexes]

    def search(self, eol_symbol=-1, max_length=512, **kwargs):
        """Performs greedy search

        :param kwargs: Contexts are expected as
               keyword arguments
        :param eol_symbol: End of line symbol, the search stops when the
               symbol is generated
        :param max_length: Maximum sequence length, the search stops when it
               is reached
        :return: Most probable sequence, corresponding probabilities and costs
        """
        super(BeamSearch, self).search(**kwargs)
        states = {name: kwargs[name] for name
                  in self.generator.state_names}
        contexts = {name: kwargs[name] for name
                    in self.generator.context_names}
        glimpses = {name: kwargs[name] for name
                    in self.generator.glimpse_names}
        inputs = dict_union(states, contexts, glimpses)

        outputs_dim = self.generator.get_dim('outputs')
        current_outputs = np.array(outputs_dim)  # TODO: outputs dimensionality
        for i in xrange(max_length):
            # Compute probabilities
            next_glimpses = self.next_glimpse_computer(inputs.values())
            next_readouts = self.next_readouts_computer(inputs.values())

            next_outputs = self.next_outputs_computer(inputs.values() +
                                                      next_readouts.values() +
                                                      [next_glimpses])
            next_states = [computer(inputs.values() +
                                    next_readouts.values() +
                                    [next_glimpses])
                           for computer in self.next_states_computers]
            next_costs = self.next_costs_computer(inputs.values() +
                                                  next_readouts.values() +
                                                  [next_glimpses])
            next_probs = self.next_probs_computer(next_readouts)

            # Choose top beam_size
            prob_batches = self._chunks(next_probs, self.batch_size)
            # Top probs
            indexes, top_probs = zip(*[self._top_probs(batch, self.beam_size)
                                       for batch in prob_batches])
            outputs = [ind[-1] for ind in indexes]
            #current_outputs.
            # Next state
            # Next output

            # if all meet eol
            if False:
                break
        raise NotImplementedError()  # TODO: remove when implemented
