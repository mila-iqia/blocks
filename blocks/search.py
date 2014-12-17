from blocks.utils import dict_union

__author__ = "Dmitry Serdyuk <serdyuk.dmitriy@gmail.com>"

from abc import ABCMeta, abstractmethod

import numpy as np

from theano import function
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
    """A class that provides an interface for search in any sequence
    generator.


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
        initial_state = []
        for name in self.generator.state_names + self.generator.glimpse_names:
            initial_state += self.generator.initial_state(name,
                                                          self.batch_size,
                                                          *args,
                                                          **kwargs)
        states = {name: kwargs[name] for name
                  in self.generator.state_names}
        contexts = {name: kwargs[name] for name
                    in self.generator.context_names}
        glimpses = {name: kwargs[name] for name
                    in self.generator.glimpse_names}

        self.initial_state_computer = function([], initial_state)

        next_glimpses = self.generator.transition.take_look(
            return_dict=True, **dict_union(states, glimpses, contexts))

        input_dict = dict_union(states, contexts, glimpses)

        self.next_glimpse_computer = function(input_dict.values(),
                                              next_glimpses)

        self.outputs = None # TODO: understand which type have outputs
        next_readouts = self.generator.readout.readout(
            feedback=self.readout.feedback(self.outputs),
            **kwargs)

        self.next_readouts_computer = function(input_dict.values() +
                                               [self.outputs],
                                               [next_readouts])

        next_outputs, next_states, next_costs = \
            self.generator.compute_next_states(next_readouts,
                                               next_glimpses,
                                               **kwargs)

        self.next_states_computer = function(input_dict.values() +
                                             [next_readouts, next_glimpses],
                                             [next_outputs, next_states,
                                              next_costs])

        next_costs = self.generator.readout.emit_probs(next_readouts)
        self.next_probs_computer = function([next_readouts],
                                            next_costs)

        super(BeamSearch, self).compile(*args, **kwargs)

    def search(self, **kwargs):
        """Performs greedy search

        :param kwargs: States, contexts, and glimpses are expected as
               keyword arguments
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

        current_outputs = np.array([])  # TODO: outputs dimensionality
        for i in xrange(0):
            # Compute probabilities
            readouts = self.next_readouts_computer(**dict_union())

            # Choose top beam_size
            # Next state
            # Next output
            pass
        raise NotImplementedError()  # TODO: remove when implemented
