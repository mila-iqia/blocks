from blocks.utils import dict_union

__author__ = "Dmitry Serdyuk <serdyuk.dmitriy@gmail.com>"

from abc import ABCMeta, abstractmethod

from theano import function

from blocks.bricks.sequence_generators import BaseSequenceGenerator


class Search(object):
    """Abstract search class"""
    __metaclass__ = ABCMeta

    def __init__(self, sequence_generator):
        if not isinstance(sequence_generator, BaseSequenceGenerator):
            raise ValueError("The input should be BaseSequenceGenerator")

        self.generator = sequence_generator

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def search(self, beam_size, **kwargs):
        """Performs search

        Parameters
        ----------
            **kwargs : Arguments needed by sequence generator

        outputs : Generated sequence
        """
        pass


class GreedySearch(Search):

    def __init__(self, sequence_generator):
        super(GreedySearch, self).__init__(sequence_generator)


class BeamSearch(Search):
    """A class that provides an interface for search in any sequence
    generator.

    Parameters
    ----------
        beam_size : int, size of beam
        batch_size : int, size of batch, should be dividable by `beam_size`
        sequence_generator : a sequence generator brick
    """
    def __init__(self, beam_size, batch_size, sequence_generator):
        super(BeamSearch, self).__init__(sequence_generator)
        assert batch_size % beam_size == 0
        self.beam_size = beam_size
        self.batch_size = batch_size

    def compile(self, *args, **kwargs):
        initial_state = []
        for name in self.generator.state_names + self.generator.glimpse_names:
            initial_state += self.generator.initial_state(name,
                                                          self.batch_size,
                                                          *args,
                                                          **kwargs)

        self.initial_state_computer = function([], initial_state)

        next_glimpses = self.generator.compute_next_glimpses(**kwargs)
        states = {name: kwargs[name] for name
                  in self.generator.state_names}
        contexts = {name: kwargs[name] for name
                    in self.generator.context_names}
        glimpses = {name: kwargs[name] for name
                    in self.generator.glimpse_names}

        self.next_glimpse_computer = function(dict_union(states, contexts, glimpses),
                                              next_glimpses)

        next_readouts = self.generator.compute_next_readouts()
        self.next_readouts_computer = function([], )

    def search(self, **kwargs):

        # Initial state
        # TODO: compute it

        for i in xrange(0):
            # Compute probabilities
            # Choose top beam_size
            # Next state
            # Next output
            pass

        pass
