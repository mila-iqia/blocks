__author__ = "Dmitry Serdyuk <serdyuk.dmitriy@gmail.com>"
from abc import ABCMeta, abstractmethod

from sequence_generators import BaseSequenceGenerator


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
        :param **kwargs: Arguments needed by sequence generator

        Outputs:
            Generated sequence
        """
        pass


class GreedySearch(Search):

    def __init__(self, sequence_generator):
        super(GreedySearch, self).__init__(sequence_generator)


class BeamSearch(Search):
    def compile(self):

        pass

    def search(self, beam_size, **kwargs):
        # Initial state
        # TODO: compute it

        for i in xrange(0):
            # Compute probabilities
            # Choose top beam_size
            # Next state
            # Next output
            pass

        pass
