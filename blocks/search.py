from blocks.utils import dict_union

from abc import ABCMeta, abstractmethod

import numpy as np

from theano import function
from theano import config
from theano import tensor

from blocks.bricks.sequence_generators import BaseSequenceGenerator


class Search(object):
    """Abstract search class

    Parameters
    ----------
        :param sequence_generator : sequence generator to be used
    """
    __metaclass__ = ABCMeta

    def __init__(self, sequence_generator):
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
    beam_size : int, size of beam
    batch_size : int, size of batch, should be dividable by `beam_size`
    sequence_generator : a sequence generator brick

    """
    def __init__(self, beam_size, batch_size, sequence_generator, x, y, x_mask,
                 y_mask, f_input, inputs):
        super(BeamSearch, self).__init__(sequence_generator)
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.sequence_generator = sequence_generator
        self.x = x
        self.y = y
        self.x_mask = x_mask
        self.y_mask = y_mask
        self.f_input = f_input
        self.inputs = inputs

    def compile(self, state_names, glipmse_names, *args, **kwargs):
        """Compiles functions for beamsearch

        Parameters
        ----------
        inputs : dict
            Dictionary of named inputs

        """
        states = {name: kwargs[name] for name
                  in self.generator.state_names}
        contexts = {name: kwargs[name] for name
                    in self.generator.context_names}
        glimpses = {name: kwargs[name] for name
                    in self.generator.glimpse_names}
        input_dict = dict_union(states, contexts, glimpses)
        self.state_names = states.keys()
        self.context_names = contexts.keys()
        self.glimpse_names = glimpses.keys()

        init_state, init_out, init_costs, init_probs = \
            self.generator.generate(iterate=False, batch_size=self.batch_size,
                                    n_steps=1, contexts=contexts,
                                    output_probs=True,
                                    return_initial_states=True,
                                    return_dict=True)
        self.init_state_computer = function(self.inputs.values(),
                                            [init_state, init_out, init_costs,
                                             init_probs])

        self.outputs = tensor.TensorType('int64', (False,) *
                                     self.generator.get_dim("outputs"))()

        curr_out = tensor.zeros_like(init_out)
        curr_states = tensor.zeros_like(init_state)
        next_state, next_out, (next_costs, next_probs) = \
            self.generator.generate(outputs=curr_out[None, :],
                                    states=curr_states,
                                    iterate=False,
                                    batch_size=self.batch_size, n_steps=1,
                                    contexts=contexts, output_probs=True,
                                    return_dict=True)

        self.next_state_computer = function([curr_out, curr_states],
                                            [next_state, next_out, next_costs,
                                             next_probs])

        super(BeamSearch, self).compile(*args, **kwargs)

    @classmethod
    def _top_probs(cls, probs, beam_size, unique=False):
        """
        Returns indexes of elements with highest probabilities

        :param probs: a 3d array of probabilities (time, batch, readout_dim)
        :param beam_size: beam size, number of top probs to return
        :return: tuple of (indexes, top probabilities)
        """
        flatten = probs.flatten()
        if unique:
            args = np.unique(np.argpartition(-flatten, beam_size))[:beam_size]
        else:
            args = np.argpartition(-flatten, beam_size)[:beam_size]
        args = args[np.argsort(-flatten[args])]
        if unique:
            # append best if needed
            if args.shape[0] < beam_size:
                args = np.append(args,
                                 np.tile(args[0], beam_size - args.shape[0]))
        # convert args back
        indexes = np.unravel_index(args, probs.shape)
        return indexes, probs[indexes]

    def search(self, start_symbol, eol_symbol=-1, max_length=512,
               **kwargs):
        """Performs greedy search

        Parameters
        ----------
        eol_symbol: End of line symbol, the search stops when the
               symbol is generated
        max_length: Maximum sequence length, the search stops when it
               is reached

        Returns
        -------
        Most probable sequence, corresponding probabilities and costs

        .. note: Contexts are expected as keyword arguments

        """
        super(BeamSearch, self).search(**kwargs)
        context_vals = {name: val for name, val in kwargs.iteritems() if name
                        in self.context_names}

        n_chunks = list(context_vals.values())[0].shape[1] # input batch size
        real_batch_size = n_chunks * self.beam_size

        aux_inputs = {name: np.tile(val, self.beam_size) for name, val
                      in context_vals.iteritems()}
        aux_inp = np.tile(inp_seq, (1, self.beam_size, 1))
        aux_mask = np.tile(inp_mask, (1, self.beam_size))

        current_outputs = np.zeros((0, n_chunks, self.beam_size),
                                   dtype='int64')
        curr_out_mask = np.ones((0, n_chunks, self.beam_size), dtype=floatX)

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
