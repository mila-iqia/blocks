from blocks.utils import dict_union

from abc import ABCMeta, abstractmethod

import numpy as np

from theano import function
from theano import config
from theano import tensor

from blocks.bricks.sequence_generators import BaseSequenceGenerator

floatX = config.floatX


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

    def compile(self, *args, **kwargs):
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

        next_glimpses = self.generator.transition.take_look(
            return_dict=True, **input_dict)

        self.next_glimpse_computer = function(self.inputs.values(),
                                              next_glimpses.values())

        self.outputs = tensor.TensorType('int64', (False,) *
                                     self.generator.get_dim("outputs"))()
        next_readouts = self.generator.readout.readout(
            feedback=self.readout.feedback(self.outputs),
            **input_dict)

        readout_inputs = (self.outputs + states.values() + glimpses.values() +
                          contexts.values())
        self.next_readouts_computer = function(readout_inputs, next_readouts)

        next_outputs, next_states, next_costs = \
            self.generator.compute_next_states(next_readouts,
                                               next_glimpses,
                                               **kwargs)

        states_inputs = contexts.values() + next_readouts + states.values()
        self.next_states_computers = [function(states_inputs, var) for var
                                      in next_states]
        self.next_outputs_computer = function(states_inputs, next_outputs)
        self.next_costs_computer = function(states_inputs, next_costs)

        next_probs = self.generator.readout.emit_probs(next_readouts)
        self.next_probs_computer = function(next_readouts, next_probs)

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

    @classmethod
    def _tile(cls, val, times):
        if val.ndim == 2:
            return np.tile(val, (1, times))
        else:
            return np.tile(val, (1, times, 1))

    @classmethod
    def _rearrange(cls, outputs, indexes):
        n_chunks = indexes.shape[0]
        beam_size = indexes.shape[1]
        new_outputs = outputs.reshape((-1, n_chunks * beam_size))
        new_outputs = new_outputs[:, indexes.flatten()]
        new_outputs = new_outputs.reshape(outputs.shape)
        return new_outputs.copy()

    def search(self, eol_symbol=-1, max_length=512, **kwargs):
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

        aux_inputs = {name: self._tile(val, self.beam_size) for name, val
                      in context_vals.iteritems()}

        current_outputs = np.zeros((0, n_chunks, self.beam_size),
                                   dtype='int64')
        curr_out_mask = np.ones((0, n_chunks, self.beam_size), dtype=floatX)

        #outputs_dim = self.generator.get_dim('outputs')
        #current_outputs = np.array(outputs_dim)
        for i in xrange(max_length):
            # Compute probabilities
            next_glimpses = self.next_glimpse_computer(input_vals.values())
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
            probs_val = self.next_probs_computer(next_readouts)

            # Choose top beam_size
            prob_batches = probs_val.reshape((n_chunks, self.beam_size, -1))
            # Top probs
            indexes, top_probs = zip(*[self._top_probs(batch, self.beam_size,
                                                       unique=i == 0)
                                       for batch in prob_batches])
            indexes = np.array(indexes)  # chunk, 2, beam
            # current_outputs.
            # here we suppose, that we have 2d outputs
            outputs = indexes[:, 1, :].copy()

            # rearrange outputs
            rearrange_ind = indexes[:, 0, :]
            current_outputs = self._rearrange(current_outputs, rearrange_ind)
            curr_out_mask = self._rearrange(curr_out_mask, rearrange_ind)
            hidden_states = [self._rearrange(s, rearrange_ind) for s in
                             hidden_states]
            probs_val = self._rearrange(probs_val, rearrange_ind)

            # construct next output
            #next_outputs = np.array(outputs).flatten()[None, :]
            outputs = outputs.reshape((1, n_chunks, self.beam_size))
            current_outputs = np.append(current_outputs,
                                        outputs.copy(), axis=0)
            # check if we meet eol
            next_out_mask = np.ones((1, n_chunks, self.beam_size),
                                    dtype=floatX)

            # Stop computing branch which met eol
            #if curr_out_mask.shape[0] >= 1:
            #    next_out_mask[0, :, :] = np.logical_and((outputs[0, :, :] != eol_symbol), curr_out_mask[-1, :, :] == 1)
            #else:
            next_out_mask[0, :, :] = (outputs[0, :, :] != eol_symbol)
            curr_out_mask = np.append(curr_out_mask, next_out_mask.copy(), axis=0)

            if np.all(current_outputs[-1, :, 0] == eol_symbol):
                break

        # Select only best
        current_outputs = current_outputs[:, :, 0]
        curr_out_mask = curr_out_mask[:, :, 0]

        return current_outputs, curr_out_mask

