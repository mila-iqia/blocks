from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np

from theano import config
from theano import function
from theano import tensor

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

floatX = config.floatX


class Search(object):
    """Abstract search class

    Parameters
    ----------
    sequence_generator : sequence generator
        Sequence generator to be used

    """
    __metaclass__ = ABCMeta

    def __init__(self, sequence_generator):
        self.generator = sequence_generator
        self.compiled = False

    @abstractmethod
    def compile(self):
        self.compiled = True

    @abstractmethod
    def search(self, **kwargs):
        r"""Performs search

        Parameters
        ----------
        \*\*kwargs :
            Arguments needed by sequence generator

        Returns
        -------
        Generated sequences
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
    beam_size : int
        Size of beam
    batch_size : int
        Size of input batch
    sequence_generator : sequence generator
        Sequence generator brick

    """
    def __init__(self, beam_size, batch_size, sequence_generator, attended,
                 attended_mask, inputs_dict):
        super(BeamSearch, self).__init__(sequence_generator)
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.sequence_generator = sequence_generator
        self.attended = attended
        self.attended_mask = attended_mask
        self.inputs_dict = inputs_dict
        self.generate_names = sequence_generator.generate_outputs()
        self.init_computer = None
        self.next_computer = None
        self.real_batch_size = batch_size * beam_size

    def compile(self, *args, **kwargs):
        """Compiles functions for beamsearch

        Parameters
        ----------
        inputs : dict
            Dictionary of named inputs

        """
        generator = self.sequence_generator
        attended = self.attended
        attended_mask = self.attended_mask
        init_generated = generator.generate(attended=attended,
                                            attended_mask=attended_mask,
                                            iterate=False, n_steps=1,
                                            batch_size=self.real_batch_size)
        init_generated = OrderedDict(zip(self.generate_names, init_generated))
        init_cg = ComputationGraph(init_generated.values())
        init_readouts = VariableFilter(application=generator.readout.emit,
                                       name='readouts')(init_cg.variables)[-1]
        init_probs = generator.readout.emitter._probs(init_readouts)

        self.init_computer = function(self.inputs_dict.values(),
                                      init_generated.values() + [init_probs])

        cur_variables = OrderedDict()
        for name, value in init_generated.iteritems():
            cur_value = tensor.zeros_like(value)
            cur_value.name = name
            cur_variables[name] = cur_value

        input_names = (generator.state_names + generator.glimpse_names +
                       ['outputs'])
        generator_inputs = {name: val for name, val
                            in cur_variables.iteritems()
                            if name in input_names}
        next_generated = generator.generate(attended=attended,
                                            attended_mask=attended_mask,
                                            iterate=False,
                                            n_steps=1,
                                            batch_size=self.real_batch_size,
                                            **generator_inputs)
        next_generated = OrderedDict(zip(self.generate_names, next_generated))
        cg_step = ComputationGraph(next_generated.values())
        readouts_step = VariableFilter(application=generator.readout.emit,
                                       name='readouts')(cg_step.variables)[-1]
        next_probs = generator.readout.emitter._probs(readouts_step)
        self.next_computer = function(self.inputs_dict.values() +
                                      [cur_variables['states']],
                                      next_generated.values() + [next_probs])
        super(BeamSearch, self).compile(*args, **kwargs)

    def compute_initial(self, inputs_dict):
        inits = self.init_computer(inputs_dict.values())
        return OrderedDict(zip(self.generate_names + ['probs'], inits))

    def compute_next(self, inputs_dict, cur_vals):
        next_val = self.next_computer(inputs_dict.values() +
                                      [cur_vals['states']])
        return OrderedDict(zip(self.generate_names + ['probs'], next_val))

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
    def _rearrange(cls, outputs, indexes):
        n_chunks = indexes.shape[0]
        beam_size = indexes.shape[1]
        new_outputs = outputs.reshape((-1, n_chunks * beam_size))
        new_outputs = new_outputs[:, indexes.flatten()]
        new_outputs = new_outputs.reshape(outputs.shape)
        return new_outputs.copy()

    def search(self, inputs_val_dict, start_symbol, eol_symbol=-1,
               max_length=512, **kwargs):
        """Performs greedy search

        Parameters
        ----------
        eol_symbol : int
            End of line symbol, the search stops when the
            symbol is generated
        max_length : int
            Maximum sequence length, the search stops when it
            is reached

        Returns
        -------
        Most probable sequences, corresponding probabilities and costs

        """
        super(BeamSearch, self).search(**kwargs)
        # input batch size
        n_chunks = list(inputs_val_dict.values())[0].shape[1]

        aux_inputs = {name: np.tile(val, self.beam_size)
                      for name, val in inputs_val_dict.iteritems()}

        current_outputs = np.zeros((0, n_chunks, self.beam_size),
                                   dtype='int64')
        curr_out_mask = np.ones((0, n_chunks, self.beam_size), dtype=floatX)

        for i in xrange(max_length):
            if i == 0:
                cur_values = self.compute_initial(aux_inputs)
                next_probs = cur_values['probs']
            else:
                cur_values = self.compute_next(aux_inputs, cur_values)
                next_probs = cur_values['probs']

            # Choose top beam_size
            prob_batches = next_probs.reshape((n_chunks, self.beam_size, -1))
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
            for name in cur_values:
                cur_values[name] = self._rearrange(cur_values[name],
                                                   rearrange_ind)
            for name in aux_inputs:
                aux_inputs[name] = self._rearrange(aux_inputs[name],
                                                   rearrange_ind)

            # construct next output
            outputs = outputs.reshape((1, n_chunks, self.beam_size))
            current_outputs = np.append(current_outputs,
                                        outputs.copy(), axis=0)
            # check if we meet eol
            next_out_mask = np.ones((1, n_chunks, self.beam_size),
                                    dtype=floatX)

            next_out_mask[0, :, :] = (outputs[0, :, :] != eol_symbol)
            curr_out_mask = np.append(curr_out_mask, next_out_mask.copy(),
                                      axis=0)

            if np.all(current_outputs[-1, :, 0] == eol_symbol):
                break

        # Select only best
        current_outputs = current_outputs[:, :, 0]
        curr_out_mask = curr_out_mask[:, :, 0]

        return current_outputs, curr_out_mask
