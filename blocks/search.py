from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from six.moves import range

import numpy

from theano import config
from theano import function
from theano import tensor

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

floatX = config.floatX


def unchunk_rename(*args, **kwargs):
    def _trace(func):
        def wrapper(self, *inputs):
            merged_input = []
            for input in inputs:
                merged_input += [OrderedDict([(name, self.merge_chunks(value))
                                              for name, value
                                              in input.iteritems()])]
            names, outputs = func(self, *merged_input)
            if add_time_dim:
                outputs = [self.divide_by_chunks(value.reshape((1,) +
                                                               value.shape))
                           for value in outputs]
            else:
                outputs = [self.divide_by_chunks(value.reshape(value.shape))
                           for value in outputs]
            return OrderedDict(zip(names, outputs))
        return wrapper
    if len(args) == 1 and callable(args[0]):
        # No arguments, this is the decorator
        # Set default values for the arguments
        add_time_dim = True
        return _trace(args[0])
    else:
        # This is just returning the decorator
        add_time_dim = kwargs['add_time_dim']
        return _trace


class BeamSearch(object):
    """Beam search.

    Parameters
    ----------
    beam_size : int
        Size of beam.
    batch_size : int
        Size of input batch.
    sequence_generator : sequence generator
        Sequence generator brick.
    attended : theano variable
        Theano variable for attended in sequence generator.
    attended_mask : theano variable
        Theano variable for attended mask in sequence generator.
    inputs_dict : dict
        Dictionary of inputs {name: theano variable for input}. The
        functions will be constructed with these inputs.

    """
    def __init__(self, beam_size, batch_size, sequence_generator, attended,
                 attended_mask, inputs_dict):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.sequence_generator = sequence_generator
        self.attended = attended
        self.attended_mask = attended_mask
        self.inputs_dict = inputs_dict
        self.generate_names = sequence_generator.generate_outputs()
        self.init_computer = None
        self.next_computer = None
        self.attended_computer = None
        self.initial_state_computer = None
        self.real_batch_size = batch_size * beam_size
        self.state_names = (sequence_generator.state_names +
                            sequence_generator.glimpse_names +
                            ['outputs'])

    def compile(self, *args, **kwargs):
        """Compiles functions for beam search."""
        generator = self.sequence_generator
        attended = self.attended
        attended_mask = self.attended_mask

        self.attended_computer = function(self.inputs_dict.values(),
                                          [attended, attended_mask],
                                          on_unused_input='ignore')

        initial_states = OrderedDict()
        for name in self.state_names:
            initial_states[name] = generator.initial_state(
                name,
                self.real_batch_size,
                attended=attended)

        self.initial_state_computer = function([attended, attended_mask],
                                               initial_states.values(),
                                               on_unused_input='ignore')
        # Construct initial values
        init_generated = generator.generate(attended=attended,
                                            attended_mask=attended_mask,
                                            iterate=False, n_steps=1,
                                            batch_size=self.real_batch_size,
                                            return_dict=True)
        init_cg = ComputationGraph(init_generated.values())
        init_probs = VariableFilter(
            application=generator.readout.emitter.probs,
            name='output')(init_cg.variables)[-1]

        # Create theano function for initial values
        self.init_computer = function([attended, attended_mask],
                                      init_generated.values() + [init_probs],
                                      on_unused_input='ignore')

        # Define inputs for next values computer
        cur_variables = OrderedDict()
        for name, value in init_generated.iteritems():
            cur_value = tensor.zeros_like(value)
            cur_value.name = name
            cur_variables[name] = cur_value

        generator_inputs = {name: val for name, val
                            in cur_variables.iteritems()
                            if name in self.state_names}
        next_generated = generator.generate(attended=attended,
                                            attended_mask=attended_mask,
                                            iterate=False,
                                            n_steps=1,
                                            batch_size=self.real_batch_size,
                                            return_dict=True,
                                            **generator_inputs)
        cg_step = ComputationGraph(next_generated.values())
        next_probs = VariableFilter(
            application=generator.readout.emitter.probs,
            name='output')(cg_step.variables)[-1]
        # Create theano function for next values
        self.next_computer = function([attended, attended_mask,
                                       cur_variables['states']],
                                      next_generated.values() + [next_probs])
        self.compiled = True

    @unchunk_rename(add_time_dim=False)
    def compute_contexts(self, inputs_dict):
        contexts = self.attended_computer(*inputs_dict.values())
        return ["attended", "attended_mask"], contexts

    @unchunk_rename
    def compute_initial_states(self, contexts):
        """Computes initial outputs and states."""
        init_states = self.initial_state_computer(*contexts.values())
        return self.state_names, init_states

    @unchunk_rename
    def compute_next(self, contexts, cur_vals):
        """Computes next states, glimpses, outputs, and probabilities.

        Parameters
        ----------
        contexts : dict
            Dictionary of contexts divided by chunks.
        cur_vals : dict
            Dictionary of current states, glimpses, and outputs.

        Returns
        -------
        Dictionary of next state, glimpses, output, probabilities values
        with names as returned by `generate_outputs`.

        """
        next_values = self.next_computer(*(contexts.values() +
                                           [cur_vals['states'][0]]))
        return self.generate_names + ['probs'], next_values

    @classmethod
    def _top_probs(cls, probs, beam_size, unique=False):
        """Returns indexes of elements with highest probabilities.

        Parameters
        ----------
        probs : numpy array
            A 3d array of probabilities (length of sequence, batch,
            readout_dim)
        beam_size : int
            Beam size, number of top probs to return

        Returns
        -------
        Tuple of (indexes, top probabilities)

        """
        flatten = probs.flatten()
        if unique:
            args = numpy.unique(
                numpy.argpartition(-flatten, beam_size))[:beam_size]
        else:
            args = numpy.argpartition(-flatten, beam_size)[:beam_size]
        args = args[numpy.argsort(-flatten[args])]
        if unique:
            # append best if needed
            if args.shape[0] < beam_size:
                args = numpy.append(
                    args,
                    numpy.tile(args[0], beam_size - args.shape[0]))
        # convert args back
        indexes = numpy.unravel_index(args, probs.shape[1:])
        return indexes, probs[0][indexes]

    def _rearrange(self, outputs, indexes):
        new_outputs = self.merge_chunks(outputs)
        new_outputs = new_outputs[:, indexes.T.flatten()]
        new_outputs = self.divide_by_chunks(new_outputs)
        return new_outputs.copy()

    def merge_chunks(self, array):
        """Merges chunks.

        Parameters
        ----------
        array : numpy array
            3D or 4D (sequence length, beam size, batch size
            [, readout dim]) array

        Returns
        -------
        2D or 3D (sequence length, beam size * batch size [, readout_dim])
        array

        """
        if len(array.shape) == 3:
            trans = (1, 2, 0)
            trans_back = (1, 0)
            out_shape = (self.beam_size * self.batch_size, array.shape[0])
        else:
            trans = (1, 2, 0, 3)
            trans_back = (1, 0, 2)
            out_shape = (self.beam_size * self.batch_size, array.shape[0],
                         array.shape[-1])
        array = array.transpose(trans)
        array = array.reshape(out_shape)
        return array.transpose(trans_back)

    def divide_by_chunks(self, array):
        """Divides input to chunks.

        Parameters
        ----------
        array : numpy array
            2D or 3D (sequence length, beam size * batch size
            [, readout dim]) array

        Returns
        -------
        3D or 4D (sequence length, beam size, batch size [, readout dim])
        array

        """
        if len(array.shape) == 2:
            first_transpose = (1, 0)
            reshape = (self.beam_size, self.batch_size, array.shape[0])
            back_transpose = (2, 0, 1)
        else:
            first_transpose = (1, 0, 2)
            reshape = (self.beam_size, self.batch_size, array.shape[0],
                       array.shape[2])
            back_transpose = (2, 0, 1, 3)
        reshaped = array.transpose(first_transpose)
        reshaped = reshaped.reshape(reshape)
        return reshaped.transpose(back_transpose)

    def search(self, inputs_val_dict, eol_symbol=-1, max_length=512, **kwargs):
        """Performs beam search.

        Parameters
        ----------
        inputs_val_dict : dict
            Dictionary of input values {name: value}. Input values may
            be in a batch of size greater than 1, in this case, several
            beam searches will be performed in parallel.
        eol_symbol : int
            End of sequence symbol, the search stops when the
            symbol is generated.
        max_length : int
            Maximum sequence length, the search stops when it
            is reached.

        Returns
        -------
        Most probable sequences, corresponding probabilities and costs.

        """
        if not self.compiled:
            self.compile()
        # Inputs repeated beam_size times
        aux_inputs = OrderedDict(
            [(name,
              self.divide_by_chunks(numpy.tile(val, (1, self.beam_size))))
             for name, val in inputs_val_dict.iteritems()])

        # Precompute contexts
        contexts = self.compute_contexts(aux_inputs)

        cur_states = self.compute_initial_states(contexts)

        cur_states['cur_outputs'] = cur_states['outputs']
        cur_states['cur_outputs_mask'] = numpy.ones_like(
            cur_states['cur_outputs'])
        cur_states['cur_probs'] = numpy.ones_like(
            cur_states['cur_outputs'])

        for i in range(max_length):
            cur_states.update(self.compute_next(contexts, cur_states))
            next_probs = (cur_states['cur_probs'][:, :, :, None] *
                          cur_states['probs'] *
                          cur_states['cur_outputs_mask'][-1, :, :, None])

            # Top probs
            indexes, top_probs = zip(*[self._top_probs(next_probs[:, :, j],
                                                       self.beam_size,
                                                       unique=i == 0)
                                       for j in range(self.batch_size)])
            cur_states['cur_probs'] = numpy.array(top_probs).T[None, :, :]
            indexes = numpy.array(indexes)  # chunk, 2, beam
            # current_outputs.
            # here we suppose, that we have 2d outputs
            outputs = indexes[:, 1, :].copy()

            # rearrange outputs
            rearrange_ind = indexes[:, 0, :]
            for name in cur_states:
                cur_states[name] = self._rearrange(cur_states[name],
                                                   rearrange_ind)
            for name in contexts:
                contexts[name] = self._rearrange(contexts[name],
                                                 rearrange_ind)

            # construct next output
            outputs = outputs.T[None, :, :]
            cur_states['cur_outputs'] = numpy.append(cur_states['cur_outputs'],
                                                     outputs.copy(), axis=0)
            # check if we meet eol
            next_out_mask = numpy.ones((1, self.beam_size, self.batch_size),
                                       dtype=floatX)

            next_out_mask[0, :, :] = ((outputs[0, :, :] != eol_symbol) *
                                      cur_states['cur_outputs_mask'][-1, :, :])
            cur_states['cur_outputs_mask'] = numpy.append(
                cur_states['cur_outputs_mask'],
                next_out_mask.copy(),
                axis=0)

            if numpy.all(cur_states['cur_outputs'][-1, :, 0] == eol_symbol):
                break

        # Select only best
        best_outputs = cur_states['cur_outputs'][1:, 0, :]
        best_out_mask = cur_states['cur_outputs_mask'][1:, 0, :]
        best_probs = cur_states['cur_probs'][0, 0, :]

        return best_outputs, best_out_mask, best_probs
