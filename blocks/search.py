from collections import OrderedDict
from six.moves import range

import numpy

from theano import config
from theano import function
from theano import tensor

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.utils import unpack

floatX = config.floatX


def construct_dict(func):
    def wrapper(self, *inputs):
        names, outputs = func(self, *inputs)
        return OrderedDict(zip(names, outputs))
    return wrapper


class BeamSearch(object):
    """Beam search.

    Parameters
    ----------
    beam_size : int
        Size of beam.
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
    def __init__(self, beam_size, sequence_generator, inputs_dict, comp_graph):
        self.beam_size = beam_size
        self.sequence_generator = sequence_generator
        self.inputs_dict = inputs_dict
        self.generate_names = sequence_generator.generate.outputs
        self.init_computer = None
        self.next_computer = None
        self.attended_computer = None
        self.initial_state_computer = None
        self.state_names = (sequence_generator.state_names +
                            sequence_generator.glimpse_names +
                            ['outputs'])
        self.comp_graph = comp_graph
        self.compiled = False

    def compile(self, *args, **kwargs):
        """Compiles functions for beam search."""
        generator = self.sequence_generator

        attended = unpack(
            VariableFilter(application=generator.generate,
                           name='attended$')(self.comp_graph.variables))
        attended_mask = unpack(
            VariableFilter(application=generator.generate,
                           name='attended_mask$')(self.comp_graph.variables))

        self.attended_computer = function(self.inputs_dict.values(),
                                          [attended, attended_mask],
                                          on_unused_input='ignore')

        initial_states = OrderedDict()
        for name in self.state_names:
            initial_states[name] = generator.initial_state(
                name,
                self.beam_size,
                attended=attended)

        self.initial_state_computer = function([attended, attended_mask],
                                               initial_states.values(),
                                               on_unused_input='ignore')

        # Define inputs for next values computer
        cur_variables = OrderedDict()
        for name, value in initial_states.iteritems():
            cur_value = tensor.zeros_like(value)
            cur_value.name = name
            cur_variables[name] = cur_value

        next_generated = generator.generate(attended=attended,
                                            attended_mask=attended_mask,
                                            iterate=False,
                                            n_steps=1,
                                            batch_size=self.beam_size,
                                            return_dict=True,
                                            **cur_variables)
        cg_step = ComputationGraph(next_generated.values())
        next_probs = VariableFilter(
            application=generator.readout.emitter.probs,
            name='output')(cg_step.variables)[-1]
        # Create theano function for next values
        self.next_computer = function([attended, attended_mask,
                                       cur_variables['states']],
                                      next_generated.values() + [next_probs])
        self.compiled = True

    @construct_dict
    def compute_contexts(self, inputs_dict):
        contexts = self.attended_computer(*inputs_dict.values())
        return ["attended", "attended_mask"], contexts

    @construct_dict
    def compute_initial_states(self, contexts):
        """Computes initial outputs and states."""
        init_states = self.initial_state_computer(*contexts.values())
        init_states = [state.reshape((1,) + state.shape)
                       for state in init_states]
        return self.state_names, init_states

    @construct_dict
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
        next_values = [state.reshape((1,) + state.shape)
                       for state in next_values]
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

    @staticmethod
    def _rearrange(outputs, indexes):
        new_outputs = outputs[:, indexes.T.flatten()]
        return new_outputs.copy()

    def search(self, inputs_val_dict, eol_symbol=-1, max_length=512):
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
            [(name, numpy.tile(val, (1, self.beam_size)))
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
            next_probs = (cur_states['cur_probs'][:, :, None] *
                          cur_states['probs'] *
                          cur_states['cur_outputs_mask'][-1, :, None])

            # Top probs
            indexes, top_probs = self._top_probs(next_probs[:, :],
                                                 self.beam_size,
                                                 unique=i == 0)
            cur_states['cur_probs'] = numpy.array(top_probs).T[None, :]
            indexes = numpy.array(indexes)  # chunk, 2, beam
            # current_outputs.
            # here we suppose, that we have 2d outputs
            outputs = indexes[1, :].copy()

            # rearrange outputs
            rearrange_ind = indexes[0, :]
            for name in cur_states:
                cur_states[name] = self._rearrange(cur_states[name],
                                                   rearrange_ind)
            for name in contexts:
                contexts[name] = self._rearrange(contexts[name],
                                                 rearrange_ind)

            # construct next output
            outputs = outputs[None, :]
            cur_states['cur_outputs'] = numpy.append(cur_states['cur_outputs'],
                                                     outputs.copy(), axis=0)
            # check if we meet eol
            next_out_mask = numpy.ones((1, self.beam_size),
                                       dtype=floatX)

            next_out_mask[0, :] = ((outputs[0, :] != eol_symbol) *
                                   cur_states['cur_outputs_mask'][-1, :])
            cur_states['cur_outputs_mask'] = numpy.append(
                cur_states['cur_outputs_mask'],
                next_out_mask.copy(),
                axis=0)

            # All first element in sequences ended
            if numpy.all(cur_states['cur_outputs'][-1, 0] == eol_symbol):
                break

        # Select only best
        best_outputs = cur_states['cur_outputs'][1:, 0]
        best_out_mask = cur_states['cur_outputs_mask'][1:, 0]
        best_probs = cur_states['cur_probs'][0, 0]

        return best_outputs, best_out_mask, best_probs
