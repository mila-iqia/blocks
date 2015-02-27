from collections import OrderedDict
from six.moves import range

import numpy

from theano import config
from theano import function

from blocks.filter import VariableFilter, get_application_call
from blocks.graph import ComputationGraph
from blocks.roles import INPUT, OUTPUT

floatX = config.floatX


class BeamSearch(object):
    """Beam search.

    Parameters
    ----------
    beam_size : int
        Size of beam.
    sequence_generator : sequence generator
        Sequence generator brick.
    comp_graph : :class:`ComputationalGraph`
        Computational graph which contains `sequence_generator`.

    """
    def __init__(self, beam_size, sequence_generator, comp_graph):
        self.beam_size = beam_size
        self.sequence_generator = sequence_generator
        self.generate_names = sequence_generator.generate.states
        self.init_computer = None
        self.next_computer = None
        self.attended_computer = None
        self.initial_state_computer = None
        self.state_names = (sequence_generator.state_names +
                            sequence_generator.glimpse_names +
                            ['outputs'])
        self.context_names = sequence_generator.context_names
        self.comp_graph = comp_graph
        self.inputs = OrderedDict(comp_graph.dict_of_inputs())
        self.need_input_states = []
        self.compiled = False

    def compile_attended_computer(self, generator, inner_cg):
        contexts_original = OrderedDict()
        for name in generator.generate.contexts:
            contexts_original[name] = VariableFilter(
                bricks=[generator],
                name='^' + name + '$',
                roles=[INPUT])(inner_cg)[0]

        self.attended_computer = function(self.inputs.values(),
                                          contexts_original.values(),
                                          on_unused_input='ignore')

    def compile_initial_state_computer(self, generator, contexts):
        initial_states = []
        for name in self.state_names:
            initial_states.append(generator.initial_state(
                name,
                self.beam_size,
                **contexts))
        self.initial_state_computer = function(contexts.values(),
                                               initial_states,
                                               on_unused_input='ignore')

    def compile_next_state_computer(self, generator, contexts, inner_cg):
        states = []
        for name in generator.generate.states:
            var = VariableFilter(bricks=[generator], name='^' + name + '$',
                                 roles=[INPUT])(inner_cg)[-1:]
            if var:
                self.need_input_states.append(name)
            states.extend(var)

        next_states = []
        for name in generator.generate.states:
            var = VariableFilter(bricks=[generator], name='^' + name + '$',
                                 roles=[OUTPUT])(inner_cg)[-1:]
            next_states.extend(var)

        next_probs = VariableFilter(
            bricks=[generator.readout.emitter],
            name='^probs')(inner_cg)[-1]
        # Create theano function for next values
        self.next_computer = function(contexts.values() + states,
                                      next_states + [next_probs])

    def compile(self, *args, **kwargs):
        """Compiles functions for beam search."""
        generator = self.sequence_generator

        application_call = get_application_call(self.comp_graph.outputs[0])
        inner_cg = ComputationGraph(application_call.inner_outputs)
        contexts = OrderedDict()
        for name in generator.generate.contexts:
            contexts[name] = VariableFilter(bricks=[generator],
                                            name='^' + name + '$',
                                            roles=[INPUT])(inner_cg)[0]
        self.compile_attended_computer(generator, inner_cg)
        self.compile_initial_state_computer(generator, contexts)
        self.compile_next_state_computer(generator, contexts, inner_cg)

        self.compiled = True

    def compute_contexts(self, inputs_dict):
        """Computes contexts from inputs."""
        contexts = self.attended_computer(*[inputs_dict[name]
                                            for name in self.inputs])
        return OrderedDict(zip(["attended", "attended_mask"], contexts))

    def compute_initial_states(self, contexts):
        """Computes initial outputs and states."""
        init_states = self.initial_state_computer(*contexts.values())
        init_states = [state.reshape((1,) + state.shape)
                       for state in init_states]
        return OrderedDict(zip(self.state_names, init_states))

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
        # First timesteps only if state is needed
        states = [cur_vals[name][0] for name in self.need_input_states]

        next_values = self.next_computer(*(contexts.values() + states))

        # Add time dimension back
        next_values = [state.reshape((1,) + state.shape)
                       for state in next_values]
        return OrderedDict(zip(self.generate_names + ['probs'], next_values))

    @classmethod
    def _top_probs(cls, probs, beam_size):
        """Returns indexes of elements with highest probabilities.

        Parameters
        ----------
        probs : numpy array
            A 3d array of probabilities (length of sequence, batch,
            readout_dim).
        beam_size : int
            Beam size, number of top probs to return.

        Returns
        -------
        Tuple of (indexes, top probabilities).

        """
        flatten = probs.flatten()
        args = numpy.argpartition(-flatten, beam_size)[:beam_size]
        args = args[numpy.argsort(-flatten[args])]
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
            Dictionary of input values {name: value}. Batch size `1` is
            supported only.
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
        inputs = OrderedDict([(name, numpy.tile(val, (1, self.beam_size)))
                              for name, val in inputs_val_dict.iteritems()])

        # Precompute contexts
        contexts = self.compute_contexts(inputs)

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
            indexes, top_probs = self._top_probs(next_probs, self.beam_size)
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
        outputs = cur_states['cur_outputs'][1:, :]
        outputs_mask = cur_states['cur_outputs_mask'][1:, :]
        probs = cur_states['cur_probs'][0, :]

        return outputs, outputs_mask, probs
