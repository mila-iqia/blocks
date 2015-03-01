from collections import OrderedDict
from six.moves import range

import numpy

from theano import config, function, tensor

from blocks.bricks.sequence_generators import SequenceGenerator
from blocks.filter import VariableFilter, get_application_call, get_brick
from blocks.graph import ComputationGraph
from blocks.roles import INPUT, OUTPUT

floatX = config.floatX


class BeamSearch(object):
    """Beam search.

    N-greedy algorithm to find the most probable sequence.

    Parameters
    ----------
    beam_size : int
        The beam size.
    samples : :class:`~theano.Variable`
        An output of a sampling computation graph built by
        :meth:`~blocks.brick.SequenceGenerator.generate`, the one
        corresponding to sampled sequences.

    See Also
    --------
    :class:`SequenceGenerator`, :class:`SequenceContentAttention`.

    Notes
    -----
    Sequence generator should use an emitter which has `probs` method and
    one of its outputs is called `probs` e.g. :class:`SoftmaxEmitter`.

    """
    def __init__(self, beam_size, samples):
        self.beam_size = beam_size

        # Extracting information from the sampling computation graph
        cg = ComputationGraph(samples)
        self.inputs = OrderedDict(cg.dict_of_inputs())
        self.generator = get_brick(samples)
        if not isinstance(self.generator, SequenceGenerator):
            raise ValueError
        self.generate_call = get_application_call(samples)
        if (not self.generate_call.application ==
                self.generator.generate):
            raise ValueError
        self.inner_cg = ComputationGraph(self.generate_call.inner_outputs)

        # Fetching names from the sequence generator
        self.context_names = self.generator.generate.contexts
        self.state_names = self.generator.generate.states

        # Parsing the inner computation graph of sampling scan
        self.contexts = [
            VariableFilter(bricks=[self.generator], name='^' + name + '$',
                                   roles=[INPUT])(self.inner_cg)[0]
             for name in self.context_names]
        self.input_states = []
        # Includes only those state names that were actually used
        # in 'generate'
        self.input_state_names = []
        for name in self.generator.generate.states:
            var = VariableFilter(bricks=[self.generator], name='^' + name + '$',
                                 roles=[INPUT])(self.inner_cg)
            if var:
                self.input_state_names.append(name)
                self.input_states.append(var[0])

        self.compiled = False

    def _compile_context_computer(self):
        self.context_computer = function(
            list(self.inputs.values()), self.contexts,
            on_unused_input='ignore')

    def _compile_initial_state_computer(self):
        initial_states = [
            self.generator.initial_state(
                name, self.beam_size,
                **dict(zip(self.context_names, self.contexts)))
            for name in self.state_names]
        self.initial_state_computer = function(
            self.contexts, initial_states, on_unused_input='ignore')

    def _compile_next_state_computer(self):
        next_states = [VariableFilter(bricks=[self.generator],
                                      name='^' + name + '$',
                                      roles=[OUTPUT])(self.inner_cg)[-1]
                       for name in self.state_names]
        next_outputs = VariableFilter(
            application=self.generator.readout.emit, roles=[OUTPUT])(
                self.inner_cg.variables)
        self.next_state_computer = function(
            self.contexts + self.input_states + next_outputs, next_states)

    def _compile_costs_computer(self):
        next_probs = VariableFilter(
            bricks=[self.generator.readout.emitter],
            name='^probs$')(self.inner_cg)[-1]
        logprobs = -tensor.log(next_probs)
        self.costs_computer = function(
            self.contexts + self.input_states, logprobs, on_unused_input='ignore')

    def compile(self):
        self._compile_context_computer()
        self._compile_initial_state_computer()
        self._compile_next_state_computer()
        self._compile_costs_computer()
        self.compiled = True

    def compute_contexts(self, inputs_dict):
        """Computes contexts from inputs.

        Parameters
        ----------
        inputs_dict : dict
            Dictionary of input arrays.

        """
        contexts = self.context_computer(*[inputs_dict[name]
                                           for name in self.inputs])
        return OrderedDict(zip(self.context_names, contexts))

    def compute_initial_states(self, contexts):
        """Computes initial outputs and states.

        Parameters
        ----------
        contexts : dict
            Dictionary of contexts {name: context}.

        """
        init_states = self.initial_state_computer(*list(contexts.values()))
        return OrderedDict(zip(self.state_names, init_states))

    def compute_costs(self, contexts, cur_states):
        states = [cur_states[name] for name in self.input_state_names]
        return self.costs_computer(*(list(contexts.values()) + states))

    def compute_next_state(self, contexts, cur_states, outputs):
        """Computes next states.

        Parameters
        ----------
        contexts : dict
            Dictionary of contexts.
        cur_states : dict
            Dictionary of current states.

        Returns
        -------
        Dictionary of next state and probabilities values
        with names as returned by `generate_outputs`.

        """
        # First timesteps only if state is needed
        states = [cur_states[name] for name in self.input_state_names]
        next_values = self.next_state_computer(*(list(contexts.values()) +
                                                 states + [outputs]))
        return OrderedDict(zip(self.state_names, next_values))

    @staticmethod
    def _top_probs(scores, beam_size, unique=False):
        """Returns indexes of elements with lowest scores.

        Parameters
        ----------
        scores : numpy array
            A 2d array of scores (batch, readout_dim).
        beam_size : int
            Beam size, number of top scores to return.
        unique : bool
            Return only unique indexes. Should be used for the first
            iteration of the beam search.

        Returns
        -------
        Tuple of (indexes, top scores).

        """
        if unique:
            flatten = scores[:1, :].flatten()
        else:
            flatten = scores.flatten()
        args = numpy.argpartition(flatten, beam_size)[:beam_size]
        args = args[numpy.argsort(flatten[args])]
        # convert args back
        indexes = numpy.unravel_index(args, scores.shape)
        return indexes, scores[indexes]

    def _rearrange(self, states, indexes):
        for name in states:
            if name.startswith('cur'):
                states[name] = states[name][:, indexes.T.flatten()]
            else:
                states[name] = states[name][indexes.T.flatten()]

    def search(self, inputs_val_dict, eol_symbol=-1, max_length=512):
        """Performs beam search.

        If the beam search was not compiled, it also compiles it.

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
        Sequences in the beam, masks, and corresponding probabilities.

        """
        if not self.compiled:
            self.compile()

        # Inputs repeated beam_size times
        inputs = OrderedDict([(name, numpy.tile(val, (1, self.beam_size)))
                              for name, val in inputs_val_dict.items()])

        # Precompute contexts
        contexts = self.compute_contexts(inputs)

        states = self.compute_initial_states(contexts)

        states['cur_outputs'] = states['outputs'][None, :]
        states['cur_outputs_mask'] = numpy.ones_like(states['cur_outputs'])
        states['cur_logprobs'] = numpy.zeros_like(states['cur_outputs'])

        for i in range(max_length):
            if numpy.all(states['cur_outputs_mask'][-1, :] == 0):
                break

            logprobs = self.compute_costs(contexts, states)
            next_probs = (states['cur_logprobs'][-1, :, None]
                    + logprobs * states['cur_outputs_mask'][-1, :, None])
            (finished,) = numpy.where(states['cur_outputs_mask'][-1, :] == 0)
            next_probs[finished, :eol_symbol] = numpy.inf
            next_probs[finished, eol_symbol + 1:] = numpy.inf

            (indexes, outputs), top_probs = self._top_probs(next_probs, self.beam_size,
                                                 unique=i == 0)
            self._rearrange(states, indexes)

            states.update(self.compute_next_state(contexts, states, outputs))
            states['cur_logprobs'] = numpy.array(top_probs).T[None, :]
            states['cur_outputs'] = numpy.append(
                states['cur_outputs'], outputs[None, :], axis=0)
            next_out_mask = ((outputs != eol_symbol) *
                              states['cur_outputs_mask'][-1, :])
            states['cur_outputs_mask'] = numpy.append(
                states['cur_outputs_mask'], next_out_mask[None, :], axis=0)

        # Drop a meaningless first element
        outputs = states['cur_outputs'][1:, :]
        outputs_mask = states['cur_outputs_mask'][1:, :]
        logprobs = states['cur_logprobs'][-1, :]

        return outputs, outputs_mask, logprobs
