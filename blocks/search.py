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
    """Approximate search for the most likely sequence.

    Beam search is an approximate algorithm for finding :math:`y^* =
    argmax_y P(y|c)`, where :math:`y` is an output sequence, :math:`c` are
    the contexts, :math:`P` is the output distribution of a
    :class:`.SequenceGenerator`. At each step it considers :math:`k`
    candidate sequence prefixes. :math:`k` is called the beam size, and the
    sequence are called the beam. The sequences are replaced with their
    :math:`k` most probable continuations, and this is repeated until
    end-of-line symbol is met.

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
    :class:`.SequenceGenerator`

    Notes
    -----
    Sequence generator should use an emitter which has `probs` method
    e.g. :class:`SoftmaxEmitter`.

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
            application=self.generator.readout.emitter.probs,
            roles=[OUTPUT])(self.inner_cg)[-1]
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
    def _smallest(matrix, k, only_first_row=False):
        """Find k smallest elements of a matrix.

        Parameters
        ----------
        matrix : numpy array
            The matrix.
        k : int
            The number of smallest elements required.
        only_first_row : bool, optional
            Consider only elements of the first row.

        Returns
        -------
        Tuple of ((row numbers, column numbers), values).

        """
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = numpy.argpartition(flatten, k)[:k]
        args = args[numpy.argsort(flatten[args])]
        return numpy.unravel_index(args, matrix.shape), flatten[args]

    def search(self, input_values, eol_symbol, max_length):
        """Performs beam search.

        If the beam search was not compiled, it also compiles it.

        Parameters
        ----------
        input_values : dict
            Dictionary of input values {name: value}. The shapes should be
            the same as if you ran sampling with batch size equal to
            `beam_size`. Put it differently, the user is responsible
            for duplicaling inputs necessary number of times, because
            this class has insufficient information to do it properly.
        eol_symbol : int
            End of sequence symbol, the search stops when the symbol is
            generated.
        max_length : int
            Maximum sequence length, the search stops when it is reached.

        Returns
        -------
        Sequences in the beam, masks, and corresponding log-probabilities.

        """
        if not self.compiled:
            self.compile()

        contexts = self.compute_contexts(input_values)
        states = self.compute_initial_states(contexts)

        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = states['outputs'][None, :]
        mask = numpy.ones_like(all_outputs[0])
        costs = numpy.zeros_like(all_outputs[0])

        for i in range(max_length):
            if mask.sum() == 0:
                break

            # We carefully hack values of the `logprobs` array to ensure
            # that all finished sequences are continued with `eos_symbol`.
            logprobs = self.compute_costs(contexts, states)
            next_costs = costs[:, None] + logprobs * mask[:, None]
            (finished,) = numpy.where(mask == 0)
            next_costs[finished, :eol_symbol] = numpy.inf
            next_costs[finished, eol_symbol + 1:] = numpy.inf

            # The `i == 0` is required because at the first step the beam
            # size is effectively only 1.
            (indexes, outputs), chosen_costs = self._smallest(
                next_costs, self.beam_size, only_first_row=i == 0)

            # Rearrange everything
            for name in states:
                states[name] = states[name][indexes]
            all_outputs = all_outputs[:, indexes]
            mask = mask[indexes]
            costs = costs[indexes]

            # Record chosen output and compute new states
            states.update(self.compute_next_state(contexts, states, outputs))
            all_outputs = numpy.append(all_outputs, outputs[None, :], axis=0)
            costs = chosen_costs
            mask = (outputs != eol_symbol) * mask

        all_outputs = all_outputs[1:]
        mask = all_outputs != eol_symbol
        # The first `eol_symbol` should be preserved: we add an additional
        # 1 to each mask row to ensure that.
        for row in mask.T:
            row[row.sum()] = 1

        return all_outputs, mask, costs
