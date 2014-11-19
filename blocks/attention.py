"""Attention mechanisms.

We consider a hypothetical agent that wants to concentrate on particular parts
of a structured input. To do that the agent needs an *attention mechanism*
that given the *state* of the agent and the input signal outputs *attention
weights*. The attention weights indicate relevance of input positions for the
agent in its current state. For technical reasons we permit an agent to have
a composite state consisting of several components, to which we will refer as
*states of the agent* or simply *states*.

"""

from theano import tensor

from blocks.bricks import Brick, MLP, Identity, lazy, application
from blocks.parallel import Parallel
from blocks.utils import update_instance


class SequenceContentAttention(Brick):
    """An attention mechanism that looks for relevant content in a sequence.

    This is the attention mechanism is used in [1]. The idea in a nutshell:

    1. the states and the sequence are transformed indepently,

    2. the transformed states are summed with every transformed sequence
       element to obtain *match vectors*,

    3. a linear combination of a match vector elements is computed and
       interpreted as *energy*,

    4. energies are normalized in softmax-like fashion. The resulting summing
       to one weights are called *attention weights*.

    .. [1] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural Machine
    Translation by Jointly Learning to Align and Translate

    """
    @lazy
    def __init__(self, state_names, state_dims, sequence_dim, match_dim,
                 state_transformer=None, sequence_transformer=None,
                 energy_computer=None, weights_init=None, biases_init=None,
                 **kwargs):
        super(SequenceContentAttention, self).__init__()
        update_instance(self, locals())

        self.state_transformers = Parallel(state_names, self.state_transformer,
                                           name="state_trans")
        if not self.sequence_transformer:
            self.sequence_transformer = MLP([Identity()], name="seq_trans")
        if not self.energy_computer:
            self.energy_computer = MLP([Identity()], name="energy_comp")
        self.children = [self.state_transformers, self.sequence_transformer,
                         self.energy_computer]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = {name: self.match_dim
                                               for name in self.state_names}
        self.sequence_transformer.dims[0] = self.sequence_dim
        self.sequence_transformer.dims[-1] = self.match_dim
        self.energy_computer.dims[0] = self.match_dim
        self.energy_computer.dims[-1] = 1

    def _push_initialization_config(self):
        for child in self.children:
            if self.weights_init:
                child.weights_init = self.weights_init
            if self.biases_init:
                child.biases_init = self.biases_init

    @application
    def take_look(self, sequence, **states):
        """Compute attention weights for a sequence.

        Parameters
        ----------
        sequence : Theano variable
            The sequence, time is the 1-st dimension.
        **states
            The states of the agent.

        """
        return self.take_look_preprocessed(self.preprocess(sequence), **states)

    @application
    def take_look_preprocessed(self, preprocessed_sequence, **states):
        """Compute attention weights for a preprocessed sequence.

        Parameters
        ----------
        preprocessed_sequence : Theano variable
            The sequence, time is the 1-st dimension.
        **states
            The states of the agent.

        """
        transformed_states = self.state_transformers.apply(return_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_sequence)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return tensor.nnet.softmax(energies)

    @application
    def preprocess(self, sequence):
        """Preprocess a sequence for computing attention weights.

        Parameters
        ----------
        sequence : Theano variable
            The sequence, time is the 1-st dimension.

        """
        return self.sequence_transformer.apply(sequence)
