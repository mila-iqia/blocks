"""Attention mechanisms.

We consider a hypothetical agent that wants to concentrate on particular
parts of a structured input. To do that the agent needs an *attention
mechanism* that given the *state* of the agent and the input signal outputs
*glimpses*.  For technical reasons we permit an agent to have a composite
state consisting of several components, to which we will refer as *states
of the agent* or simply *states*.

"""
from abc import ABCMeta, abstractmethod

from theano import tensor
from six import add_metaclass

from blocks.bricks import (MLP, Identity, Initializable, Sequence,
                           Feedforward, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Parallel, Distribute
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.utils import dict_union, dict_subset


class SequenceContentAttention(Initializable):
    """Attention mechanism that looks for relevant content in a sequence.

    This is the attention mechanism used in [BCB]_. The idea in a nutshell:

    1. The states and the sequence are transformed independently,

    2. The transformed states are summed with every transformed sequence
       element to obtain *match vectors*,

    3. A match vector is transformed into a single number interpreted as
       *energy*,

    4. Energies are normalized in softmax-like fashion. The resulting
       summing to one weights are called *attention weights*,

    5. Linear combination of the sequence elements with attention weights
       is computed.

    This linear combinations from 5 and the attention weights from 4 form
    the set of glimpses produced by this attention mechanism. The former
    will be referred to as *glimpses* in method documentation.

    Parameters
    ----------
    state_names : list of str
        The names of the agent states.
    sequence_dim : int
        The dimension of the sequence elements.
    match_dim : int
        The dimension of the match vector.
    state_transformer : :class:`.Brick`
        A prototype for state transformations. If ``None``, the default
        transformation from :class:`.Parallel` is used.
    sequence_transformer : :class:`.Brick`
        The transformation to be applied to the sequence. If ``None`` an
        affine transformation is used.
    energy_computer : :class:`.Brick`
        Computes energy from the match vector. If ``None``, an affine
        transformations is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    .. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
       Machine Translation by Jointly Learning to Align and Translate.

    """
    @lazy
    def __init__(self, state_names, state_dims, sequence_dim, match_dim,
                 state_transformer=None, sequence_transformer=None,
                 energy_computer=None,
                 **kwargs):
        super(SequenceContentAttention, self).__init__(**kwargs)
        self.state_names = state_names
        self.state_dims = state_dims
        self.sequence_dim = sequence_dim
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not sequence_transformer:
            sequence_transformer = MLP([Identity()], name="seq_trans")
        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.sequence_transformer = sequence_transformer
        self.energy_computer = energy_computer

        self.children = [self.state_transformers, sequence_transformer,
                         energy_computer]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = {name: self.match_dim
                                               for name in self.state_names}
        self.sequence_transformer.dims[0] = self.sequence_dim
        self.sequence_transformer.dims[-1] = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1

    @application(outputs=['glimpses', 'weights'])
    def take_look(self, sequence, preprocessed_sequence=None, mask=None,
                  **states):
        r"""Compute attention weights and produce glimpses.

        Parameters
        ----------
        sequence : :class:`~tensor.TensorVariable`
            The sequence, time is the 1-st dimension.
        preprocessed_sequence : :class:`~tensor.TensorVariable`
            The preprocessed sequence. If ``None``, is computed by calling
            :meth:`preprocess`.
        mask : :class:`~tensor.TensorVariable`
            A 0/1 mask specifying available data. 0 means that the
            corresponding sequence element is fake.
        \*\*states
            The states of the agent.

        Returns
        -------
        glimpses : theano variable
            Linear combinations of sequence elements with the attention
            weights.
        weights : theano variable
            The attention weights. The first dimension is batch, the second
            is time.

        """
        if not preprocessed_sequence:
            preprocessed_sequence = self.preprocess(sequence)
        transformed_states = self.state_transformers.apply(return_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_sequence)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        unormalized_weights = tensor.exp(energies)
        if mask:
            unormalized_weights *= mask
        weights = unormalized_weights / unormalized_weights.sum(axis=0)
        glimpses = (tensor.shape_padright(weights) * sequence).sum(axis=0)
        return glimpses, weights.dimshuffle(1, 0)

    @take_look.property('inputs')
    def take_look_inputs(self):
        return (['sequence', 'preprocessed_sequence', 'mask'] +
                self.state_names)

    @application
    def initial_glimpses(self, name, batch_size, sequence):
        if name == "glimpses":
            return tensor.zeros((batch_size, self.sequence_dim))
        elif name == "weights":
            return tensor.zeros((batch_size, sequence.shape[0]))
        else:
            raise ValueError("Unknown glimpse name {}".format(name))

    @application(inputs=['sequence'], outputs=['preprocessed_sequence'])
    def preprocess(self, sequence):
        """Preprocess a sequence for computing attention weights.

        Parameters
        ----------
        sequence : :class:`~tensor.TensorVariable`
            The sequence, time is the 1-st dimension.

        """
        return self.sequence_transformer.apply(sequence)

    def get_dim(self, name):
        if name in ['glimpses', 'sequence', 'preprocessed_sequence']:
            return self.sequence_dim
        if name in ['mask', 'weights']:
            return 0
        return super(SequenceContentAttention, self).get_dim(name)


class ShallowEnergyComputer(Sequence, Initializable, Feedforward):
    """A simple energy computer: First tanh, then weighted sum."""
    @lazy
    def __init__(self, **kwargs):
        super(ShallowEnergyComputer, self).__init__(
            [Tanh().apply, MLP([Identity()]).apply], **kwargs)

    @property
    def input_dim(self):
        return self.children[1].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[1].input_dim = value

    @property
    def output_dim(self):
        return self.children[1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[1].output_dim = value


@add_metaclass(ABCMeta)
class AbstractAttentionRecurrent(BaseRecurrent):
    """The interface for attention-equipped recurrent transitions.

    When a recurrent network is equipped with an attention mechanism its
    transition typically consists of two steps: (1) the glimpses are taken
    by the attention mechanism and (2) the next states are computed using
    the current states and the glimpses. It is required for certain
    usecases (such as sequence generator) that apart from a do-it-all
    recurrent application method interfaces for the first step and
    the second steps of the transition are provided.

    """
    @abstractmethod
    def apply(self, **kwargs):
        """Compute next states taking glimpses on the way."""
        pass

    @abstractmethod
    def take_look(self, **kwargs):
        """Compute glimpses given the current states."""
        pass

    @abstractmethod
    def compute_states(self, **kwargs):
        """Compute next states given current states and glimpses."""
        pass


class AttentionRecurrent(AbstractAttentionRecurrent, Initializable):
    """Combines an attention mechanism and a recurrent transition.

    This brick equips a recurrent transition with an attention mechanism.
    In order to do this two more contexts are added: one to be attended and
    a mask for it. It is also possible to use the contexts of the given
    recurrent transition for these purposes and not add any new ones,
    see `add_context` parameter.

    At the beginning of each step attention mechanism produces glimpses;
    these glimpses together with the current states are used to compute the
    next state and finish the transition. In some cases glimpses from the
    previous steps are also necessary for the attention mechanism, e.g.
    in order to focus on an area close to the one from the previous step.
    This is also supported: such glimpses become states of the new
    transition.

    To let the user control the way glimpses are used, this brick also
    takes a "distribute" brick as parameter that distributes the
    information from glimpses across the sequential inputs of the wrapped
    recurrent transition.

    Parameters
    ----------
    transition : :class:`.BaseRecurrent`
        The recurrent transition.
    attention : :class:`.Brick`
        The attention mechanism.
    distribute : :class:`.Brick`, optional
        Distributes the information from glimpses across the input
        sequence of the transition. By default a :class:`.Distribute` is
        used, and those inputs containing the "mask" substring in their
        name are not affected.
    add_contexts : bool, optional
        If ``True``, new contexts for the attended and the attended mask
        are added to this transition. ``True`` by default.
    attended_name : str
        The name of the attended context. If ``None``, "attended"
        or the first context of the recurrent transition is used,
        depending on the value of `add_contents` flag.
    attended_mask_name : str
        The name of the mask for the attended context. If ``None``,
        "attended_mask" or the second context of the recurrent transition
        is used depending on the value of `add_contents` flag.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    Those coming to Blocks from Groundhog might recognize that this is
    a `RecurrentLayerWithSearch`, but on steroids :)

    """
    def __init__(self, transition, attention, distribute,
                 add_contexts=True,
                 attended_name=None, attended_mask_name=None,
                 **kwargs):
        super(AttentionRecurrent, self).__init__(**kwargs)
        self.sequence_names = transition.apply.sequences
        self.state_names = transition.apply.states
        self.context_names = transition.apply.contexts
        if not attended_name:
            if add_contexts:
                attended_name = 'attended'
            else:
                attended_name = self.context_names[0]
        if not attended_mask_name:
            if add_contexts:
                attended_mask_name = 'attended_mask'
            else:
                attended_mask_name = self.context_names[1]
        if not distribute:
            normal_inputs = [name for name in self.sequence_names
                             if not 'mask' in name]
            distribute = Distribute(normal_inputs,
                                    attention.take_look.outputs[0])

        self.transition = transition
        self.attention = attention
        self.distribute = distribute
        self.attended_name = attended_name
        self.attended_mask_name = attended_mask_name

        self.preprocessed_attended_name = "preprocessed_" + self.attended_name

        self.glimpse_names = self.attention.take_look.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_look` signature.
        self.previous_glimpses_needed = [
            name for name in self.glimpse_names
            if name in self.attention.take_look.inputs]

        self.children = [self.transition, self.attention, self.distribute]

    def _push_allocation_config(self):
        self.attention.state_dims = self.transition.get_dims(self.state_names)
        self.attention.sequence_dim = self.transition.get_dim(
            self.attended_name)
        self.distribute.target_dims = dict_subset(
            dict_union(
                self.transition.get_dims(self.sequence_names)),
            self.distribute.target_names)
        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)

    @application
    def take_look(self, **kwargs):
        r"""Compute glimpses with the attention mechanism.

        A thin wrapper over `self.attention.take_look`: takes care
        of choosing and renaming the necessary arguments.

        Parameters
        ----------
        \*\*kwargs
            Should contain contexts, previous step states and glimpses.

        Returns
        -------
        glimpses : list of :class:`~tensor.TensorVariable`
            Current step glimpses.

        """
        return self.attention.take_look(
            kwargs[self.attended_name],
            kwargs.get(self.preprocessed_attended_name),
            mask=kwargs.get(self.attended_mask_name),
            **dict_subset(kwargs,
                          self.state_names + self.previous_glimpses_needed))

    @take_look.property('outputs')
    def take_look_outputs(self):
        return self.glimpse_names

    @application
    def compute_states(self, **kwargs):
        r"""Compute current states when glimpses have already been computed.

        Combines an application of the `distribute` that alter the
        sequential inputs of the wrapped transition and an application of
        the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain everything what `self.transition` needs
            and in addition current glimpses.

        Returns
        -------
        current_states : list of :class:`~tensor.TensorVariable`
            Current states computed by `self.transition`.

        """
        # Masks are not mandatory, that's why 'must_have=False'
        sequences = dict_subset(kwargs, self.sequence_names,
                                pop=True, must_have=False)
        states = dict_subset(kwargs, self.state_names, pop=True)
        glimpses = dict_subset(kwargs, self.glimpse_names, pop=True)
        sequences.update(self.distribute.apply(
            return_dict=True,
            **dict_subset(dict_union(sequences, glimpses),
                          self.distribute.apply.inputs)))
        current_states = self.transition.apply(
            iterate=False, return_list=True,
            **dict_union(sequences, states, kwargs))
        return current_states

    @compute_states.property('outputs')
    def compute_states_outputs(self):
        return self.state_names

    @recurrent
    def do_apply(self, **kwargs):
        r"""Process a sequence attending the attended context every step.

        In addition to the original sequence this method also requires
        its preprocessed version, the one computed by the `preprocess`
        method of the attention mechanism.

        Parameters
        ----------
        \*\*kwargs
            Should contain current inputs, previous step states, contexts,
            the preprocessed attended context, previous step glimpses.

        Returns
        -------
        outputs : list of :class:`~tensor.TensorVariable`
            The current step states and glimpses.

        """
        attended = kwargs[self.attended_name]
        preprocessed_attended = kwargs.pop(self.preprocessed_attended_name)
        attended_mask = kwargs.get(self.attended_mask_name)
        sequences = dict_subset(kwargs, self.sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self.state_names, pop=True)
        glimpses = dict_subset(kwargs, self.glimpse_names, pop=True)

        current_glimpses = self.take_look(
            return_dict=True,
            **dict_union(
                states, glimpses,
                {self.attended_name: attended,
                 self.attended_mask_name: attended_mask,
                 self.preprocessed_attended_name: preprocessed_attended}))
        current_states = self.compute_states(
            return_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self.transition.apply.sequences

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self.transition.apply.contexts + [
            self.preprocessed_attended_name]

    @do_apply.property('states')
    def do_apply_states(self):
        return self.transition.apply.states + self.glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self.transition.apply.states + self.glimpse_names

    @application
    def apply(self, **kwargs):
        """Preprocess a sequence attending the attended context at every step.

        Preprocesses the attended context and runs :meth:`do_apply`. See
        :meth:`do_apply` documentation for further information.

        """
        preprocessed_attended = self.attention.preprocess(
            kwargs[self.attended_name])
        return self.do_apply(
            **dict_union(kwargs,
                         {self.preprocessed_attended_name:
                          preprocessed_attended}))

    @apply.delegate
    def apply_delegate(self):
        # TODO: Nice interface for this trick?
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self.transition.apply.contexts

    @application
    def initial_state(self, state_name, batch_size, **kwargs):
        if state_name in self.glimpse_names:
            return self.attention.initial_glimpses(
                state_name, batch_size, kwargs[self.attended_name])
        return self.transition.initial_state(state_name, batch_size, **kwargs)

    def get_dim(self, name):
        if name in self.glimpse_names:
            return self.attention.get_dim(name)
        if name == self.preprocessed_attended_name:
            (original_name,) = self.attention.preprocess.outputs
            return self.attention.get_dim(original_name)
        return self.transition.get_dim(name)
