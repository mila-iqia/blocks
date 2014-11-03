"""Sequence generation framework"""

import copy
from functools import partial
from theano import tensor
from abc import abstractmethod
from blocks.bricks import (
    Brick, MLP, BaseRecurrent, Identity, ForkInputs, ApplySignature,
    zero_state)
from blocks.utils import dict_union


class BaseSequenceGenerator(BaseRecurrent):
    """A generic sequence generator.

    This class combines two components, a readout network and an
    attention-equipped recurrent transition, into a context-dependent
    sequence generator. The generation algorithm description follows.

    *Definitions:*

    * states of the generator are the states of the transition as specified
      by its `apply` method signature

    * contexts of the generator are the contexts of the transition as specified
      in its `apply` method signature

    * glimpses are intermediate entities computed from the states, the contexts
      and the previous step glimpses. They are computed by in the transition's
      `apply` method when not given or by explicitly calling the transition's
      `take_look` method. The output names of `take_look` specify the set of
      glimpses taken by the generator.

    *Algorithm:*

    0. The initial states are computed from the contexts. The transition
       signature is assumed to contain the initialization logic. Fake previous
       outputs and fake previous glimpses are created using `initial_outputs`
       and `initial_glimpses` methods of the readout and the transition
       respectively.

    1. Given the contexts, the current state and glimpses from the previous
       step the attention mechanism hidden in the transition produces current
       step glimpses. This happens in the `take_look` method of the transition.

    2. Using the contexts, the fed back output from the previous step,
       the current states and glimpses, the readout brick is used to generate
       the new output by calling its `readout` and `emit` methods.

    3. The new output is fed back in the `feedback` method of the readout
       brick. This feedback, together with the contexts, the glimpses and
       the previous states is used to get the new states in the transition's
       `apply` method.

    4. Back to step 1, if desired sequence length is not yet reached.

    *Notes:*

    * For machine translation we would have only one glimpse: the weighted
      average of the annotations.

    * For speech recognition we would have three: the weighted average,
      the alignment and the monotonicity penalty.

    Parameters
    ----------
        readout : a subclass of AbstractReadout
            The readout component of the sequence generator.
        transition : a subclass of AbstractAttentionTransition
            The transition component of the sequence generator.

    """
    @Brick.lazy_method
    def __init__(self, readout, transition,
                 weights_init=None, biases_init=None, **kwargs):
        super(BaseSequenceGenerator, self).__init__(**kwargs)
        self.__dict__.update(**locals())
        del self.self

        signature = self.transition.signature('apply')
        assert len(signature.input_names) == 2
        assert 'mask' in signature.input_names
        self.state_names = signature.state_names
        self.context_names = signature.context_names

        self.glimpse_names = (
            self.transition.signature('take_look').output_names)

        self.children = [self.readout, self.transition]

    def _push_allocation_config(self):
        # Configure readout
        apply_signature = self.transition.signature('apply')
        take_look_signature = self.transition.signature('take_look')
        state_dims = {name: apply_signature.dims[name]
                      for name in apply_signature.state_names}
        context_dims = {name: apply_signature.dims[name]
                        for name in apply_signature.context_names}
        self.glimpse_dims = {name: take_look_signature.output_dims[i]
            for i, name in enumerate(take_look_signature.output_names)}
        self.readout.source_dims = dict_union(
            state_dims, context_dims, self.glimpse_dims)

        # Configure transition
        feedback_signature = self.readout.signature('feedback')
        assert len(feedback_signature.output_dims) == 1
        self.transition.input_dim = feedback_signature.output_dims[0]

    def _push_initialization_config(self):
        for brick in [self.readout, self.transition]:
            if self.weights_init:
                brick.weights_init = self.weights_init
            if self.biases_init:
                brick.biases_init = self.weights_init

    @Brick.apply_method
    def cost(self, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        Parameters
        ----------
        outputs : Theano variable
            The 3(2) dimensional tensor containing output sequences.
            The dimension 0 must stand for time, the dimension 1 for the
            position on the batch.
        mask : The 0/1 matrix identifying fake outputs.

        Notes
        -----
        The contexts are expected as keyword arguments.

        """
        batch_size = outputs.shape[-2]

        # Prepare input for the iterative part
        contexts = {name: kwargs[name] for name in self.context_names}
        feedback = self.readout.feedback(outputs)

        # Run the recurrent network
        results = self.transition.apply(
            feedback, mask=mask, iterate=True,
            return_initial_states=True, return_dict=True, **contexts)

        # Separate the deliverables
        states = {name: results[name][:-1] for name in self.state_names}
        glimpses = {name: results[name] for name in self.glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(
                None, batch_size, **contexts)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)

        # In case the user needs some glimpses or states or smth else
        also_return =  kwargs.get("also_return")
        if also_return:
            others = {name: results[name] for name in also_return}
            return (costs, others)
        return costs

    @BaseRecurrent.recurrent_apply_method
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : Theano variable
            The outputs from the previous step.

        Notes
        -----
            The contexts, previous states and glimpses are expected
            as keyword arguments.

        """

        states = {name: kwargs[name] for name in self.state_names}
        contexts = {name: kwargs[name] for name in self.context_names}
        glimpses = {name: kwargs[name] for name in self.glimpse_names}

        next_glimpses = self.transition.take_look(
            return_dict=True,
            **dict_union(states, glimpses, contexts))
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, next_glimpses, contexts))
        next_outputs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        next_states = self.transition.apply(
            self.readout.feedback(next_outputs), return_list=True,
            **dict_union(states, glimpses, contexts))
        return next_states + [next_outputs] + next_glimpses.values() + [next_costs]

    @generate.signature_method
    def generate_signature(self, *args, **kwargs):
        signature = copy.deepcopy(self.transition.signature('apply'))

        signature.state_names.append('outputs')
        signature.dims['outputs'] = (
            self.readout.signature('emit').output_dims[0])
        signature.state_init_funcs['outputs'] = self.readout.initial_outputs

        for name in self.glimpse_names:
            signature.glimpse_names.append(name)
            signature.glimpse_dims[name] = self.glimpse_dims[name]
            signature.state_init_funcs[name] = partial(
                self.initial_glimpses, name)

        signature.num_outputs = 1

        return signature


class AbstractReadout(Brick):
    """A base class for a readout component of a sequence generator.

    Yields outputs combining information from multiple sources.

    """

    @abstractmethod
    def emit(self, **kwargs):
        pass

    @abstractmethod
    def feedback(self, **kwargs):
        pass

    @abstractmethod
    def cost(self, **kwargs):
        pass

    @abstractmethod
    def initial_outputs(self, dim, batch_size, **kwargs):
        pass


class AbstractAttentionTransition(BaseRecurrent):
    """A base class for a transition component of a sequence generator.

    A recurrent transition combined with an attention mechanism.

    """

    @abstractmethod
    def apply(self, **kwargs):
        pass

    @abstractmethod
    def take_look(self, **kwargs):
        pass

    @abstractmethod
    def initial_glimpses(self, glimpse_name, dim, batch_size, **kwargs):
        pass


class SimpleReadout(AbstractReadout):

    @Brick.lazy_method
    def __init__(self, readout_dim, source_names,
                 weights_init, biases_init, **kwargs):
        super(SimpleReadout, self).__init__(**kwargs)
        self.__dict__.update(**locals())

        self.projectors = [MLP(name="project_{}".format(name),
                               activations=[Identity()])
                           for name in self.source_names]
        self.children = self.projectors

    def _push_allocation_config(self):
        for name, projector in zip(self.source_names, self.projectors):
            projector.dims[0] = self.source_dims[name]
            projector.dims[-1] = self.readout_dim

    def _push_initialization_config(self):
        for child in self.children:
            if self.weights_init:
                child.weights_init = self.weights_init
            if self.biases_init:
                child.biases_init = self.biases_init

    @Brick.apply_method
    def readout(self, **kwargs):
        projections = [projector.apply(kwargs[name]) for name, projector in
                       zip(self.source_names, self.projectors)]
        if len(projections) == 1:
            return projections[0]
        return sum(projections[1:], projections[0])

    @Brick.apply_method
    def emit(self, readouts, **kwargs):
        return readouts

    @emit.signature_method
    def emit_signature(self, *args, **kwargs):
        return ApplySignature(output_dims=[self.readout_dim])

    @Brick.apply_method
    def feedback(self, outputs, **kwargs):
        return outputs

    @feedback.signature_method
    def feedback_signature(self, *args, **kwargs):
        return ApplySignature(output_dims=[self.readout_dim])

    @Brick.apply_method
    def initial_outputs(self, dim, *args, **kwargs):
        if not dim:
            dim = self.readout_dim
        return zero_state(dim, *args, **kwargs)


class AttentionTransition(AbstractAttentionTransition):

    @Brick.lazy_method
    def __init__(self, transition, input_dim, attention=None, **kwargs):
        super(AttentionTransition, self).__init__(**kwargs)
        self.__dict__.update(**locals())
        del self.self
        del self.kwargs

        if self.attention:
            raise NotImplementedError()

        self.fork_inputs = ForkInputs(self.transition)
        self.children = [self.transition, self.fork_inputs]

    def _push_allocation_config(self):
        self.fork_inputs.input_dim = self.input_dim

    def _push_initialization_config(self):
        for child in self.children:
            if self.weights_init:
                child.weights_init = self.weights_init
            if self.biases_init:
                child.biases_init = self.biases_init

    @BaseRecurrent.recurrent_apply_method
    def apply(self, feedback, *args, **kwargs):
        return self.fork_inputs.apply(feedback, *args, **kwargs)

    @apply.signature_method
    def apply_signature(self, *args, **kwargs):
        signature = self.fork_inputs.signature('apply')
        common_input_index = signature.input_names.index(ForkInputs.common_input)
        signature.input_names[common_input_index] = 'feedback'
        return signature

    @Brick.apply_method
    def take_look(self, *args, **kwargs):
        return None

    @take_look.signature_method
    def take_look_signature(self, *args, **kwargs):
        return ApplySignature([], [])

    @Brick.apply_method
    def initial_glimpses(self, *args, **kwargs):
        raise NotImplementedError()
