import copy
from functools import partial
from theano import tensor
from abc import abstractmethod
from blocks.bricks import (
    Brick, MLP, BaseRecurrent, Identity, ForkInputs, ApplySignature,
    zero_state)
from blocks.utils import dict_union


class BaseSequenceGenerator(BaseRecurrent):
    """A context-dependent sequence generator.

    This brick can generate a sequence by the following algorithm:

    1. Hidden state is initialized from the contexts using
    the `initializer` brick.

    2. The current state, the previous output and the contexts are
    summarized in a single varible using the `readout` brick.

    3. An output is generated from the summary by the `emitter` brick.

    4. The next state is obtained as follows. The feedback produced from the
    output by the `feedback` brick is fed into the `transition`. It is assumed
    that `transition` has only one input.

    5. Continue from step 2 while the number of steps done is less then
    required.

    Some of the aforementioned blocks can be set in the __init__ method.
    The rest can be replaced if necessary by a usual child replacement
    procedure.

    """
    @Brick.lazy_method
    def __init__(self, transition, emitter, feedback, null_output,
                 initializer=None, readout=None,
                 weights_init=None, biases_init=None, **kwargs):
        super(BaseSequenceGenerator, self).__init__(**kwargs)
        self.__dict__.update(**locals())
        del self.self

        self.transition_signature_func = (
            self.transition.get_signature_func('apply'))
        signature = self.transition_signature_func()
        assert len(signature.input_names) == 2
        assert signature.num_outputs == 0
        self.state_names = signature.state_names
        self.context_names = signature.context_names

        if not self.initializer:
            self.initializer = NullStateInitializer(
                self.state_names, self.context_names, null_output)
        if not self.readout:
            self.readout = DefaultReadout(
                self.state_names, self.context_names)

        self.children = [self.transition, self.initializer,
                         self.readout, self.emitter, self.feedback]

    def _push_allocation_config(self):
        transition_signature = self.transition_signature_func()
        state_dims = [transition_signature.dims[name]
                      for name in transition_signature.state_names]
        context_dims = [transition_signature.dims[name]
                        for name in transition_signature.context_names]

        feedback_signature = self.feedback.get_signature_func('apply')()
        self.readout.feedback_dim = feedback_signature.output_dims[0]

        for brick in self.readout, self.initializer:
            brick.state_dims = state_dims
            brick.context_dims = context_dims

    def _push_initialization_config(self):
        # Does not touch `transition` initilization
        for brick in [self.initializer, self.readout, self.feedback]:
            if self.weights_init:
                brick.weights_init = self.weights_init
            if self.biases_init:
                brick.biases_init = self.weights_init

    @Brick.apply_method
    def cost(self, outputs, mask=None, **kwargs):
        batch_size = outputs.shape[-2]

        contexts = {name: kwargs[name] for name in self.context_names}
        initial_states = {name: self.initializer.initialize_state(
                          name, batch_size=batch_size, **contexts)
                          for name in self.state_names}

        feedback = self.feedback.apply(outputs)
        states = [state[:-1] for state in self.transition.apply(
            feedback, mask=mask, iterate=True,
            return_initial_states=True, return_list=True,
            **dict_union(initial_states, contexts))]
        states = dict(zip(self.state_names, states))

        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.feedback.apply(self.initializer.initialize_state(
                'outputs', batch_size=batch_size, **contexts)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, contexts))
        return self.emitter.cost(readouts, outputs)

    @BaseRecurrent.recurrent_apply_method
    def generate(self, outputs, **kwargs):
        states = {name: kwargs[name] for name in self.state_names}
        contexts = {name: kwargs[name] for name in self.context_names}
        readouts = self.readout.readout(
            feedback=self.feedback.apply(outputs),
            **dict_union(states, contexts))
        next_outputs = self.emitter.emit(readouts)
        next_costs = self.emitter.cost(readouts, next_outputs)
        next_states = self.transition.apply(
            self.feedback.apply(next_outputs), **dict_union(states, contexts))
        return next_states, next_outputs, next_costs

    @generate.signature_method
    def generate_signature(self, *args, **kwargs):
        signature = copy.deepcopy(self.transition_signature_func())
        signature.state_names.append('outputs')
        signature.dims['outputs'] = (
            self.emitter.get_signature_func('emit')().output_dims[0])
        signature.num_outputs = 1
        for state_name in signature.state_names:
            # Adapt `self.initializer` to the standard
            # state initialization function interface
            def call_initializer(state_name, self, dim, batch_size,
                                 *args, **kwargs):
                contexts = {name: kwargs[name] for name in self.context_names}
                return self.initializer.initialize_state(
                    state_name, batch_size=batch_size, **contexts)
            signature.state_init_funcs[state_name] = (
                partial(call_initializer, state_name))
        return signature


class SequenceGenerator(BaseSequenceGenerator):
    """A more user-friendly subclass of the BaseSequenceGenerator.

    This class is responsible for wrapping the transition and adding the
    necessary contexts.

    .. todo:: context addition

    """

    def __init__(self, transition, *args, **kwargs):
        self.fork_inputs_wrapper = ForkInputs(transition)
        super(SequenceGenerator, self).__init__(self.fork_inputs_wrapper,
                                                *args, **kwargs)

    def _push_allocation_config(self):
        feedback_signature = self.feedback.get_signature_func('apply')()
        self.fork_inputs_wrapper.input_dim = feedback_signature.output_dims[0]
        super(SequenceGenerator, self)._push_allocation_config()

    def _push_initialization_config(self):
        if self.weights_init:
            self.fork_inputs_wrapper.weights_init = self.weights_init
        if self.biases_init:
            self.fork_inputs_wrapper.biases_init = self.biases_init
        super(SequenceGenerator, self)._push_initialization_config()


class NullStateInitializer(Brick):

    def __init__(self, state_names, context_names, null_output, **kwargs):
        super(NullStateInitializer, self).__init__(**kwargs)
        self.__dict__.update(**locals())

        assert len(self.context_names) == 0

    @Brick.apply_method
    def initialize_state(self, state_name, batch_size, **kwargs):
        if state_name == 'outputs':
            shape = [batch_size] + [self.null_output.shape[i]
                                    for i in range(self.null_output.ndim)]
            return tensor.alloc(self.null_output, *shape)
        else:
            state_index = self.state_names.index(state_name)
            return zero_state(
                None, self.state_dims[state_index], batch_size)


class DefaultReadout(Brick):

    @Brick.lazy_method
    def __init__(self, state_names, context_names,
                 weights_init, biases_init, **kwargs):
        super(DefaultReadout, self).__init__(**kwargs)
        self.__dict__.update(**locals())

        self.input_names = ['feedback'] + state_names + context_names
        self.projectors = [MLP(name="project_{}".format(name),
                               activations=[Identity()])
                           for name in self.input_names]
        self.children = self.projectors

    def _push_allocation_config(self):
        self.dims = [self.feedback_dim] + self.state_dims + self.context_dims
        for projector, dim in zip(self.projectors, self.dims):
            projector.dims[0] = dim
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
                       zip(self.input_names, self.projectors)]
        if len(projections) == 1:
            return projections[0]
        return sum(projections[1:], projections[0])


class TrivialEmitter(Brick):

    def __init__(self, output_dim, **kwargs):
        super(TrivialEmitter, self).__init__(**kwargs)
        self.output_dim = output_dim

    @Brick.apply_method
    def emit(self, readouts):
        return readouts

    @emit.signature_method
    def emit_signature(self, *args, **kwargs):
        return ApplySignature(output_dims=[self.output_dim])

    @abstractmethod
    def cost(self, readouts, outputs):
        pass


class TrivialFeedback(Brick):

    @Brick.lazy_method
    def __init__(self, output_dim, **kwargs):
        super(TrivialFeedback, self).__init__(**kwargs)
        self.output_dim = output_dim

    @Brick.apply_method
    def apply(self, outputs):
        return outputs

    @apply.signature_method
    def apply_signature(self):
        return ApplySignature(output_dims=[self.output_dim])
