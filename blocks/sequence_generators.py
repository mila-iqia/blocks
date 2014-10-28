from theano import tensor
from functools import partial
from blocks.bricks import Brick, MLP, BaseRecurrent, Identity, ForkInputs


class NullStateInitializer(Brick):

    def __init__(self, state_names, context_names, null_output, **kwargs):
        super(Brick, self).__init__(**kwargs)
        self.__dict__.update(**locals())

        assert len(self.context_names) == 0

    def initialize_state(self, state_name, batch_size, **kwargs):
        state_index = self.state_names.index(state_name)
        if state_name == 'outputs':
            pattern = (batch_size,) + (1,) * self.null_output.ndim
            return tensor.tile(pattern)
        else:
            return BaseRecurrent.zero_state(
                None, self.state_dims[state_index], batch_size)


class DefaultSummarizer(Brick):

    @Brick.lazy_init
    def __init__(self, state_names, context_names, summary_dim,
                 weights_init, biases_init, **kwargs):
        super(Brick, self).__init__(**kwargs)
        self.__dict__.update(**locals())

        self.input_names = ['feedback'] + state_names + context_names
        self.projectors = [MLP(name="project_{}".format(name),
                               activations=[Identity()])
                           for name in self.input_names]
        self.children = self.projectors

    def _push_allocation_config(self):
        self.dims = self.state_dims + self.context_dims
        for projector, dim in zip(self.projectors, self.dims):
            projector.dims[0] = dim
            projector.dims[-1] = self.summary_dim

    def _push_initialization_config(self):
        for child in self.children:
            if self.weights_init:
                child.weights_init = self.weights_init
            if self.biases_init:
                child.biases_init = self.biases_init

    @Brick.apply_method
    def summarize(self, **kwargs):
        projections = [projector(kwargs[name]) for name, projector in
                       zip(self.input_names, self.projectors)]
        if len(projections) == 1:
            return projections[0]
        return sum(projections[1:], projections[0])


class AddContexts(BaseRecurrent):
    pass


def dict_union(*dicts, **kwargs):
    result = dict()
    for d in list(dicts) + [kwargs]:
        assert len(set(result.keys()).intersection(set(d.keys()))) == 0
        result.update(d)
    return result


class SequenceGenerator(BaseRecurrent):
    """A context-dependent sequence generator.

    This brick can generate a sequence by the following algorithm:

    1. Hidden state is initialized from the contexts using
    the `initializer` brick.

    2. The current state, the previous output and the contexts are
    summarized in a single varible using the `summarizer` brick.

    3. An output is generated from the summary by the `generator` brick.

    4. The next state is obtained as follows. The feedback produced from the
    output by the `feedback` brick is fed into the `transition` brick
    wrapped by `fork_inputs` and `add_contexts` blocks. The wrappers ensure
    that the transition function has one input and uses contexts.

    5. Continue from step 2 while the number of steps done is less then
    required.

    Some of the aforementioned blocks can be set in the __init__ method.
    The rest can be replaced if necessary by a usual child replacement
    procedure.

    """
    @Brick.lazy_init
    def __init__(self, dim, summary_dim, context_names, context_dims,
                 transition, generator, feedback, null_output,
                 weights_init=None, biases_init=None, **kwargs):
        super(SequenceGenerator, self).__init__(**kwargs)
        self.__dict__.update(**locals())
        del self.self

        # Wrap the transition to ensure that is one input and uses contexts
        self.add_contexts = AddContexts(transition, context_names)
        self.fork_inputs = ForkInputs(self.add_contexts)
        self.wrapped_transition = self.fork_inputs

        # Get the transition state names
        self.wrapped_signature_func = partial(getattr(
            self.wrapped_transition.__class__, 'apply').signature, self)
        signature = self.wrapped_signature_func()
        self.states_names = signature.state_names

        self.initializer = NullStateInitializer(self.state_names,
                                                context_names, null_output)
        self.summarizer = DefaultSummarizer(self.state_names,
                                            context_names)

        self.children = [self.wrapped_transition, self.initializer,
                         self.summarizer, self.generator, self.feedback]

    def _push_allocation_config(self):
        self.add_contexts.context_dims = self.context_dims
        self.fork_inputs.input_dim = self.dim

        self.transition.dim = self.dim
        signature = self.wrapped_signature_func()
        state_dims = {name: signature.dims[name]
                      for name in signature.state_names}

        for brick in self.summarizer, self.initializer:
            brick.state_dims = state_dims
            brick.context_dims = self.context_dims

    def _push_initialization_config(self):
        # Does not touch `transition` initilization
        for brick in [self.add_contexts, self.fork_inputs, self.initializer,
                      self.summarizer, self.generator, self.feedback]:
            if self.weights_init:
                brick.weights_init = self.weights_init
            if self.biases_init:
                brick.biases_init = self.weights_init

    def cost(self, outputs, mask, **kwargs):
        batch_size = outputs.shape[-2]

        contexts = [kwargs[name] for name in self.context_names]
        initial_states = {name: self.initializer.initialize_state(
                            name, batch_size=batch_size, **contexts)
                          for name in self.state_names}

        feedback = self.feedback.apply(outputs)
        states = self.wrapped_transition.apply(
            feedback, iterate=True, return_initial_states=True,
            **dict_union(initial_states, contexts))
        states = dict(zip(self.state_names, states))

        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(feedback[0],
            self.feedback.apply(self.initializer.initialize_state(
                'outputs', batch_size=batch_size, **contexts)))
        summaries = self.summarizer(feedback=feedback,
                                    **dict_union(states, contexts))
        return self.generator.cost(summaries[:-1], outputs)

    @BaseRecurrent.recurrent_apply_method
    def generate(self, outputs, **kwargs):
        states = [kwargs[name] for name in self.state_names]
        contexts = [kwargs[name] for name in self.context_names]
        summaries = self.summarizer(feedback=self.feedback.apply(outputs),
                                    **dict_union(states, contexts))
        next_outputs = self.generator.generate(summaries)
        next_costs = self.generator.cost(summaries, next_outputs)
        next_states = self.wrapped_transition.apply(
            self.feedback.apply(next_outputs), **dict_union(states, contexts))
        return next_states, next_outputs, next_costs

    @generate.signature_method
    def generate_signature(self):
        signature = self.wrapped_signature_func()
        signature.state_names.append('outputs')
        for state_name in signature.state_names:
            def call_initializer(self, batch_size, *args, **kwargs):
                contexts = [kwargs[name] for name in self.context_names]
                return self.initializer.initialize_state(
                    state_name, batch_size=batch_size, **contexts)
            signature.state_init_func[state_name] = call_initializer
        return signature
