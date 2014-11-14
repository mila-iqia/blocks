import inspect

import theano
from theano import tensor

from blocks.bricks import (application_wrapper, DefaultRNG, Identity, lazy,
                           tag)
from blocks.utils import shared_floatx_zeros


def recurrent(*args, **kwargs):
    """Wraps an apply method to allow its iterative application.

    This decorator allows you to use implementation of an RNN
    transition to process sequences without writing the
    iteration-related code again and again. In the most general form
    information flow of a recurrent network can be described as
    follows: depending on the context variables and driven by input
    sequences the RNN updates its states and produces output sequences.
    Thus the input variables of your transition function play one of
    three roles: an input, a context or a state. These roles should be
    specified in the method's signature to make iteration possible.

    Parameters
    ----------
    inputs : list of strs
        Names of the arguments of the apply method that play input
        roles.
    states : list of strs
        Names of the arguments of the apply method that play state
        roles.
    contexts : list of strs
        Names of the arguments of the apply method that play context
        roles.
    outputs : list of strs
        Names of the outputs.

    """
    def recurrent_wrapper(application, application_method):
        arg_spec = inspect.getargspec(application_method)
        arg_names = arg_spec.args[1:]

        def recurrent_apply(brick, *args, **kwargs):
            """Iterates a transition function.

            Parameters
            ----------
            iterate : bool
                If ``True`` iteration is made. By default ``True``
                which means that transition function is applied as is.
            reverse : bool
                If ``True``, the sequences are processed in backward
                direction. ``False`` by default.

            .. todo::

                * Handle `updates` returned by the `theano.scan`
                    routine.
                * ``kwargs`` has a random order; check if this is a
                    problem.

            """
            # Extract arguments related to iteration.
            iterate = kwargs.pop('iterate', True)
            reverse = kwargs.pop('reverse', False)

            # Push everything to kwargs
            for arg, arg_name in zip(args, arg_names):
                kwargs[arg_name] = arg
            # Separate kwargs that aren't sequence, context or state variables
            scan_arguments = (application.sequences + application.states +
                              application.contexts)
            rest_kwargs = {key: value for key, value in kwargs.items()
                           if key not in scan_arguments}

            # Check what is given and what is not
            def only_given(arg_names):
                return {arg_name: kwargs[arg_name] for arg_name in arg_names
                        if arg_name in kwargs}
            sequences_given = only_given(application.sequences)
            contexts_given = only_given(application.contexts)

            # TODO Assumes 1 time dim!
            if len(sequences_given):
                shape = sequences_given.values()[0].shape
                if not iterate:
                    batch_size = shape[0]
                else:
                    n_steps = shape[0]
                    batch_size = shape[1]
            else:
                # TODO Raise error if n_steps and batch_size not found?
                n_steps = kwargs.pop('n_steps')
                batch_size = kwargs.pop('batch_size')

            # Ensure that all initial states are available.
            for state_name in application.states:
                if state_name in kwargs and callable(kwargs[state_name]):
                    # TODO Allow initialization function to be passed
                    pass
                elif state_name not in kwargs:
                    # TODO init_func returns 2D-tensor, fails for iterate=False
                    if hasattr(brick, 'initial_state'):
                        init_func = brick.initial_state
                    else:
                        init_func = zero_state
                    dim = brick.dims[state_name]
                    kwargs[state_name] = init_func(dim, batch_size)
            states_given = only_given(application.states)
            assert len(states_given) == len(application.states)

            # Theano issue 1772
            for name, state in states_given.items():
                states_given[name] = tensor.unbroadcast(state,
                                                        *range(state.ndim))

            # Apply methods
            if not iterate:
                return application_method(brick, **kwargs)

            def scan_function(*args):
                args = list(args)
                arg_names = (sequences_given.keys() + states_given.keys() +
                             contexts_given.keys())
                kwargs = dict(zip(arg_names, args))
                kwargs.update(rest_kwargs)
                return application_method(brick, **kwargs)
            result, updates = theano.scan(
                scan_function, sequences=sequences_given.values(),
                outputs_info=states_given.values(),
                non_sequences=contexts_given.values(),
                n_steps=n_steps,
                go_backwards=reverse)
            assert not updates  # TODO Handle updates
            return result

        return recurrent_apply

    # Decorator can be used with or without arguments
    assert (args and not kwargs) or (not args and kwargs)
    if args:
        application_method, = args
        application = application_wrapper()(application_method)
        return application.wrap(recurrent_wrapper).wrap(tag)
    else:
        def wrapper(application_method):
            application = application_wrapper(**kwargs)(application_method)
            return application.wrap(recurrent_wrapper).wrap(tag)
        return wrapper


# class Wrap3D(Brick):
#     """Convert 3D arrays to 2D and back in order to apply 2D bricks."""
#     @Brick.lazy_method
#     def __init__(self, wrapped, apply_method='apply', **kwargs):
#         super(Wrap3D, self).__init__(**kwargs)
#         self.children = [wrapped]
#         self.apply_method = apply_method
#
#     @application
#     def apply(self, inp):
#         wrapped, = self.children
#         flat_shape = ([inp.shape[0] * inp.shape[1]] +
#                       [inp.shape[i] for i in range(2, inp.ndim)])
#         output = getattr(wrapped, self.apply_method)(inp.reshape(flat_shape))
#         full_shape = ([inp.shape[0], inp.shape[1]] +
#                       [output.shape[i] for i in range(1, output.ndim)])
#         return output.reshape(full_shape)


def zero_state(dim, batch_size, *args, **kwargs):
    """Create an initial state consisting of zeros.

    The default state initialization routine. It is not made a method
    to ensure that the brick argument can be omitted.

    """
    return tensor.zeros((batch_size, dim), dtype=theano.config.floatX)


class Recurrent(DefaultRNG):
    """Simple recurrent layer with optional activation.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    weights_init : object
        The :class:`utils.NdarrayInitialization` object to initialize the
        weight matrix with.
    activation : Brick
        The brick to apply as activation.

    .. todo::

       Implement deep transitions (by using other bricks). Currently, this
       probably re-implements too much from the Linear brick.

       Other important features:

       * Carrying over hidden state between batches
       * Return k last hidden states

    """
    @lazy
    def __init__(self, dim, weights_init, activation=None, **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        if activation is None:
            activation = Identity()
        self.__dict__.update(locals())
        del self.self
        del self.kwargs
        self.dims = {state: dim for state in self.apply.states}

    @property
    def W(self):
        return self.params[0]

    def _allocate(self):
        self.params.append(shared_floatx_zeros((self.dim, self.dim)))

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inp', 'mask'], states=['state'], outputs=['state'],
               contexts=[])
    def apply(self, inp=None, state=None, mask=None):
        """Given data and mask, apply recurrent layer.

        Parameters
        ----------
        inp : Theano variable
            The 2 dimensional input, in the shape (batch, features).
        state : Theano variable
            The 2 dimensional state, in the shape (batch, features).
        mask : Theano variable
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        .. todo::

           * Mask should become part of ``MaskedTensorVariable`` type so
             that it can be passed around transparently.
           * We should stop assuming that batches are the second dimension,
             in order to support nested RNNs i.e. where the first n axes
             are time, n + 1 is the batch, and n + 2, ... are features.
             Masks will become n + 1 dimensional as well then.

        """
        next_state = inp + tensor.dot(state, self.W)
        next_state = self.activation.apply(next_state)
        if mask:
            next_state = (mask[:, None] * next_state +
                          (1 - mask[:, None]) * state)
        return next_state


# class GatedRecurrent(DefaultRNG):
#     """Gated recurrent neural network.
# 
#     Gated recurrent neural network (GRNN) as introduced in [1]. Every unit
#     of a GRNN is equiped with update and reset gates that fascilitate better
#     gradient propagation.
# 
#     Parameters
#     ----------
#     activation : Brick or None
#         The brick to apply as activation. If `None` an `Identity` brick is
#         used.
#     gated_activation : Brick or None
#         The brick to apply as activation for gates. If `None` a `Sigmoid`
#         brick is used.
#     dim : int
#         The dimension of the hidden state.
#     weights_init : object
#         The :class:`utils.NdarrayInitialization` object to initialize the
#         weight matrix with.
#     use_upgate_gate : bool
#         If True the update gates are used.
#     use_reset_gate : bool
#         If True the reset gates are used.
# 
# 
#     .. [1] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre,
#          Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk and Yoshua Bengio.
#          Learning Phrase Representations using RNN Encoder-Decoder
#          for Statistical Machine Translation. EMNLP 2014.
#     """
# 
#     @Brick.lazy_method
#     def __init__(self, activation, gate_activation, dim, weights_init,
#                  use_update_gate=True, use_reset_gate=True, **kwargs):
#         super(GatedRecurrent, self).__init__(**kwargs)
# 
#         if not activation:
#             activation = Identity()
#         if not gate_activation:
#             gate_activation = Sigmoid()
# 
#         self.__dict__.update(locals())
#         del self.self
#         del self.kwargs
# 
#     @property
#     def state2state(self):
#         return self.params[0]
# 
#     @property
#     def state2update(self):
#         return self.params[1]
# 
#     @property
#     def state2reset(self):
#         return self.params[2]
# 
#     def _allocate(self):
#         def new_param(name):
#             shared = shared_floatx_zeros((self.dim, self.dim))
#             shared.name = name
#             return shared
#         self.params.append(new_param('state2state'))
#         self.params.append(new_param('state2update')
#                            if self.use_update_gate else None)
#         self.params.append(new_param('state2reset')
#                            if self.use_reset_gate else None)
# 
#     def _initialize(self):
#         self.weights_init.initialize(self.state2state, self.rng)
#         if self.use_update_gate:
#             self.weights_init.initialize(self.state2update, self.rng)
#         if self.use_reset_gate:
#             self.weights_init.initialize(self.state2reset, self.rng)
# 
#     @Brick.recurrent_apply_method
#     def apply(self, inps, update_inps=None, reset_inps=None,
#               states=None, mask=None):
#         """Apply the gated recurrent transition.
# 
#         Parameters
#         ----------
#         states : Theano variable
#             The 2 dimensional matrix of current states in the shape
#             (batch_size, features). Required for `one_step` usage.
#         inps : Theano matrix of floats
#             The 2 dimensional matrix of inputs in the shape
#             (batch_size, features)
#         update_inps : Theano variable
#             The 2 dimensional matrix of inputs to the update gates in the shape
#             (batch_size, features). None when the update gates are not used.
#         reset_inps : Theano variable
#             The 2 dimensional matrix of inputs to the reset gates in the shape
#             (batch_size, features). None when the reset gates are not used.
#         mask : Theano variable
#             A 1D binary array in the shape (batch,) which is 1 if
#             there is data available, 0 if not. Assumed to be 1-s
#             only if not given.
# 
#         Returns
#         -------
#         output : Theano variable
#             Next states of the network.
#         """
#         if (self.use_update_gate != (update_inps is not None)) or \
#                 (self.use_reset_gate != (reset_inps is not None)):
#             raise ValueError("Configuration and input mismatch: You should "
#                              "provide inputs for gates if and only if the "
#                              "gates are on.")
# 
#         states_reset = states
# 
#         if self.use_reset_gate:
#             reset_values = self.gate_activation.apply(
#                 states.dot(self.state2reset) + reset_inps)
#             states_reset = states * reset_values
# 
#         next_states = self.activation.apply(
#             states_reset.dot(self.state2state) + inps)
# 
#         if self.use_update_gate:
#             update_values = self.gate_activation.apply(
#                 states.dot(self.state2update) + update_inps)
#             next_states = (next_states * update_values
#                            + states * (1 - update_values))
# 
#         if mask:
#             next_states = (mask[:, None] * next_states
#                            + (1 - mask[:, None]) * states)
# 
#         return next_states
# 
#     @apply.signature_method
#     def apply_signature(self, *args, **kwargs):
#         s = RecurrentApplySignature(
#             input_names=['mask', 'inps'], state_names=['states'],
#             output_names=['states'],
#             dims=dict(inps=self.dim, states=self.dim))
#         if self.use_update_gate:
#             s.input_names.append('update_inps')
#             s.dims['update_inps'] = self.dim
#         if self.use_reset_gate:
#             s.input_names.append('reset_inps')
#             s.dims['reset_inps'] = self.dim
#         s.forkable_input_names = s.input_names[1:]
#         return s
# 
# 
# class BidirectionalRecurrent(DefaultRNG):
#     @Brick.lazy_method
#     def __init__(self, dim, weights_init, activation=None, hidden_init=None,
#                  combine='concatenate', **kwargs):
#         super(BidirectionalRecurrent, self).__init__(**kwargs)
#         if hidden_init is None:
#             hidden_init = Constant(0)
#         self.__dict__.update(locals())
#         del self.self
#         self.children = [Recurrent(), Recurrent()]
# 
#     def _push_allocation_config(self):
#         for child in self.children:
#             for attr in ['dim', 'activation', 'hidden_init']:
#                 setattr(child, attr, getattr(self, attr))
# 
#     def _push_initialization_config(self):
#         for child in self.children:
#             child.weights_init = self.weights_init
# 
#     @Brick.apply_method
#     def apply(self, inp, mask):
#         forward = self.children[0].apply(inp, mask)
#         backward = self.children[1].apply(inp, mask, reverse=True)
#         output = tensor.concatenate([forward[-1], backward[-1]], axis=1)
#         return output
