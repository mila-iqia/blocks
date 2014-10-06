"""Blocks module

This defines the basic interface of blocks.
"""
from abc import ABCMeta, abstractmethod
import logging

import numpy as np
from theano import tensor

from blocks.utils import pack, sharedX, unpack

BLOCK_PREFIX = 'block'
DEFAULT_SEED = [2014, 10, 5]

logger = logging.getLogger(__name__)


class Block(object):
    """A Block, which encapsulates Theano operations with parameters.

    A Block is a group of parameterized Theano operations. That is,
    it defines a series of Theano operations which can rely on a set of
    parameters.

    Parameters
    ----------
    name : str, optional
        The name of this block. This can be used to filter the application
        of certain modifications by block names. By default the block
        receives the name of its class (lowercased).
    output_mode : str, optional
        The output mode of this bock. Defaults to `'default'`.

    Attributes
    ----------
    tags : list of strings
        A list of class attributes which will be copied to the output
        variables `tag` field, and hence passed on to the next block.
    supported_output_modes : list of strings
        A list of output modes supported by this block.
    params : list of Theano shared variables
        After calling the :meth:`allocate` method this attribute will be
        populated with the shared variables storing this block's
        parameters.
    allocated : bool
        True after :meth:`allocate` has been called, False otherwise.
    initialized : bool
        True after :meth:`initialize` has been called, False otherwise.

    """
    __metaclass__ = ABCMeta
    tags = ['params', 'monitor_channels', 'dim']
    supported_output_modes = ['default']

    def __init__(self, name=None, output_mode='default'):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}_{}'.format(BLOCK_PREFIX, name)

        assert output_mode in self.supported_output_modes
        self.output_mode = output_mode

        self.allocated = False
        self.initialized = False

    def allocate(self):
        """Allocate shared variables for parameters.

        Based on the current configuration of this block, such as the
        number of inputs, the output mode, etc. create Theano shared
        variables to store the parameters. This method must be called
        *before* calling the block's :meth:`apply` method.

        Raises
        ------
        ValueError
            If the state of this block is insufficient to determine how to
            allocate parameters.

        """
        self._allocate()
        self.allocated = True

    def _allocate(self):
        self.params = []

    def initialize(self, rng):
        """Initialize parameters.

        Intialize parameters, such as weight matrices and biases.

        Parameters
        ----------
        rng : object
            A `numpy.random.RandomState` object which will be used to
            initialize the parameters. This needs to be passed in order to
            ensure reproducible results.

        """
        self._initialize(rng)
        self.initialized = True

    def _initialize(self):
        pass

    def apply(self, *states_below):
        """Apply Block operation to inputs.

        Parameters
        ----------
        states_below : list of Theano variables
            The inputs which to apply this block to.

        Raises
        ------
        ValueError
            If the parameters have not been allocated yet using the
            :meth:`allocate` method.
        ValueError
            If the inputs are incompatible with this Block.

        """
        # Copy the inputs and tag them
        states_below = list(states_below)
        for i, state_below in enumerate(states_below):
            states_below[i] = state_below.copy()
            # TODO Copy any required tags here
            states_below[i].name = self.name + '_input_{}'.format(i)

        # Perform the apply
        states = pack(self._apply(*states_below))

        # Tag output variables
        for i, state in enumerate(states):
            # TODO Apply any tags here
            states[i].name = self.name + '_output_{}'.format(i)
        return unpack(states)

    @abstractmethod
    def _apply(self, *states_below):
        return states_below


class Linear(Block):
    """A linear transformation with optional bias.

    Linear block which applies a linear (affine) transformation by
    multiplying the input with a weight matrix. Optionally a bias is added.

    Parameters
    ----------
    input_dim : int
        The dimension of the input.
    output_dim : int
        The dimension of the output.
    weights_init : object
        A `NdarrayInitialization` instance which will be used by the
        :meth:`initialize` method to initialize the weight matrix.
    biases_init : object, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required when `use_bias` is `True`.
    use_bias : bool, optional
        Whether to use a bias. Defaults to `True`.

    Notes
    -----

    A linear transformation with bias is a matrix multiplication followed
    by a vector summation.

    .. math:: f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}

    """
    def __init__(self, input_dim, output_dim, weights_init, biases_init=None,
                 use_bias=True, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights_init = weights_init
        if use_bias and biases_init is None:
            raise ValueError
        self.biases_init = biases_init
        self.use_bias = use_bias

    def _allocate(self):
        self.params = [sharedX(np.empty((0, 0)))]
        if self.use_bias:
            self.params.append(sharedX(np.empty((0,))))

    def _initialize(self, rng):
        if self.use_bias:
            W, b = self.params
            b.set_value(self.biases_init.initialize(rng, (self.output_dim,)))
        else:
            W, = self.params
        W.set_value(self.weights_init.initialize(
            rng, (self.input_dim, self.output_dim)))

    def _apply(self, *states_below):
        if self.use_bias:
            W, b = self.params
        else:
            W, = self.params
        states = []
        for state_below in states_below:
            state = tensor.dot(state_below, W)
            if self.use_bias:
                state += b
            states.append(state)
        return states


class Tanh(Linear):
    def _apply(self, *states_below):
        states = [tensor.tanh(state) for state in
                  super(Tanh, self)._apply(*states_below)]
        return states


class Softmax(Block):
    def _apply(self, state_below):
        state = tensor.nnet.softmax(state_below)
        return state


class Cost(Block):
    pass


class CrossEntropy(Cost):
    def _apply(self, y, y_hat):
        state = -(y * tensor.log(y_hat)).sum(axis=1).mean()
        return state
