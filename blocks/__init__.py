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

logging.basicConfig()
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
    rng : object, optional
        A `numpy.random.RandomState` object. This can be used by the layer
        to e.g. initialize parameters.
    output_mode : str, optional
        The output mode of this bock. Defaults to `'default'`.

    Attributes
    ----------
    tags : list of strings
        A list of class attributes which will be copied to the output
        variables' `tag` field, and hence passed on to the next block.
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

    Examples
    --------
    Two usage patterns of blocks are as follows:

    >>> x = theano.tensor.vector()
    >>> linear = Linear(5, 3, weights_init=IsotropicGaussian(),
                        biases_init=Constant(0))
    >>> linear.apply(x)
    block_linear_output_0

    More fine-grained control is also possible:

    >>> linear = Linear(4, 3, weights_init=IsotropicGaussian(),
                        biases_init=Constant(0))
    >>> linear.allocate()
    >>> linear.apply(x, initialize=False)
    block_linear_output_0
    >>> linear.initialize()

    """
    __metaclass__ = ABCMeta
    tags = ['params', 'monitor_channels', 'dim']
    supported_output_modes = ['default']

    def __init__(self, name=None, rng=None, output_mode='default'):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}_{}'.format(BLOCK_PREFIX, name)

        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        assert output_mode in self.supported_output_modes
        self.output_mode = output_mode

        self.allocated = False
        self.initialized = False

    def allocate(self):
        """Allocate shared variables for parameters.

        Based on the current configuration of this block, such as the
        number of inputs, the output mode, etc. create Theano shared
        variables to store the parameters. This method must be called
        *before* calling the block's :meth:`apply` method. After
        allocation, parameters are accesible through the :attr:`params`
        attribute.

        Raises
        ------
        ValueError
            If the state of this block is insufficient to determine the
            number of parameters or their dimensionality to be initialized.

        """
        self._allocate()
        self.allocated = True

    def _allocate(self):
        self.params = []

    def initialize(self):
        """Initialize parameters.

        Intialize parameters, such as weight matrices and biases.

        """
        if not self.allocated:
            self.allocate()
        self._initialize()
        self.initialized = True

    def _initialize(self):
        pass

    def apply(self, *states_below, **kwargs):
        """Apply Block operation to inputs.

        Parameters
        ----------
        states_below : list of Theano variables
            The inputs which to apply this block to.
        initialize : bool, optional
            If `True`, the values of the parameters will be initialized if
            they not are already. If `False`, the parameters will need to
            be initialized with a call to :meth:`initialize` (or by
            manually setting them) before running the compiled Theano
            function of this block. Defaults to `True`.

        Raises
        ------
        ValueError
            If the inputs are incompatible with this Block.

        Notes
        -----
        If the parameters of this block have not been allocated yet, the
        :meth:`allocate` method will be called in order to do so.

        """
        initialize = kwargs.get('initialize', True)
        if not self.initialized and initialize:
            self.initialize()
        elif not self.allocated:
            self.allocate()
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

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            b.set_value(self.biases_init.initialize(self.rng,
                                                    (self.output_dim,)))
        else:
            W, = self.params
        W.set_value(self.weights_init.initialize(
            self.rng, (self.input_dim, self.output_dim)))

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
