"""Bricks module

This defines the basic interface of bricks.
"""
from abc import ABCMeta
import logging

import numpy as np
from theano import tensor

from blocks.utils import pack, sharedX, unpack

BRICK_PREFIX = 'brick'
BRICK_SEPARATOR = '_'
DEFAULT_SEED = [2014, 10, 5]

logging.basicConfig()
logger = logging.getLogger(__name__)


class Brick(object):
    """A brick encapsulates Theano operations with parameters.

    A Brick is a group of parameterized Theano operations. Bricks can be
    considered connected, disjoint subsets of nodes in Theano's
    computiational graph.

    Parameters
    ----------
    name : str, optional
        The name of this brick. This can be used to filter the application
        of certain modifications by brick names. By default the brick
        receives the name of its class (lowercased). The name is expected
        to be unique within a block.
    rng : object, optional
        A `numpy.random.RandomState` object. This can be used by the brick
        to e.g. initialize parameters.
    initialize : bool, optional
        If `True` then the parameters of this brick will automatically be
        allocated and initialized by calls to the :meth:`allocate` and
        :meth:`initialize`. If `False` these methods need to be called
        manually after initializing. Defaults to `True`.

    Attributes
    ----------
    params : list of Theano shared variables
        After calling the :meth:`allocate` method this attribute will be
        populated with the shared variables storing this brick's
        parameters.
    allocated : bool
        True after :meth:`allocate` has been called, False otherwise.
    initialized : bool
        True after :meth:`initialize` has been called, False otherwise.

    Notes
    -----
    Brick implementations can override the :meth:`_setup` method, which
    will be called with the arguments passed to the :meth:`__init__`
    method, in order to initialize and configure their brick.

    The methods :meth:`_allocate` and :meth:`_initialize` need to be
    overridden if the brick needs to allocate shared variables and
    initialize their values in order to function.

    A brick can have any number of methods which apply the brick on Theano
    variables. These methods should be decorated with the
    :meth:`apply_method` decorator.

    Examples
    --------
    Two usage patterns of bricks are as follows:

    >>> x = theano.tensor.vector()
    >>> linear = Linear(5, 3, weights_init=IsotropicGaussian(),
                        biases_init=Constant(0))
    >>> linear.apply(x)
    brick_linear_output_0

    More fine-grained control is also possible:

    >>> linear = Linear(5, 3, weights_init=IsotropicGaussian(),
                        biases_init=Constant(0), initialize=False)
    >>> linear.allocate()
    >>> linear.apply(x)
    brick_linear_output_0
    >>> linear.initialize()

    """
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        rng = kwargs.pop('rng', None)
        initialize = kwargs.pop('initialize', True)

        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}{}{}'.format(BRICK_PREFIX, BRICK_SEPARATOR, name)
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        self._setup(*args, **kwargs)

        self.allocated = False
        self.initialized = False

        if initialize:
            self.initialize()

    def _setup(self):
        """Configure the brick.

        This is the setup method called by the :meth:`__init__`
        method. Derived classes should overwrite this method and perform
        the configuration of their brick here.

        .. todo ::
           Determine the conditions for calling this method a second time.
           It could give issues if the parameters have already been
           initialized. But maybe that's the user's responsibility,
           considering that this is a non-public method anyway.
        """
        pass

    @staticmethod
    def apply_method(apply):
        """Wraps methods that apply a brick to inputs in different ways.

        This decorator will provide some necessary pre- and post-processing
        of the Theano variables, such as tagging them with the brick that
        created them and naming them.

        Parameters
        ----------
        apply : method
            A method which takes Theano variablse as an input, and returns
            the output of the Brick

        Raises
        ------
        ValueError
            If the Brick has not been initialized yet.

        """
        def wrapped_apply(self, *states_below):
            if not self.allocated:
                raise ValueError("{}: parameters have not been allocated".
                                 format(self.__class__.__name__))
            states_below = list(states_below)
            for i, state_below in enumerate(states_below):
                states_below[i] = state_below.copy()
            outputs = pack(apply(self, *states_below))
            for output in outputs:
                output.tag.owner_brick = self
            return unpack(outputs)
        return wrapped_apply

    def allocate(self):
        """Allocate shared variables for parameters.

        Based on the current configuration of this brick, such as the
        number of inputs, the output mode, etc. create Theano shared
        variables to store the parameters. This method must be called
        *before* calling the brick's :meth:`apply` method. After
        allocation, parameters are accessible through the :attr:`params`
        attribute.

        Raises
        ------
        ValueError
            If the state of this brick is insufficient to determine the
            number of parameters or their dimensionality to be initialized.

        """
        self.params = []
        self._allocate()
        self.allocated = True

    def _allocate(self):
        pass

    def initialize(self):
        """Initialize parameters.

        Intialize parameters, such as weight matrices and biases.

        Notes
        -----
        If the brick has not allocated its parameters yet, this method will
        call the :meth:`allocate` method in order to do so.
        """
        if not self.allocated:
            self.allocate()
        self._initialize()
        self.initialized = True

    def _initialize(self):
        pass


class Linear(Brick):
    """A linear transformation with optional bias.

    Linear brick which applies a linear (affine) transformation by
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

    See also
    --------
    :class:`Brick`

    """
    def _setup(self, input_dim, output_dim, weights_init, biases_init=None,
               use_bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights_init = weights_init
        if use_bias and biases_init is None:
            raise ValueError
        self.biases_init = biases_init
        self.use_bias = use_bias

    def _allocate(self):
        self.params.append(sharedX(np.empty((0, 0))))
        if self.use_bias:
            self.params.append(sharedX(np.empty((0,))))

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng, (self.output_dim,))
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng,
                                     (self.input_dim, self.output_dim))

    @Brick.apply_method
    def apply(self, *states_below):
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


class Tanh(Brick):
    @Brick.apply_method
    def apply(self, *states_below):
        states = [tensor.tanh(state) for state in states_below]
        return states


class Softmax(Brick):
    @Brick.apply_method
    def apply(self, state_below):
        state = tensor.nnet.softmax(state_below)
        return state


class Cost(Brick):
    pass


class CrossEntropy(Cost):
    @Brick.apply_method
    def apply(self, y, y_hat):
        state = -(y * tensor.log(y_hat)).sum(axis=1).mean()
        return state
