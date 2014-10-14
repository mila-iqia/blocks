"""Bricks module

This defines the basic interface of bricks.
"""
from abc import ABCMeta
from functools import wraps
import inspect
import logging

import numpy as np
from theano import tensor

from blocks.utils import pack, reraise_as, sharedX, unpack

BRICK_PREFIX = 'brick_'
INPUT_SUFFIX = '_input'
OUTPUT_SUFFIX = '_output'
DEFAULT_SEED = [2014, 10, 5]
PARAM_OWNER_TAG = 'param_owner'

logging.basicConfig()
logger = logging.getLogger(__name__)


class Parameters(list):
    """Automatically tag parameters with the bricks that they belong to.

    If parameters are shared by bricks, both of them will show up in the
    owners list.

    Parameters
    ----------
    owner : Brick
        The :class:`Brick` that owns this set of parameters.
    params : list of Theano variables, optional
        A initial set of parameters.

    .. todo ::

       We could consider quietly tagging the shared variables here,
       whenever a normal variable is passed.
    """
    def __init__(self, owner, params=None):
        self.owner = owner
        if params is None:
            params = []
        for param in params:
            getattr(param.tag, PARAM_OWNER_TAG, []).append(self.owner)
        super(Parameters, self).__init__(params)

    def __setitem__(self, key, param):
        getattr(param.tag, PARAM_OWNER_TAG, []).append(self.owner)
        super(Parameters, self).__setitem__(key, param)

    def append(self, param):
        getattr(param.tag, PARAM_OWNER_TAG, []).append(self.owner)
        super(Parameters, self).append(param)

    def extend(self, params):
        for param in params:
            getattr(param.tag, PARAM_OWNER_TAG, []).append(self.owner)
        super(Parameters, self).extend(params)

    def insert(self, index, param):
        getattr(param.tag, PARAM_OWNER_TAG, []).append(self.owner)
        super(Parameters, self).insert(index, param)


class Brick(object):
    """A brick encapsulates Theano operations with parameters.

    A brick goes through the following stages:

    1. Construction: The call to :meth:`__init__` constructs a
       :class:`Brick instance with a name and creates any child bricks as
       well.
    2. Allocation: This allocates the shared Theano variables required for
       the parameters. In order to do so, some configuration might be
       required in order to determine the dimensionality of the shared
       variables.
    3. The following can be done in either order:

       a) Application: By applying the brick to a set of Theano
          variables a part of the computational graph of the final model is
          constructed.
       b) Initialization: This sets the initial values of the parameters,
          which is needed to call the final compiled Theano function.

    At each different stage, a brick might need a certain set of
    configuration settings. All of these settings can be passed to the
    :meth:`__init__` constructor. However, by default many bricks support
    *lazy initialization*. This means that the configuration settings can
    be set later.

    .. note::

       Some arguments to :meth:`__init__` are *always* required, even when
       lazy initialization is enabled. Other arguments must be given before
       calling :meth:`allocate`, while others yet only need to be given in
       order to call :meth:`initialize`. Always read the documentation of
       each brick carefully.

    Trying to initialize or apply a brick will automatically result in
    :meth:`allocate` being called if it wasn't called before.

    Lazy initialization can be turned off by setting ``Brick.lazy =
    False``. In this case, there is no need to call :meth:`initialize`
    manually anymore, but all the configuration must be passed to the
    :meth:`__init__` method.

    Parameters
    ----------
    name : str, optional
        The name of this brick. This can be used to filter the application
        of certain modifications by brick names. By default, the brick
        receives the name of its class (lowercased).

    Attributes
    ----------
    lazy : bool
        ``True`` by default. When bricks are lazy, not all configuration
        needs to be provided to the constructor, allowing it to be set in
        another way after construction. Many parts of the library rely on
        this behaviour. However, it does require a separate call to
        :meth:`initialize`. If set to ``True`` on the other hand, bricks
        will be ready to run after construction.
    params : list of Theano shared variables
        After calling the :meth:`allocate` method this attribute will be
        populated with the shared variables storing this brick's
        parameters. Any variable added to this list will automatically be
        tagged with this brick as an owner.
    allocated : bool
        ``False`` if :meth:`allocate` has not been called yet, or if the
        configuration of this brick has changed since the last call to
        :meth:`allocate`. ``True`` otherwise.
    initialized : bool
        ``False`` if :meth:`allocate` has not been called yet, or if the
        configuration of this brick has changed since the last call to
        :meth:`allocate`. ``True`` otherwise.
    allocation_config_pushed : bool
        ``False`` if :meth:`allocate` or :meth:`push_allocation_config`
        haven't been called yet, or if the configuration of this brick has
        changed since the last call. ``True`` otherwise.
    initialization_config_pushed : bool
        ``False`` if :meth:`initialize` or :meth:`push_initialization_config`
        haven't been called yet, or if the configuration of this brick has
        changed since the last call. ``True`` otherwise.
    allocation_config : list of strings
        These are the attributes that define the configuration required to
        call :meth:`allocate`. Changing them will cause :attr:`allocated`
        to be set to ``False`` again and :meth:`allocate` to be called next
        time :meth:`initialize` or an application method is called, causing
        the parameters to be reset. Note that the parameters of child
        bricks might also be re-allocated.
    initialization_config : list of strings
        These are the attributes that define the configuration required to
        call :meth:`initialize`. Changing them will cause
        :attr:`initialization` to be set to ``False`` again and will cause
        :meth:`initialize` to be called next time this brick is applied, if
        the brick is not lazy.

    Notes
    -----
    Brick implementations *must* call the :meth:`__init__` constructor of
    their parent using `super(BlockImplementation,
    self).__init__(**kwargs)` at the *beginning* of the overriding
    `__init__`.

    The methods :meth:`_allocate` and :meth:`_initialize` need to be
    overridden if the brick needs to allocate shared variables and
    initialize their values in order to function.

    A brick can have any number of methods which apply the brick on Theano
    variables. These methods should be decorated with the
    :meth:`apply_method` decorator.

    To provide support for lazy initialization, apply the :meth:`lazy`
    decorator to the :meth:`__init__` method.

    Examples
    --------
    By default, bricks have lazy initialization enabled.

    >>> import theano
    ... from blocks.initialization import IsotropicGaussian, Constant
    ... linear = Linear(input_dim=5, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    ... x = theano.tensor.vector()
    ... linear.apply(x)  # Calls linear.allocate() automatically
    ... linear.output_dim = 3
    ... linear.initialize()

    In simple cases, eager bricks are easier to deal with.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    ... Brick.lazy = False
    ... linear = Linear(5, 3, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    ... linear.apply(x)

    """
    __metaclass__ = ABCMeta
    lazy = True
    allocation_config = ['name']
    initialization_config = []

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}{}'.format(BRICK_PREFIX, name)

        self.children = []

        self.allocated = False
        self.allocation_config_pushed = False
        self.initialized = False
        self.initialization_config_pushed = False

    def __setattr__(self, name, value):
        # If config changes, initialize/allocate must be called again
        if hasattr(self, name) and getattr(self, name) != value:
            if name in self.allocation_config:
                # TODO Add warning about resetting of parameters?
                self.allocated = False
                self.allocation_config_pushed = False
            if name in self.initialization_config:
                # TODO Add warning about need for re-initialization?
                self.initialized = False
                self.initialization_config_pushed = False
        super(Brick, self).__setattr__(name, value)

    @property
    def params(self):
        """The parameters of this brick.

        Notes
        -----
        Setting the parameters will set :attr:`allocated` to ``True``,
        which allows you to set the parameters manually as well e.g.

        >>> brick = Brick()
        >>> brick.params = Parameters(brick)
        >>> brick.allocated
        True

        However, keep in mind that this could prevent child bricks from
        being correctly allocated.

        .. warning::

           If a normal list is passed instead of a :class:`Parameters`
           instance , the parameters won't automatically be tagged as
           belonging to this brick.

        """
        return self._params

    @params.setter
    def params(self, params):
        self._params = params
        self.allocated = True

    @staticmethod
    def apply_method(func):
        """Wraps methods that apply a brick to inputs in different ways.

        This decorator will provide some necessary pre- and post-processing
        of the Theano variables, such as tagging them with the brick that
        created them and naming them.

        Application methods will allocate the brick parameters with a call
        :meth:`allocate` if they have not been allocated already.

        Parameters
        ----------
        func : method
            A method which takes Theano variables as an input, and returns
            the output of the Brick

        Raises
        ------
        LazyInitializationError
            If parameters needed to perform the application of this brick
            have not been provided yet.

        """
        @wraps(func)
        def wrapped_apply(self, *inputs, **kwargs):
            if not self.allocated:
                self.allocate()
            if not self.initialized and not self.lazy:
                self.initialize()
            inputs = list(inputs)
            for i, inp in enumerate(inputs):
                inputs[i] = inp.copy()
                inputs[i].tag.owner = self
                inputs[i].name = self.name + INPUT_SUFFIX
            outputs = pack(func(self, *inputs, **kwargs))
            for i, output in enumerate(outputs):
                # TODO Tag with dimensions, axes, etc. for error-checking
                outputs[i] = output.copy()
                outputs[i].tag.owner = self
                outputs[i].name = self.name + OUTPUT_SUFFIX
            return unpack(outputs)
        return wrapped_apply

    @staticmethod
    def lazy_method(func):
        """Makes the initialization lazy.

        Any positional argument not given will be set to ``None``.
        Positional arguments can also be given as keyword arguments.

        .. todo ::

           Use ``UNDEF`` or ``None`` as a default?


        Parameters
        ----------
        func : method
            The __init__ method to make lazy.

        Examples
        --------

        >>> class SomeBrick(Brick):
        ...     @Brick.lazy_method
        ...     def __init__(self, a, b, c='c', d=None):
        ...         print a, b, c, d
        >>> brick = SomeBrick('a')
        a None c None
        >>> brick = SomeBrick(d='d', b='b')
        None b c d
        >>> Brick.lazy = False
        >>> brick = SomeBrick('a')
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: __init__() takes at least 3 arguments (2 given)
        >>> Brick.lazy = True  # Reset for other doctests

        """
        arg_spec = inspect.getargspec(func)
        arg_names = arg_spec.args[1:]
        defaults = arg_spec.defaults

        def init(self, *args, **kwargs):
            if not self.lazy:
                return func(self, *args, **kwargs)
            # Fill any missing positional arguments with UNDEF
            args = args + (None,) * (len(arg_names) - len(defaults) -
                                     len(args))

            # Check if positional arguments were passed as keyword arguments
            args = list(args)
            for i, arg_name in enumerate(arg_names[:-len(defaults)]):
                if arg_name in kwargs:
                    if args[i] is not None:
                        raise ValueError
                    args[i] = kwargs.pop(arg_name)

            return func(self, *args, **kwargs)
        return init

    def allocate(self):
        """Allocate shared variables for parameters.

        Based on the current configuration of this :class:`Brick` create
        Theano shared variables to store the parameters.  After allocation,
        parameters are accessible through the :attr:`params` attribute.

        This method calls the :meth:`allocate` method of all children
        first, allowing the :meth:`_allocate` method to override the
        parameters of the children if needed.

        Raises
        ------
        ValueError
            If the configuration of this brick is insufficient to determine
            the number of parameters or their dimensionality to be
            initialized.

        Notes
        -----
        This method sets the :attr:`params` attribute to an empty
        :class:`Parameters` instance. This is in order to ensure that calls
        to this method completely reset the parameters.

        .. warning ::

           Always ``append`` to this list (or ``extend``, ``insert``,
           ``set``), but never use ``self.params = [...]``. This will
           disable the automatic tagging of parameters.

        """
        if not self.allocation_config_pushed:
            self.push_allocation_config()
        for child in self.children:
            try:
                child.allocate()
            except:
                self.allocation_config_pushed = False
                raise
        self.params = Parameters(self)  # This sets self.allocated to True
        self._allocate()

    def _allocate(self):
        """Brick implementation of parameter initialization.

        Implement this if your brick needs to allocate its parameters.

        .. warning::

           This method should never be called directly. Call
           :meth:`initialize` instead.

        """
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
        if not self.initialization_config_pushed:
            self.push_initialization_config()
        for child in self.children:
            try:
                child.initialize()
            except:
                self.initialization_config_pushed = False
                raise
        try:
            self._initialize()
        except Exception:
            if self.lazy:
                reraise_as("Lazy initialization is enabled, so please make "
                           "sure you have set all the required configuration "
                           "for this method call.")
            else:
                raise
        self.initialized = True

    def _initialize(self):
        """Brick implementation of parameter initialization.

        Implement this if your brick needs to initialize its parameters.

        .. warning::

           This method should never be called directly. Call
           :meth:`initialize` instead.

        """
        pass

    def push_allocation_config(self):
        self._push_allocation_config()
        self.allocation_config_pushed = True

    def _push_allocation_config(self):
        """Brick implementation of configuring child before allocation.

        Implement this if your brick needs to set the configuration of its
        children before allocation.

        .. warning::

           This method should never be called directly. Call
           :meth:`push_allocation_config` instead.

        """
        pass

    def push_initialization_config(self):
        self._push_initialization_config()
        self.initialization_config_pushed = True

    def _push_initialization_config(self):
        """Brick implementation of configuring child before initialization.

        Implement this if your brick needs to set the configuration of its
        children before initialization.

        .. warning::

           This method should never be called directly. Call
           :meth:`push_initialization_config` instead.

        """
        pass


class DefaultRNG(Brick):
    """A mixin class for Bricks which need a RNG to initialize.

    Parameters
    ----------
    rng : object
        A ``numpy.RandomState`` instance.

    Attributes
    ----------
    rng : object
        If the RNG has been set, return it. Otherwise, return a RNG with a
        default seed which can be set at a module level using
        ``blocks.bricks.DEFAULT_SEED = seed``.

    Notes
    -----
    Put this class first in the list of base classes to make sure that
    it gets called instead of the :class:`Brick` class's :meth:`__init__`
    method.

    """
    def __init__(self, rng=None, **kwargs):
        self.rng = rng
        super(DefaultRNG, self).__init__(**kwargs)

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            return np.random.RandomState(DEFAULT_SEED)

    @rng.setter
    def rng(self, rng):
        self._rng = rng


class Linear(DefaultRNG):
    """A linear transformation with optional bias.

    Linear brick which applies a linear (affine) transformation by
    multiplying the input with a weight matrix. Optionally a bias is added.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`initialize`.
    output_dim : int
        The dimension of the output. Required by :meth:`initialize`.
    weights_init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by :meth:`initialize`.
    biases_init : object, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`initialize` when `use_bias` is
        `True`.
    use_bias : bool, optional
        Whether to use a bias. Defaults to `True`.

    Notes
    -----

    A linear transformation with bias is a matrix multiplication followed
    by a vector summation.

    .. math:: f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}

    """
    @Brick.lazy_method
    def __init__(self, input_dim, output_dim, weights_init,
                 biases_init=None, use_bias=True, **kwargs):
        self.__dict__.update(locals())
        del self.self
        super(Linear, self).__init__(**kwargs)

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
    def apply(self, inp):
        """Apply the linear transformation.

        Parameters
        ----------
        inp : Theano variable
            The input on which to apply the transformation

        Returns
        -------
        output : Theano variable
            The transformed input plus optional bias

        """
        if self.use_bias:
            W, b = self.params
        else:
            W, = self.params
        output = tensor.dot(inp, W)
        if self.use_bias:
            output += b
        return output


def _activation_factory(name, activation):
    """Class factory for Bricks which perform simple Theano calls."""
    class Activation(Brick):
        """Element-wise application of {0} function.

        Parameters
        ----------
        inp : Theano variable
            The Theano variable on which to apply the {0} function..

        Returns
        -------
        output : Theano variable
            The Theano variable with the {0} function applied.

        """
        @Brick.apply_method
        def apply(self, inp):
            """Apply the {0} function element-wise.

            Parameters
            ----------
            inp : Theano variable
                Theano variable to apply {0} to, element-wise.

            Returns
            -------
            output : Theano variable
                The input with the activation function applied.

            """
            output = activation(inp)
            return output
    Activation.__name__ = name
    Activation.__doc__ = Activation.__doc__.format(name.lower())
    Activation.apply.__func__.__doc__ = \
        Activation.apply.__func__.__doc__.format(name.lower())
    return Activation

Tanh = _activation_factory('Tanh', tensor.tanh)
Sigmoid = _activation_factory('Sigmoid', tensor.nnet.sigmoid)
Softmax = _activation_factory('Softmax', tensor.nnet.softmax)


class Cost(Brick):
    pass


class CostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (batch, features).
    """
    __metaclass__ = ABCMeta

    @Brick.apply_method
    def apply(self, y, y_hat):
        self.cost_matrix._raw(y, y_hat).sum(axis=1).mean()


class BinaryCrossEntropy(CostMatrix):
    @Brick.apply_method
    def cost_matrix(self, y, y_hat):
        cost = tensor.nnet.binary_crossentropy(y_hat, y)
        return cost


class AbsoluteError(CostMatrix):
    @Brick.apply_method
    def cost_matrix(self, y, y_hat):
        cost = tensor.abs(y - y_hat)
        return cost


class SquaredError(CostMatrix):
    @Brick.apply_method
    def apply(self, y, y_hat):
        cost = tensor.sqr(y - y_hat)
        return cost


class MLP(DefaultRNG):
    """A simple multi-layer perceptron

    Parameters
    ----------
    activations : bricks or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`initialize`.
    weights_init : :class:`utils.NdarrayInitialization`
        The initialization scheme to initialize all the weights with.
    biases_init : :class:`utils.NdarrayInitialization`
        The initialization scheme to initialize all the biases with.
    use_bias : bool
        Whether or not to use biases.

    Notes
    -----
    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP([Tanh(), None], [30, 20, 10],
    ...           IsotropicGaussian(), Constant(0))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """
    initialization_config = ['dims', 'weights_init', 'use_bias', 'biases_init']

    @Brick.lazy_method
    def __init__(self, activations, dims, weights_init, biases_init=None,
                 use_bias=True, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        self.children = (self.linear_transformations +
                         [activation for activation in activations
                          if activation is not None])
        self.__dict__.update(locals())
        del self.self

    def _push_initialization_config(self):
        assert len(self.dims) - 1 == len(self.linear_transformations)
        for input_dim, output_dim, layer in zip(self.dims[:-1], self.dims[1:],
                                                self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            for attr in ['weights_init', 'use_bias', 'biases_init']:
                setattr(layer, attr, getattr(self, attr))

    @Brick.apply_method
    def apply(self, inp):
        output = inp
        for activation, linear in zip(self.activations,
                                      self.linear_transformations):
            if activation is None:
                output = linear.apply(output)
            else:
                output = activation.apply(linear.apply(output))
        return output
