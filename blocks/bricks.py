"""Bricks module

This defines the basic interface of bricks.
"""
from abc import ABCMeta
import inspect
import logging

import numpy as np
from theano import tensor

from blocks.utils import pack, reraise_as, sharedX, unpack
from blocks import SEPARATOR, DEFAULT_SEED

BRICK_PREFIX = 'brick'

logging.basicConfig()
logger = logging.getLogger(__name__)


class Undefined(object):
    """The class of the `UNDEF` value.

    The sole purpose of this class is to create an object `UNDEF`. This
    object has semantics of an undefined configuration setting.

    `UNDEF` always evaluates to `False`, just like `None`.

    """
    def __nonzero__(self):
        return False

    def __repr__(self):
        return 'UNDEF'

UNDEF = Undefined()


class LazyInitializationError(ValueError):
    def __init__(self, message, arg):
        super(Exception, self).__init__(message)
        self.arg = arg


class Brick(object):
    """A brick encapsulates Theano operations with parameters.

    A Brick is a group of parameterized Theano operations. Bricks can be
    considered connected, disjoint subsets of nodes in Theano's
    computiational graph.

    Bricks support lazy initialization. This means that bricks can be
    initialized by calling :meth:`__init__` with only a subset of the
    required arguments. These settings must then be set by hand later
    before calling the methods that require them (such as :meth:`allocate`,
    :meth:`initialize`), and application methods.

    To turn off lazy initialization in general set `Brick.lazy_init =
    False`.

    By default, Bricks will try to initialize their parameters when being
    applied for the first time (eager application). To turn this off and
    enable "lazy application" for all Bricks set `Bricks.lazy_apply` to
    `True`.

    Parameters
    ----------
    name : str, optional
        The name of this brick. This can be used to filter the application
        of certain modifications by brick names. By default, the brick
        receives the name of its class (lowercased). The name is expected
        to be unique within a block.
    rng : object, optional
        A `numpy.random.RandomState` object. This can be used by the brick
        to e.g. initialize parameters.

    Attributes
    ----------
    lazy_init : bool
        `True` by default, allowing blocks to be configured in a lazy
        manner. This means that not all parameters have to be passed to the
        constructor by default. Parameters not passed will be set to
        :class:`Uninitialized` instead. Set to `False` to disable lazy
        initialization.
    lazy_apply : bool
        `False` by default, making blocks eager which means they will try
        to initialize their parameters before being applied. Setting this
        to `True` means that :meth:`initialize` needs to be called manually
        before running the compiled Theano function.
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
    Brick implementations *must* call the :meth:`__init__` constructor of
    their parent using `super(BlockImplementation,
    self).__init__(**kwargs)`.

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
    A simple usage of a brick is as follows:

    >>> import theano
    ... from blocks.initialization import IsotropicGaussian, Constant
    ... x = theano.tensor.vector()
    ... linear = Linear(5, 3, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    ... linear.apply(x)

    By default, bricks have lazy initialization enabled.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    ... linear = Linear(input_dim=5, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    ... linear.output_dim = 3
    ... linear.apply(x)

    Bricks also support lazy application, which means that they will not
    try to initialize their parameters before building the Theano
    expression. This allows you to postpone configuring your block even
    further.

    >>> Brick.lazy_apply = True
    ... linear = Linear(5, weights_init=IsotropicGaussian(),
    ...                 use_bias=False)
    ... linear.apply(x)
    ... layer.output_dim = 3
    ... linear.initialize()

    """
    __metaclass__ = ABCMeta
    lazy_init = True
    lazy_apply = False

    def __init__(self, name=None, rng=None):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}{}{}'.format(BRICK_PREFIX, SEPARATOR, name)
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        self.allocated = False
        self.initialized = False

        if not self.lazy_init:
            self.initialize()

    def __getattribute__(self, name):
        value = super(Brick, self).__getattribute__(name)
        if value is UNDEF:
            raise LazyInitializationError(
                "{}: {} has not been initialized".
                format(self.__class__.__name__, name), name)
        return value

    @staticmethod
    def apply_method(func):
        """Wraps methods that apply a brick to inputs in different ways.

        This decorator will provide some necessary pre- and post-processing
        of the Theano variables, such as tagging them with the brick that
        created them and naming them.

        By default, application methods are eager, and will try to
        initialize the parameters of the brick. If the configuration of
        this brick is insufficient to do so, an exception will be raised
        (usually a :class:`LazyInitizializationError`). To prevent this
        from happening, one can set `Brick.lazy_apply = True`, in which
        case the parameters need to be initialized manually with a call to
        :meth:`initialize` before using the resulting Theano function.

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
        def wrapped_apply(self, *inputs, **kwargs):
            if not self.allocated:
                self.allocate()
            if not self.initialized and not self.lazy_apply:
                try:
                    self.initialize()
                except LazyInitializationError as e:
                    reraise_as(LazyInitializationError(
                        "`{}`: Unable to initialize parameters because of "
                        "missing configuration (`{}`). Either set this "
                        "configuration value, or set the `lazy_apply` "
                        "attribute to `True` to prevent `{}` from trying to "
                        "initialize the parameters.".format(
                            self.__class__.__name__, e.arg, func.__name__),
                        e.arg))
            inputs = list(inputs)
            for i, inp in enumerate(inputs):
                inputs[i] = inp.copy()
            outputs = pack(func(self, *inputs, **kwargs))
            for output in outputs:
                # TODO Tag with dimensions, axes, etc. for error-checking
                output.tag.owner_brick = self
            return unpack(outputs)
        return wrapped_apply

    @staticmethod
    def lazy(func):
        """Makes the initialization lazy.

        Any positional argument not given will be set to UNDEF. Positional
        arguments can also be given as keyword arguments.

        Parameters
        ----------
        func : method
            The __init__ method to make lazy.

        Examples
        --------

        >>> class SomeBrick(Brick):
        ...     @Brick.lazy
        ...     def __init__(self, a, b, c='c', d=None):
        ...         print a, b, c, d
        >>> brick = SomeBrick('a')
        a UNDEF c UNDEF
        >>> brick = SomeBrick(d='d', b='b')
        UNDEF b c d
        >>> Brick.lazy_init = False
        >>> brick = SomeBrick('a')
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: __init__() takes at least 3 arguments (2 given)
        >>> brick = SomeBrick('a', 'b')
        a b c None

        """
        arg_spec = inspect.getargspec(func)
        arg_names = arg_spec.args[1:]
        defaults = arg_spec.defaults

        def init(self, *args, **kwargs):
            if not self.lazy_init:
                return func(self, *args, **kwargs)
            # Fill any missing positional arguments with UNDEF
            args = args + (UNDEF,) * (len(arg_names) - len(defaults) -
                                      len(args))

            # Check if positional arguments were passed as keyword arguments
            args = list(args)
            for i, arg_name in enumerate(arg_names[:-len(defaults)]):
                if arg_name in kwargs:
                    if args[i] is not UNDEF:
                        raise ValueError("Positional argument `{}` at index "
                                         "{} was also given as a keyword".
                                         format(arg_name, i))
                    args[i] = kwargs.pop(arg_name)

            # Set keyword arguments not given, with default None, to UNDEF too
            for arg_name, default in zip(arg_names[-len(defaults):], defaults):
                if arg_name not in kwargs and default is None:
                    kwargs[arg_name] = UNDEF

            return func(self, *args, **kwargs)
        return init

    def allocate(self):
        """Allocate shared variables for parameters.

        Based on the current configuration of this brick, such as the
        number of inputs, the output mode, etc. create Theano shared
        variables to store the parameters. This method will be called by
        every application method. After allocation, parameters are
        accessible through the :attr:`params` attribute.

        Raises
        ------
        ValueError
            If the configuration of this brick is insufficient to determine
            the number of parameters or their dimensionality to be
            initialized.

        Notes
        -----
        This method already sets the :attr:`params` attribute to an empty
        list, so implementations of :meth:`_allocate` can simply append to
        this list as needed. This is to make sure that parameters are
        completely reset by calls to this method.

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

    See also
    --------
    :class:`Brick`

    """
    @Brick.lazy
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
        if self.use_bias:
            W, b = self.params
        else:
            W, = self.params
        output = tensor.dot(inp, W)
        if self.use_bias:
            output += b
        return output


class Tanh(Brick):
    @Brick.apply_method
    def apply(self, inp):
        output = tensor.tanh(inp)
        return output


class Softmax(Brick):
    @Brick.apply_method
    def apply(self, inp):
        output = tensor.nnet.softmax(inp)
        return output


class Cost(Brick):
    pass


class CrossEntropy(Cost):
    @Brick.apply_method
    def apply(self, y, y_hat):
        cost = -(y * tensor.log(y_hat)).sum(axis=1).mean()
        return cost
