"""The interface of bricks and some simple implementations."""
import inspect
import logging
from abc import ABCMeta
from collections import OrderedDict
from functools import partial
from itertools import chain

import numpy
from six import add_metaclass
from theano import tensor

from blocks.utils import (pack, repr_attrs, reraise_as, shared_floatx_zeros,
                          unpack)

DEFAULT_SEED = [2014, 10, 5]

logger = logging.getLogger(__name__)


class _Brick(ABCMeta):
    """Replace application methods with :class:`Application` instances."""
    def __call__(cls, *args, **kwargs):
        brick = super(_Brick, cls).__call__(*args, **kwargs)
        applications = {}
        properties = {}
        delegates = {}

        for key in dir(brick):
            value = getattr(brick, key)
            # Replace the application methods by Application instances
            if hasattr(value, 'application_signature'):
                application = Application(value)
                setattr(brick, key, application)
                applications[value.__func__] = application

            # Collect the application properties and delegates
            if hasattr(value, 'application_property'):
                application_function, name = getattr(value,
                                                     'application_property')
                properties[application_function] = name, value.__func__
            if hasattr(value, 'application_delegate'):
                application_function = getattr(value, 'application_delegate')
                delegates[application_function] = partial(value.__func__,
                                                          brick)

        # Attach the properties and delegates to the Application instance
        for application_function, (name, func) in properties.items():
            applications[application_function].properties[name] = func
        for application_function, func in delegates.items():
            applications[application_function].delegate = func
        return brick


@add_metaclass(_Brick)
class Brick(object):
    """A brick encapsulates Theano operations with parameters.

    A brick goes through the following stages:

    1. Construction: The call to :meth:`__init__` constructs a
       :class:`Brick` instance with a name and creates any child bricks as
       well.
    2. Allocation of parameters:

       a) Allocation configuration of children: The
          :meth:`push_allocation_config` method configures any children of
          this block.
       b) Allocation: The :meth:`allocate` method allocates the shared
          Theano variables required for the parameters. Also allocates
          parameters for all children.

    3. The following can be done in either order:

       a) Application: By applying the brick to a set of Theano
          variables a part of the computational graph of the final model is
          constructed.
       b) The initialization of parameters:

          1. Initialization configuration of children: The
             :meth:`push_initialization_config` method configures any
             children of this block.
          2. Initialization: This sets the initial values of the
             parameters by a call to :meth:`initialize`, which is needed
             to call the final compiled Theano function.  Also initializes
             all children.

    Not all stages need to be called explicitly. Step 3(a) will
    automatically allocate the parameters if needed. Similarly, step
    3(b.2) and 2(b) will automatically perform steps 3(b.1) and 2(a) if
    needed. They only need to be called separately if greater control is
    required. The only two methods which always need to be called are an
    application method to construct the computational graph, and the
    :meth:`initialize` method in order to initialize the parameters.

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
    name : str
        The name of this brick.
    lazy : bool
        ``True`` by default. When bricks are lazy, not all configuration
        needs to be provided to the constructor, allowing it to be set in
        another way after construction. Many parts of the library rely on
        this behaviour. However, it does require a separate call to
        :meth:`initialize`. If set to ``False`` on the other hand, bricks
        will be ready to run after construction.
    params : list of Theano shared variables
        After calling the :meth:`allocate` method this attribute will be
        populated with the shared variables storing this brick's
        parameters.
    children : list of bricks
        The children of this brick.
    allocated : bool
        ``False`` if :meth:`allocate` has not been called yet. ``True``
        otherwise.
    initialized : bool
        ``False`` if :meth:`allocate` has not been called yet. ``True``
        otherwise.
    allocation_config_pushed : bool
        ``False`` if :meth:`allocate` or :meth:`push_allocation_config`
        hasn't been called yet. ``True`` otherwise.
    initialization_config_pushed : bool
        ``False`` if :meth:`initialize` or
        :meth:`push_initialization_config` hasn't been called yet. ``True``
        otherwise.

    Notes
    -----
    To provide support for lazy initialization, apply the :meth:`lazy`
    decorator to the :meth:`__init__` method.

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

    If a brick has children, they must be listed in the :attr:`children`
    attribute. Moreover, if the brick wants to control the configuration of
    its children, the :meth:`_push_allocation_config` and
    :meth:`_push_initialization_config` methods need to be overridden.

    Examples
    --------
    By default, bricks have lazy initialization enabled.

    >>> import theano
    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> linear = Linear(input_dim=5, output_dim=3,
    ...                 weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    >>> x = theano.tensor.vector()
    >>> linear.apply(x)  # Calls linear.allocate() automatically
    linear_apply_output
    >>> linear.initialize()  # Initializes the weight matrix

    In simple cases, eager bricks are easier to deal with.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> Brick.lazy = False
    >>> linear = Linear(5, 3, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    >>> linear.apply(x)
    linear_apply_output

    """
    #: See :attr:`Brick.lazy`
    lazy = True

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

        self.children = []

        self.allocated = False
        self.allocation_config_pushed = False
        self.initialized = False
        self.initialization_config_pushed = False

    def __repr__(self):
        return repr_attrs(self, 'name')

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
        This method sets the :attr:`params` attribute to an empty list.
        This is in order to ensure that calls to this method completely
        reset the parameters.

        """
        if not self.allocation_config_pushed:
            self.push_allocation_config()
        for child in self.children:
            child.allocate()
        self.params = []
        try:
            self._allocate()
        except Exception:
            if self.lazy:
                reraise_as("Lazy initialization is enabled, so please make "
                           "sure you have set all the required configuration "
                           "for this method call.")
            else:
                raise
        self.allocated = True

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
            child.initialize()
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
        """Push the configuration for allocation to child bricks.

        Bricks can configure their children, based on their own current
        configuration. This will be automatically done by a call to
        :meth:`allocate`, but if you want to override the configuration of
        child bricks manually, then you can call this function manually.

        """
        self._push_allocation_config()
        self.allocation_config_pushed = True
        for child in self.children:
            try:
                child.push_allocation_config()
            except Exception:
                self.allocation_config_pushed = False
                raise

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
        """Push the configuration for initialization to child bricks.

        Bricks can configure their children, based on their own current
        configuration. This will be automatically done by a call to
        :meth:`initialize`, but if you want to override the configuration
        of child bricks manually, then you can call this function manually.

        """
        self._push_initialization_config()
        self.initialization_config_pushed = True
        for child in self.children:
            try:
                child.push_initialization_config()
            except Exception:
                self.initialization_config_pushed = False
                raise

    def _push_initialization_config(self):
        """Brick implementation of configuring child before initialization.

        Implement this if your brick needs to set the configuration of its
        children before initialization.

        .. warning::

           This method should never be called directly. Call
           :meth:`push_initialization_config` instead.

        """
        pass

    def get_dim(self, name):
        """Get dimension of an input/output variable of a brick.

        Parameters
        ----------
        name : str
            The name of the variable.

        """
        raise ValueError("No dimension information for {} available"
                         .format(name))

    def get_dims(self, names):
        """Get dictionary of dimensions for a set of input/output variables.

        Parameters
        ----------
        names : list of str
            The dictinonary of variable names.

        Returns
        -------
        dims : dict
            Dictionary of (variable name, variable dimension) pairs.

        """
        return {name: self.get_dim(name) for name in names}


def lazy(func):
    """Makes the initialization lazy.

    Any positional argument not given will be set to ``None``. Positional
    arguments can also be given as keyword arguments.

    Parameters
    ----------
    func : method
        The __init__ method to make lazy.

    Examples
    --------
    >>> class SomeBrick(Brick):
    ...     @lazy
    ...     def __init__(self, a, b, c='c', d=None):
    ...         print(a, b, c, d)
    >>> brick = SomeBrick('a')
    a None c None
    >>> brick = SomeBrick(d='d', b='b')
    None b c d
    >>> Brick.lazy = False
    >>> brick = SomeBrick('a')
    Traceback (most recent call last):
      ...
    TypeError: __init__() takes at least 3 arguments (2 given)
    >>> Brick.lazy = True  # Reset for other doctests

    """
    arg_spec = inspect.getargspec(func)
    arg_names = arg_spec.args[1:]
    defaults = arg_spec.defaults
    if defaults is None:
        defaults = []

    def init(self, *args, **kwargs):
        if not self.lazy:
            return func(self, *args, **kwargs)
        # Fill any missing positional arguments with None
        args = args + (None,) * (len(arg_names) - len(defaults) -
                                 len(args))

        # Check if positional arguments were passed as keyword arguments
        args = list(args)
        for i, arg_name in enumerate(arg_names[:len(arg_names)
                                               - len(defaults)]):
            if arg_name in kwargs:
                if args[i] is not None:
                    raise ValueError
                args[i] = kwargs.pop(arg_name)

        return func(self, *args, **kwargs)
    return init


class VariableRole(object):
    """A collection of constants referring to variable roles."""
    COST = "cost"
    INPUT = "input"
    OUTPUT = "output"
    MONITOR = "monitor"
    ADDITIONAL_COST = "additional_cost"


class ApplicationCall(object):
    """A link between the variable tags and bricks.

    The application call can be used to attach to an apply call auxiliary
    variables (e.g. monitors or regularizers) that do not form part of the
    main computation graph.

    The application call object is created before the call to the
    application method and can be accessed by specifying an
    application_call argument.

    Parameters
    ----------
    brick : object
        The brick whose application is called
    application : object
        The application object being called

    """
    def __init__(self, application):
        self.application = application
        self.brick = application.brick
        self.auxiliary_variables = []
        self.updates = []

    def add_auxiliary_variable(self, expression, role, name=None):
        if name is not None:
            expression.name = name
        expression.tag.role = role
        self.auxiliary_variables.append(expression)

    def add_monitor(self, expression, name=None):
        return self.add_auxiliary_variable(expression,
                                           role=VariableRole.MONITOR,
                                           name=name)

    def add_additional_cost(self, expression, name=None):
        return self.add_auxiliary_variable(expression,
                                           role=VariableRole.ADDITIONAL_COST,
                                           name=name)


class Application(object):
    """A particular application of a brick.

    When a brick is constructed, its application methods are automatically
    replaced by an instance of this class.

    Parameters
    ----------
    application_method : object
        The application method (member of a brick) to replace.

    Attributes
    ----------
    properties : dict
        Properties are methods of the brick which are called when a
        particular attribute is requested of the application.
    delegate : function
        When an attribute can't be found on the application instance and
        this attribute is configured, the attribute is requested from the
        object returned by this function.

    """
    call_stack = []

    def __init__(self, application_method):
        self.application_method = application_method
        self.brick = application_method.__self__
        self.properties = {}
        for key, value in application_method.application_signature.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        if name in self.properties:
            return self.properties[name](self.brick)
        elif hasattr(self, 'delegate'):
            return getattr(self.delegate(), name)
        else:
            raise AttributeError

    def __call__(self, *args, **kwargs):
        """Wraps an application method.

        This wrapper will provide some necessary pre- and post-processing
        of the Theano variables, such as tagging them with the brick that
        created them and naming them. These changes will apply to Theano
        variables given as positional arguments and keywords arguments.

        .. warning::

            Properly set tags are important for correct functioning of the
            framework. Do not provide inputs to your apply method in a way
            different than passing them as positional or keyword arguments,
            e.g. as list or tuple elements.

        Notes
        -----
        Application methods will allocate the brick parameters with a call
        :meth:`allocate` if they have not been allocated already.

        """
        return_dict = kwargs.pop('return_dict', False)
        return_list = kwargs.pop('return_list', False)
        if return_list and return_dict:
            raise ValueError

        # Find the names of the inputs to the application method
        args_names, varargs_name, _, _ = inspect.getargspec(
            self.application_method)
        args_names = args_names[1:]

        # Construct the ApplicationCall, used to store data in for this call
        call = ApplicationCall(self)
        if 'application_call' in args_names:
            kwargs['application_call'] = call

        # Allocate before applying, and optionally initialize
        if not self.brick.allocated:
            self.brick.allocate()
        if not self.brick.initialized and not self.brick.lazy:
            self.brick.initialize()

        # Annotate all the input variables which are Theano variables
        def copy_and_tag(variable, role, name):
            """Helper method to copy a variable and annotate it."""
            copy = variable.copy()
            copy.name = "{}_{}_{}".format(  # Theano name
                self.brick.name, self.application_method.__name__, name)
            copy.tag.application_call = call
            copy.tag.name = name  # Blocks name
            copy.tag.role = role
            return copy

        args = list(args)
        for i, input_ in enumerate(args):
            if isinstance(input_, tensor.Variable):
                if i < len(args_names):
                    name = args_names[i]
                else:
                    name = "{}_{}".format(varargs_name, i - len(args_names))
                args[i] = copy_and_tag(input_, VariableRole.INPUT, name)
        for name, input_ in kwargs.items():
            if isinstance(input_, tensor.Variable):
                kwargs[name] = copy_and_tag(input_, VariableRole.INPUT, name)

        # Run the application method on the annotated variables
        if self.call_stack and self.call_stack[-1] is not self.brick and \
                self.brick not in self.call_stack[-1].children:
            raise ValueError
        self.call_stack.append(self.brick)
        try:
            outputs = self.application_method(*args, **kwargs)
            outputs = pack(outputs)
        finally:
            self.call_stack.pop()

        # Rename and annotate output variables
        for i, output in enumerate(outputs):
            if isinstance(output, tensor.Variable):
                try:
                    name = self.outputs[i]
                except AttributeError:
                    name = "output_{}".format(i)
                except IndexError:
                    reraise_as(ValueError("Unexpected outputs"))
                # TODO Tag with dimensions, axes, etc. for error-checking
                outputs[i] = copy_and_tag(outputs[i],
                                          VariableRole.OUTPUT, name)

        # Return values
        if return_list:
            return outputs
        if return_dict:
            return OrderedDict(zip(self.outputs, outputs))
        return unpack(outputs)


def application(*args, **kwargs):
    r"""Decorator for methods that apply a brick to inputs.

    This decorator performs two functions: It creates an application method
    (tagging the inputs and outputs as such in the Theano graph) and
    creates a signature for this method (allowing other bricks to query the
    in- and outputs of this method).

    Parameters
    ----------
    \*args, optional
        The application method to wrap.
    \*\*kwargs, optional
        Attributes to set on the :class:`Application` instance as part of
        the application's signature.

    Notes
    -----
    This decorator can be used both with and without passing attributes
    that will become part of the signature.

    Technically speaking this decorator only tags the method in question as
    being an application method. When the brick is constructed, the
    :class:`_Brick` metaclass is responsible for replacing this method with
    an instance of :class:`Application` which will perform the actual
    wrapping.

    """
    assert (args and not kwargs) or (not args and kwargs)

    def application_property(application_function, name):
        def tag_application_property(application_property):
            application_property.application_property = (application_function,
                                                         name)
            return application_property
        return tag_application_property

    def tag_application_delegate(application_function, application_delegate):
        application_delegate.application_delegate = application_function
        return application_delegate

    def tag_application_function(application_function):
        application_function.application_signature = kwargs
        application_function.property = partial(application_property,
                                                application_function)
        application_function.delegate = partial(tag_application_delegate,
                                                application_function)
        return application_function

    if args:
        application_function, = args
        return tag_application_function(application_function)
    else:
        def application(application_function):
            return tag_application_function(application_function)
        return application


class Random(Brick):
    """A mixin class for Bricks which need Theano RNGs.

    Parameters
    ----------
    theano_rng : object
        A ``tensor.shared_randomstreams.RandomStreams`` instance.

    """
    def __init__(self, theano_rng=None, **kwargs):
        super(Random, self).__init__(**kwargs)
        self.theano_rng = theano_rng

    @property
    def theano_rng(self):
        """Returns Brick's Theano RNG, or a default one.

        The default seed which can be set at a module level using
        ``blocks.bricks.DEFAULT_SEED = seed``.

        """
        if getattr(self, '_theano_rng', None) is not None:
            return self._theano_rng
        else:
            return tensor.shared_randomstreams.RandomStreams(DEFAULT_SEED)

    @theano_rng.setter
    def theano_rng(self, theano_rng):
        self._theano_rng = theano_rng


class Initializable(Brick):
    """Base class for bricks which push parameter initialization.

    Many bricks will initialize children which perform a linear
    transformation, often with biases. This brick allows the weights
    and biases initialization to be configured in the parent brick and
    pushed down the hierarchy.

    Parameters
    ----------
    weights_init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by :meth:`initialize`.
    biases_init : object, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`initialize` when `use_bias` is
        `True`. Only supported by bricks for which :attr:`has_biases` is
        ``True``.
    use_bias : bool, optional
        Whether to use a bias. Defaults to `True`. Required by
        :meth:`initialize`. Only supported by bricks for which
        :attr:`has_biases` is ``True``.
    rng : object
        A ``numpy.RandomState`` instance.

    Attributes
    ----------
    has_biases : bool
        ``False`` if the brick does not support biases, and only has
        :attr:`weights_init`.  For an example of this, see
        :class:`Bidirectional`. If this is ``False``, the brick does not
        support the arguments ``biases_init`` or ``use_bias``.

    """
    has_biases = True

    @lazy
    def __init__(self, weights_init, biases_init=None, use_bias=True, rng=None,
                 **kwargs):
        super(Initializable, self).__init__(**kwargs)
        self.weights_init = weights_init
        if self.has_biases:
            self.biases_init = biases_init
        else:
            if biases_init is not None or not use_bias:
                raise ValueError("This brick does not support biases config")
        self.use_bias = use_bias
        self.rng = rng

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            return numpy.random.RandomState(DEFAULT_SEED)

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    def _push_initialization_config(self):
        for child in self.children:
            if isinstance(child, Initializable):
                child.rng = self.rng
                if self.weights_init:
                    child.weights_init = self.weights_init
        if hasattr(self, 'biases_init') and self.biases_init:
            for child in self.children:
                if (isinstance(child, Initializable) and
                        hasattr(child, 'biases_init')):
                    child.biases_init = self.biases_init
        super(Initializable, self)._push_initialization_config()


class Linear(Initializable):
    r"""A linear transformation with optional bias.

    Linear brick which applies a linear (affine) transformation by
    multiplying the input with a weight matrix. Optionally a bias is added.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    A linear transformation with bias is a matrix multiplication followed
    by a vector summation.

    .. math:: f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}

    """
    @lazy
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _allocate(self):
        self.params.append(shared_floatx_zeros((self.input_dim,
                                                self.output_dim),
                           name="W"))
        if self.use_bias:
            self.params.append(shared_floatx_zeros((self.output_dim,),
                               name="b"))

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : Theano variable
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
        output = tensor.dot(input_, W)
        if self.use_bias:
            output += b
        return output


class Maxout(Brick):
    """Maxout pooling transformation.

    A brick that does max pooling over groups of input units. If you use
    this code in a research project, please cite [GWFM1313]_.

    .. [GWFM1313] Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
       Courville, and Yoshua Bengio, *Maxout networks*, ICML (2013), pp.
       1319-1327.

    Parameters
    ----------
    num_pieces : int
        The size of the groups the maximum is taken over.

    Notes
    -----
    Maxout applies a set of linear transformations to a vector and selects
    for each output dimension the result with the highest value.

    """
    @lazy
    def __init__(self, num_pieces, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_pieces = num_pieces

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the maxout transformation.

        Parameters
        ----------
        input_ : Theano variable
            The input on which to apply the transformation

        Returns
        -------
        output : Theano variable
            The transformed input

        """
        last_dim = input_.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)]
                     + [output_dim, self.num_pieces])
        output = tensor.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                            axis=input_.ndim)
        return output


class LinearMaxout(Initializable):
    """Maxout pooling following a linear transformation.

    This code combines the :class:`Linear` brick with a :class:`Maxout`
    brick.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`allocate`.
    num_pieces : int
        The number of linear functions. Required by :meth:`allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    .. todo:: Name of :attr:`linear_transformation` shouldn't be hardcoded.

    """
    @lazy
    def __init__(self, input_dim, output_dim, num_pieces, **kwargs):
        super(LinearMaxout, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_pieces = num_pieces

        self.linear_transformation = Linear(name='linear_to_maxout',
                                            input_dim=input_dim,
                                            output_dim=output_dim * num_pieces,
                                            weights_init=self.weights_init,
                                            biases_init=self.biases_init,
                                            use_bias=self.use_bias)
        self.maxout_transformation = Maxout(name='maxout',
                                            num_pieces=num_pieces)
        self.children = [self.linear_transformation,
                         self.maxout_transformation]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation followed by maxout.

        Parameters
        ----------
        input_ : Theano variable
            The input on which to apply the transformations

        Returns
        -------
        output : Theano variable
            The transformed input

        """
        pre_activation = self.linear_transformation.apply(input_)
        output = self.maxout_transformation.apply(pre_activation)
        return output


class ActivationDocumentation(_Brick):
    def __new__(cls, name, bases, classdict):
        classdict['__doc__'] = \
            """Elementwise application of {0} function.""".format(name.lower())
        if 'apply' in classdict:
            classdict['apply'].__doc__ = \
                """Apply the {0} function elementwise.

                Parameters
                ----------
                input_ : Theano variable
                    Theano variable to apply {0} to, elementwise.

                Returns
                -------
                output : Theano variable
                    The input with the activation function applied.

                """.format(name.lower())
        return type.__new__(cls, name, bases, classdict)


@add_metaclass(ActivationDocumentation)
class Activation(Brick):
    """A base class for simple, elementwise activation functions.

    This base class ensures that activation functions are automatically
    documented using the :class:`ActivationDocumentation` metaclass.

    """
    pass


class Identity(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_


class Tanh(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.tanh(input_)


class Sigmoid(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.sigmoid(input_)


class Rectifier(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.switch(input_ > 0, input_, 0)


class Softmax(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softmax(input_)


class Sequence(Brick):
    """A sequence of bricks.

    This brick simply applies a sequence of bricks, assuming that their in-
    and outputs are compatible.

    Parameters
    ----------
    bricks : list of :class:`Brick` instances
        The bricks in the order that they need to be applied.
    application_methods : list of application method names, optional
        If not given, it uses ``'apply'`` for each brick.

    """
    def __init__(self, bricks, application_methods=None, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        if application_methods is None:
            application_methods = ['apply' for _ in bricks]
        if not len(application_methods) == len(bricks):
            raise ValueError
        self.children = bricks
        self.application_methods = application_methods

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        child_input = input_
        for child, application_method in zip(self.children,
                                             self.application_methods):
            output = getattr(child, application_method)(*pack(child_input))
            child_input = output
        return output


class MLP(Sequence, Initializable):
    """A simple multi-layer perceptron.

    Parameters
    ----------
    activations : bricks or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> Brick.lazy = True
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(),
    ...           biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """
    @lazy
    def __init__(self, activations, dims, **kwargs):
        self.activations = activations

        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        # Interleave the transformations and activations
        children = [child for child in list(chain(*zip(
            self.linear_transformations, activations))) if child is not None]
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(MLP, self).__init__(children, **kwargs)

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in zip(self.dims[:-1], self.dims[1:],
                                                self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias
