import numpy
import theano
from theano import tensor
from theano import printing
from theano.gof.graph import Constant
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable


def shared_floatx_zeros_matching(shared_variable, name=None, **kwargs):
    r"""Create another shared variable with matching shape and broadcast.

    Parameters
    ----------
    shared_variable : :class:'tensor.TensorSharedVariable'
        A Theano shared variable with the desired shape and broadcastable
        flags.
    name : :obj:`str`, optional
        The name for the shared variable. Defaults to `None`.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx_zeros`
        function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A new shared variable, initialized to all zeros, with the same
        shape and broadcastable flags as `shared_variable`.


    """
    if not is_shared_variable(shared_variable):
        raise ValueError('argument must be a shared variable')
    return shared_floatx_zeros(shared_variable.get_value().shape,
                               name=name,
                               broadcastable=shared_variable.broadcastable,
                               **kwargs)


def shared_floatx_zeros(shape, **kwargs):
    r"""Creates a shared variable array filled with zeros.

    Parameters
    ----------
    shape : tuple
        A tuple of integers representing the shape of the array.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx` function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A Theano shared variable filled with zeros.

    """
    return shared_floatx(numpy.zeros(shape), **kwargs)


def shared_floatx_nans(shape, **kwargs):
    r"""Creates a shared variable array filled with nans.

    Parameters
    ----------
    shape : tuple
         A tuple of integers representing the shape of the array.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx` function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A Theano shared variable filled with nans.

    """
    return shared_floatx(numpy.nan * numpy.zeros(shape), **kwargs)


def shared_floatx(value, name=None, borrow=False, dtype=None, **kwargs):
    r"""Transform a value into a shared variable of type floatX.

    Parameters
    ----------
    value : :class:`~numpy.ndarray`
        The value to associate with the Theano shared.
    name : :obj:`str`, optional
        The name for the shared variable. Defaults to `None`.
    borrow : :obj:`bool`, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : :obj:`str`, optional
        The `dtype` of the shared variable. Default value is
        :attr:`config.floatX`.
    \*\*kwargs
        Keyword arguments to pass to the :func:`~theano.shared` function.

    Returns
    -------
    :class:`tensor.TensorSharedVariable`
        A Theano shared variable with the requested value and `dtype`.

    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name, borrow=borrow, **kwargs)


def shared_like(variable, name=None, **kwargs):
    r"""Construct a shared variable to hold the value of a tensor variable.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        The variable whose dtype and ndim will be used to construct
        the new shared variable.
    name : :obj:`str` or :obj:`None`
        The name of the shared variable. If None, the name is determined
        based on variable's name.
    \*\*kwargs
        Keyword arguments to pass to the :func:`~theano.shared` function.

    """
    variable = tensor.as_tensor_variable(variable)
    if name is None:
        name = "shared_{}".format(variable.name)
    return theano.shared(numpy.zeros((0,) * variable.ndim,
                                     dtype=variable.dtype),
                         name=name, **kwargs)


def check_theano_variable(variable, n_dim, dtype_prefix):
    """Check number of dimensions and dtype of a Theano variable.

    If the input is not a Theano variable, it is converted to one. `None`
    input is handled as a special case: no checks are done.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable` or convertible to one
        A variable to check.
    n_dim : int
        Expected number of dimensions or None. If None, no check is
        performed.
    dtype_prefix : str
        Expected dtype prefix or None. If None, no check is performed.

    """
    if variable is None:
        return

    if not isinstance(variable, tensor.Variable):
        variable = tensor.as_tensor_variable(variable)

    if n_dim and variable.ndim != n_dim:
        raise ValueError(
            "Wrong number of dimensions:\n\texpected {}, got {}".format(
                n_dim, variable.ndim))

    if dtype_prefix and not variable.dtype.startswith(dtype_prefix):
        raise ValueError(
            "Wrong dtype prefix:\n\texpected starting with {}, got {}".format(
                dtype_prefix, variable.dtype))


def is_graph_input(variable):
    """Check if variable is a user-provided graph input.

    To be considered an input the variable must have no owner, and not
    be a constant or shared variable.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`

    Returns
    -------
    bool
        ``True`` If the variable is a user-provided input to the graph.

    """
    return (not variable.owner and
            not isinstance(variable, SharedVariable) and
            not isinstance(variable, Constant))


def is_shared_variable(variable):
    """Check if a variable is a Theano shared variable.

    Notes
    -----
    This function excludes shared variables that store the state of Theano
    random number generators.

    """
    return (isinstance(variable, SharedVariable) and
            not isinstance(variable, RandomStateSharedVariable) and
            not hasattr(variable.tag, 'is_rng'))


def put_hook(variable, hook_fn, *args):
    r"""Put a hook on a Theano variables.

    Ensures that the hook function is executed every time when the value
    of the Theano variable is available.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        The variable to put a hook on.
    hook_fn : function
        The hook function. Should take a single argument: the variable's
        value.
    \*args : list
        Positional arguments to pass to the hook function.

    """
    return printing.Print(global_fn=lambda _, x: hook_fn(x, *args))(variable)
