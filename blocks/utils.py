import theano


def pack(arg):
    """Pack variables into a list.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. Lists will be
        returned as is, and tuples will be cast to lists. Any other
        variable will be returned in a singleton list.

    Returns
    -------
    list
        List containing the arguments

    """
    if isinstance(arg, (list, tuple)):
        return list(arg)
    else:
        return [arg]


def unpack(arg):
    """Unpack variables from a list or tuple.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. If passed a
        list or tuple of length one, the only element of that list will
        be returned. If passed a tuple of length greater than one, it
        will be cast to a list before returning. Any other variable
        will be returned as is.

    Returns
    -------
    object
        A list of length greater than one, or any other Python object
        except tuple.

    """
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1:
            return arg[0]
        else:
            return list(arg)
    else:
        return arg


def sharedX(value, name=None, borrow=False, dtype=None):
    """Transform a value into a shared variable of type floatX.

    Parameters
    ----------
    value : array_like
        The value to associate with the Theano shared.
    name : str, optional
        The name for the shared varaible. Defaults to `None`.
    borrow : bool, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : str, optional
        The `dtype` of the shared variable. Default value is
        `theano.config.floatX`.

    Returns
    -------
    theano.compile.SharedVariable
        A Theano shared variable with the requested value and `dtype`.

    """

    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)
