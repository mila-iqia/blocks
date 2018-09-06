from __future__ import print_function
import six
import sys
import contextlib
from collections import OrderedDict, deque

# for documentation
import numpy  # noqa: F401


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


def reraise_as(new_exc):
    """Reraise an exception as a different type or with a message.

    This function ensures that the original traceback is kept, making for
    easier debugging.

    Parameters
    ----------
    new_exc : :class:`Exception` or :obj:`str`
        The new error to be raised e.g. (ValueError("New message"))
        or a string that will be prepended to the original exception
        message

    Notes
    -----
    Note that when reraising exceptions, the arguments of the original
    exception are cast to strings and appended to the error message. If
    you want to retain the original exception arguments, please use:

    >>> try:
    ...     1 / 0
    ... except Exception as e:
    ...     reraise_as(Exception("Extra information", *e.args))
    Traceback (most recent call last):
      ...
    Exception: 'Extra information, ...

    Examples
    --------
    >>> class NewException(Exception):
    ...     def __init__(self, message):
    ...         super(NewException, self).__init__(message)
    >>> try:
    ...     do_something_crazy()
    ... except Exception:
    ...     reraise_as(NewException("Informative message"))
    Traceback (most recent call last):
      ...
    NewException: Informative message ...

    """
    orig_exc_type, orig_exc_value, orig_exc_traceback = sys.exc_info()

    if isinstance(new_exc, six.string_types):
        new_exc = orig_exc_type(new_exc)

    if hasattr(new_exc, 'args'):
        if len(new_exc.args) > 0:
            # We add all the arguments to the message, to make sure that this
            # information isn't lost if this exception is reraised again
            new_message = ', '.join(str(arg) for arg in new_exc.args)
        else:
            new_message = ""
        new_message += '\n\nOriginal exception:\n\t' + orig_exc_type.__name__
        if hasattr(orig_exc_value, 'args') and len(orig_exc_value.args) > 0:
            if getattr(orig_exc_value, 'reraised', False):
                new_message += ': ' + str(orig_exc_value.args[0])
            else:
                new_message += ': ' + ', '.join(str(arg)
                                                for arg in orig_exc_value.args)
        new_exc.args = (new_message,) + new_exc.args[1:]

    new_exc.__cause__ = orig_exc_value
    new_exc.reraised = True
    six.reraise(type(new_exc), new_exc, orig_exc_traceback)


def unpack(arg, singleton=False):
    """Unpack variables from a list or tuple.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. If passed a
        list or tuple of length one, the only element of that list will
        be returned. If passed a tuple of length greater than one, it
        will be cast to a list before returning. Any other variable
        will be returned as is.
    singleton : bool
        If ``True``, `arg` is expected to be a singleton (a list or tuple
        with exactly one element) and an exception is raised if this is not
        the case. ``False`` by default.

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
            if singleton:
                raise ValueError("Expected a singleton, got {}".
                                 format(arg))
            return list(arg)
    else:
        return arg


def dict_subset(dict_, keys, pop=False, must_have=True):
    """Return a subset of a dictionary corresponding to a set of keys.

    Parameters
    ----------
    dict_ : dict
        The dictionary.
    keys : iterable
        The keys of interest.
    pop : bool
        If ``True``, the pairs corresponding to the keys of interest are
        popped from the dictionary.
    must_have : bool
        If ``True``, a ValueError will be raised when trying to retrieve a
        key not present in the dictionary.

    Returns
    -------
    result : ``OrderedDict``
        An ordered dictionary of retrieved pairs. The order is the same as
        in the ``keys`` argument.

    """
    not_found = object()

    def extract(k):
        if pop:
            if must_have:
                return dict_.pop(k)
            return dict_.pop(k, not_found)
        if must_have:
            return dict_[k]
        return dict_.get(k, not_found)

    result = [(key, extract(key)) for key in keys]
    return OrderedDict([(k, v) for k, v in result if v is not not_found])


def dict_union(*dicts, **kwargs):
    r"""Return union of a sequence of disjoint dictionaries.

    Parameters
    ----------
    dicts : dicts
        A set of dictionaries with no keys in common. If the first
        dictionary in the sequence is an instance of `OrderedDict`, the
        result will be OrderedDict.
    \*\*kwargs
        Keywords and values to add to the resulting dictionary.

    Raises
    ------
    ValueError
        If a key appears twice in the dictionaries or keyword arguments.

    """
    dicts = list(dicts)
    if dicts and isinstance(dicts[0], OrderedDict):
        result = OrderedDict()
    else:
        result = {}
    for d in list(dicts) + [kwargs]:
        duplicate_keys = set(result.keys()) & set(d.keys())
        if duplicate_keys:
            raise ValueError("The following keys have duplicate entries: {}"
                             .format(", ".join(str(key) for key in
                                               duplicate_keys)))
        result.update(d)
    return result


def repr_attrs(instance, *attrs):
    r"""Prints a representation of an object with certain attributes.

    Parameters
    ----------
    instance : object
        The object of which to print the string representation
    \*attrs
        Names of attributes that should be printed.

    Examples
    --------
    >>> class A(object):
    ...     def __init__(self, value):
    ...         self.value = value
    >>> a = A('a_value')
    >>> repr(a)  # doctest: +SKIP
    <blocks.utils.A object at 0x7fb2b4741a10>
    >>> repr_attrs(a, 'value')  # doctest: +SKIP
    <blocks.utils.A object at 0x7fb2b4741a10: value=a_value>

    """
    orig_repr_template = ("<{0.__class__.__module__}.{0.__class__.__name__} "
                          "object at {1:#x}")
    if attrs:
        repr_template = (orig_repr_template + ": " +
                         ", ".join(["{0}={{0.{0}}}".format(attr)
                                    for attr in attrs]))
    repr_template += '>'
    orig_repr_template += '>'
    try:
        return repr_template.format(instance, id(instance))
    except Exception:
        return orig_repr_template.format(instance, id(instance))


def ipdb_breakpoint(x):
    """A simple hook function for :func:`put_hook` that runs ipdb.

    Parameters
    ----------
    x : :class:`~numpy.ndarray`
        The value of the hooked variable.

    """
    import ipdb
    ipdb.set_trace()


def print_sum(x, header=None):
    if not header:
        header = 'print_sum'
    print(header + ':', x.sum())


def print_shape(x, header=None):
    if not header:
        header = 'print_shape'
    print(header + ':', x.shape)


@contextlib.contextmanager
def change_recursion_limit(limit):
    """Temporarily changes the recursion limit."""
    old_limit = sys.getrecursionlimit()
    if old_limit < limit:
        sys.setrecursionlimit(limit)
    yield
    sys.setrecursionlimit(old_limit)


def extract_args(expected, *args, **kwargs):
    r"""Route keyword and positional arguments to a list of names.

    A frequent situation is that a method of the class gets to
    know its positional arguments only when an instance of the class
    has been created. In such cases the signature of such method has to
    be `*args, **kwargs`. The downside of such signatures is that the
    validity of a call is not checked.

    Use :func:`extract_args` if your method knows at runtime, but not
    at evaluation/compile time, what arguments it actually expects,
    in order to check that they are correctly received.

    Parameters
    ----------
    expected : list of str
        A list of strings denoting names for the expected arguments,
        in order.
    args : iterable
        Positional arguments that have been passed.
    kwargs : Mapping
        Keyword arguments that have been passed.

    Returns
    -------
    routed_args : OrderedDict
        An OrderedDict mapping the names in `expected` to values drawn
        from either `args` or `kwargs` in the usual Python fashion.

    Raises
    ------
    KeyError
        If a keyword argument is passed, the key for which is not
        contained within `expected`.
    TypeError
        If an expected argument is accounted for in both the positional
        and keyword arguments.
    ValueError
        If certain arguments in `expected` are not assigned a value
        by either a positional or keyword argument.

    """
    # Use of zip() rather than equizip() intentional here. We want
    # to truncate to the length of args.
    routed_args = dict(zip(expected, args))
    for name in kwargs:
        if name not in expected:
            raise KeyError('invalid input name: {}'.format(name))
        elif name in routed_args:
            raise TypeError("got multiple values for "
                            "argument '{}'".format(name))
        else:
            routed_args[name] = kwargs[name]
    if set(expected) != set(routed_args):
        raise ValueError('missing values for inputs: {}'.format(
            [name for name in expected
             if name not in routed_args]))
    return OrderedDict((key, routed_args[key]) for key in expected)


def find_bricks(top_bricks, predicate):
    """Walk the brick hierarchy, return bricks that satisfy a predicate.

    Parameters
    ----------
    top_bricks : list
        A list of root bricks to search downward from.
    predicate : callable
        A callable that returns `True` for bricks that meet the
        desired criteria or `False` for those that don't.

    Returns
    -------
    found : list
        A list of all bricks that are descendants of any element of
        `top_bricks` that satisfy `predicate`.

    """
    found = []
    visited = set()
    to_visit = deque(top_bricks)
    while len(to_visit) > 0:
        current = to_visit.popleft()
        if current not in visited:
            visited.add(current)
            if predicate(current):
                found.append(current)
            to_visit.extend(current.children)
    return found
