"""Implements the batch normalization training graph transform.

Specifically, this module contains the implementation for the
transformation of a batch-normalized inference graph into training graph,
which uses minibatch statistics in place of population statistics.

"""
import collections
import contextlib

from ..roles import BATCH_NORM_OFFSET, BATCH_NORM_DIVISOR, INPUT, OUTPUT
from ..utils import find_bricks


@contextlib.contextmanager
def batch_normalization(*bricks):
    r"""Context manager to run batch normalization in "training mode".

    Parameters
    ----------
    \*bricks
        One or more bricks which will be inspected for descendant
        instances of :class:`~blocks.bricks.BatchNormalization`.

    Notes
    -----
    Graph replacement using :func:`apply_batch_normalization`, while
    elegant, can lead to Theano graphs that are quite large and result
    in very slow compiles. This provides an alternative mechanism for
    building the batch normalized training graph. It can be somewhat
    less convenient as it requires building the graph twice if one
    wishes to monitor the output of the inference graph during training.

    Examples
    --------
    First, we'll create a :class:`~blocks.bricks.BatchNormalizedMLP`.
    This behaves almost exactly like a regular :class:`~blocks.bricks.MLP`
    except that it contains :class:`~blocks.bricks.BatchNormalization`
    bricks placed before each activation function.

    >>> import theano
    >>> from blocks.bricks import BatchNormalizedMLP, Tanh
    >>> from blocks.initialization import Constant, IsotropicGaussian
    >>> mlp = BatchNormalizedMLP([Tanh(), Tanh()], [4, 5, 6],
    ...                          weights_init=IsotropicGaussian(0.1),
    ...                          biases_init=Constant(0))
    >>> mlp.initialize()
    >>> data = numpy.arange(12, dtype=theano.config.floatX).reshape(3, 4)
    >>> x = theano.tensor.matrix('x')

    First, we'll construct an output variable as we would normally. This
    is getting normalized by the *population* statistics, which by
    default are initialized to 0 (mean) and 1 (standard deviation),
    respectively.

    >>> y = mlp.apply(x)

    And now, to construct an output with batch normalization enabled,
    i.e. normalizing pre-activations using per-minibatch statistics, we
    simply make a similar call inside of a `with` statement:

    >>> with batch_normalization(mlp):
    ...     y_bn = mlp.apply(x)

    Let's verify that these two graphs behave differently on the
    same data:

    >>> import numpy
    >>> data = numpy.arange(12, dtype=theano.config.floatX).reshape(3, 4)
    >>> inf_y = y.eval({x: data})
    >>> trn_y = y_bn.eval({x: data})
    >>> numpy.allclose(inf_y, trn_y)
    False

    """
    # Avoid circular imports.
    from blocks.bricks import BatchNormalization

    bn = find_bricks(bricks, lambda b: isinstance(b, BatchNormalization))
    # Can't use either nested() (deprecated) nor ExitStack (not available
    # on Python 2.7). Well, that sucks.
    for brick in bn:
        brick.__enter__()
    yield
    for brick in bn:
        brick.__exit__()


def apply_batch_normalization(computation_graph):
    """Transform a graph into a batch-normalized training graph.

    Parameters
    ----------
    computation_graph : instance of :class:`ComputationGraph`
        The computation graph containing :class:`BatchNormalization`
        brick applications.

    Returns
    -------
    batch_normed_computation_graph : instance of :class:`ComputationGraph`
        The computation graph, with :class:`BatchNormalization`
        applications transformed to use minibatch statistics instead
        of accumulated population statistics.
    update_pairs : list of tuples
        A list of 2-tuples where the first element of each tuple is the
        shared variable containing a "population" mean or standard
        deviation, and the second is a Theano variable for the
        corresponding statistics on a minibatch. Note that multiple
        applications of a single :class:`blocks.bricks.BatchNormalization`
        may appear in the graph, and therefore a single population variable
        may map to several different minibatch variables.

    See Also
    --------
    :func:`batch_normalization`, for an alternative method to produce
    batch normalized graphs.

    """
    # Avoid circular imports.
    from blocks.bricks import BatchNormalization
    from ..filter import VariableFilter, get_application_call

    # Create filters for variables involved in a batch normalization brick
    # application.
    def make_variable_filter(role):
        return VariableFilter(bricks=[BatchNormalization], roles=[role])

    # Group inputs and outputs into dicts indexed by application call.
    def get_app_call_dict(variable_filter):
        return collections.OrderedDict((get_application_call(v), v) for v in
                                       variable_filter(computation_graph))

    # Compose these two so that we get 4 dicts, grouped by application
    # call, of different variable roles involved in BatchNormalization.
    inputs, outputs, means, stdevs = map(get_app_call_dict,
                                         map(make_variable_filter,
                                             [INPUT, OUTPUT, BATCH_NORM_OFFSET,
                                              BATCH_NORM_DIVISOR]))

    assert len(set([len(inputs), len(outputs), len(means), len(stdevs)])) == 1

    # Remove any ApplicationCalls that were not generated by apply(), or
    # were generated by an apply() while already in training mode.
    remove = filter(lambda a: (a.metadata.get('training_mode', False) or
                               a.application.application !=
                               BatchNormalization.apply), inputs.keys())
    for app_call in remove:
        for mapping in (inputs, outputs, means, stdevs):
            del mapping[app_call]

    replacements = []
    update_pairs = []
    for app_call in inputs:
        old_output = outputs[app_call]
        unpacked = inputs[app_call].owner.inputs[0]
        with app_call.application.brick:
            new_output = app_call.application.brick.apply(unpacked)
        replacements.append((old_output, new_output))
        new_app_call = get_application_call(new_output)
        update_pairs.append((app_call.application.brick.population_mean,
                             new_app_call.metadata['offset']))
        update_pairs.append((app_call.application.brick.population_stdev,
                             new_app_call.metadata['divisor']))
    return computation_graph.replace(replacements), update_pairs
