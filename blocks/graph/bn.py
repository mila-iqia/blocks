"""Implements the batch normalization training graph transform.

Specifically, this module contains the implementation for the
transformation of a batch-normalized inference graph into training graph,
which uses minibatch statistics in place of population statistics.

"""
import collections
from theano import tensor

from . import add_annotation
from ..roles import (BATCH_NORM_OFFSET, BATCH_NORM_DIVISOR,
                     BATCH_NORM_POPULATION_STATISTICS,
                     BATCH_NORM_MINIBATCH_ESTIMATE, INPUT, add_role,
                     has_roles)


def batch_normalize(computation_graph, epsilon=1e-4):
    """Activate batch normalization in a graph.

    Parameters
    ----------
    computation_graph : instance of :class:`ComputationGraph`
        The computation graph containing :class:`BatchNormalization`
        brick applications.
    epsilon : float, optional
        The stabilizing constant for the minibatch standard deviation
        computation. Added to the variance inside the square root, as
        in the batch normalization paper.

    Returns
    -------
    batch_normed_computation_graph : instance of :class:`ComputationGraph`
        The computation graph, with :class:`BatchNormalization`
        applications transformed to use minibatch statistics instead
        of accumulated population statistics.
    population_to_minibatch : OrderedDict
        A mapping of variables used in the original graph for population
        means and standard deviations to the minibatch-derived quantities
        that replace them. Useful to define updates in order to track
        the approximate population statistics during learning.

    Notes
    -----
    Assumes the minibatch axis is 0. Other axes are unsupported at
    this time.

    """
    # Avoid a circular import.
    from ..filter import VariableFilter, get_application_call

    # Create filters for variables involved in a batch normalization brick
    # application.
    def make_variable_filter(role):
        return VariableFilter(roles=[role])

    mean_filter, stdev_filter, input_filter = map(make_variable_filter,
                                                  [BATCH_NORM_OFFSET,
                                                   BATCH_NORM_DIVISOR, INPUT])

    # Group means, standard deviations, and inputs into dicts indexed by
    # application call.
    def get_application_call_dict(variable_filter):
        return collections.OrderedDict((get_application_call(v), v) for v in
                                       variable_filter(computation_graph))

    means, stdevs, inputs = map(get_application_call_dict,
                                [mean_filter, stdev_filter, input_filter])

    assert (set(means.keys()) == set(stdevs.keys()) and
            set(means.keys()) == set(inputs.keys()))
    assert set(means.values()).isdisjoint(stdevs.values())

    replacements = []
    # Perform replacement for each application call.
    for application_call in means:
        axes = tuple(i for i, b in enumerate(means[application_call]
                                             .broadcastable) if b)
        minibatch_mean = inputs[application_call].mean(axis=axes,
                                                       keepdims=True)
        minibatch_mean.name = 'minibatch_offset'
        # Stabilize in the same way as the batch normalization manuscript.
        minibatch_std = tensor.sqrt(tensor.var(inputs[application_call],
                                               axis=axes, keepdims=True) +
                                    epsilon)
        minibatch_std.name = 'minibatch_divisor'

        def prepare_replacement(old, new, role, application_call):
            """Add roles and tags to replaced variables."""
            add_role(new, BATCH_NORM_MINIBATCH_ESTIMATE)
            add_role(new, role)
            add_annotation(new, application_call)
            add_annotation(new, application_call.application.brick)
            new.tag.replacement_of = old
            replacements.append((old, new))

        prepare_replacement(means[application_call], minibatch_mean,
                            BATCH_NORM_OFFSET, application_call)
        prepare_replacement(stdevs[application_call], minibatch_std,
                            BATCH_NORM_DIVISOR, application_call)

    new_graph = computation_graph.replace(replacements)

    population_to_minibatch = collections.OrderedDict()
    for original_graph_node, replacement in replacements:
        pop_stats = original_graph_node
        while not has_roles(pop_stats, [BATCH_NORM_POPULATION_STATISTICS]):
            pop_stats = pop_stats.owner.inputs[0]
        population_to_minibatch[pop_stats] = replacement
    return new_graph, population_to_minibatch
