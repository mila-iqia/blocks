"""Annotated computation graph management."""
import logging
from itertools import chain

import theano
from theano import Variable
from theano.gof import graph
from theano.gof.sched import make_dependence_cmp, sort_apply_nodes
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.bricks.base import VariableRole
from blocks.utils import is_graph_input, is_shared_variable

logger = logging.getLogger(__name__)
dependence = make_dependence_cmp()


class ComputationGraph(object):
    """Encapsulates a managed Theano computation graph.

    This implies that it not only contains the variables required to
    compute the given outputs, but also all the auxiliary variables and
    updates that were attached to these variables through the annotation
    system.

    All variables are presented in topologically sorted order according to
    the apply nodes that they are an input to.

    Parameters
    ----------
    outputs : Theano variable or list of Theano variables
        The output(s) of the computation graph.

    Attributes
    ----------
    inputs : list of Theano variables
        The inputs of the computation graph. This does not include shared
        variables and constants.
    shared_variables : list of Theano shared variables
        All the shared variables in the graph.
    outputs : list of Theano variables
        The outputs of the computations graph (as passed to the
        constructor).
    auxiliary_variables : list of Theano variables
        All variables which have the :attr:`Variable.AUXILIARY` role.
    intermediary_variables : list of Theano variables
        Any variable that is not part of :attr:`inputs` or :attr:`outputs`.
    variables : list of Theano variables
        All variables (including auxiliary) in the managed graph.
    updates : list of (Theano variable, Theano expression) pairs
        All the updates found attached to the annotations.

    """
    def __init__(self, outputs):
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = outputs
        self._get_variables()

    @property
    def inputs(self):
        """Inputs to the graph, excluding constants and shared variables."""
        return [var for var in self.variables if is_graph_input(var)]

    @property
    def intermediary_variables(self):
        return [var for var in self.variables if
                var not in self.inputs and
                var not in self.outputs]

    @property
    def shared_variables(self):
        return [var for var in self.variables if is_shared_variable(var)]

    @property
    def auxiliary_variables(self):
        return [var for var in self.variables if hasattr(var.tag, 'roles') and
                VariableRole.AUXILIARY in var.tag.roles]

    def _get_variables(self):
        """Collect variables, updates and auxiliary variables."""
        application_calls = set()
        annotations = set()
        bricks = set()
        updates = []

        # Sort apply nodes topologically, get variables and remove duplicates
        inputs = graph.inputs(self.outputs)
        sorted_apply_nodes = sort_apply_nodes([inputs], self.outputs,
                                              [dependence])
        seen = set()
        variables = [var for var in list(chain(
            *[apply_node.inputs for apply_node in sorted_apply_nodes]))
            if not (var in seen or seen.add(var))] + self.outputs

        # While preserving order add auxiliary variables, and collect updates
        i = 0
        while i < len(variables):
            var = variables[i]
            for tag in ('application_call', 'annotation', 'brick'):
                annotation = getattr(var.tag, tag, None)
                if annotation and annotation not in locals()[tag + 's']:
                    new_avs = [av for av in annotation.auxiliary_variables
                               if av not in variables]
                    variables = variables[:i + 1] + new_avs + variables[i + 1:]
                    i += len(new_avs)
                    updates.extend(annotation.updates)
            i += 1

        self.variables = variables
        self.updates = updates

    def get_variables(self, roles=None, predicate=None):
        """Return variables of the computation graph.

        Parameters
        ----------
        roles : list of :class:`VariableRole` attributes, optional
            If given, only returns variables that have any of the roles
            given.
        predicate : function
            A function which takes a variable as an input, and returns
            ``True`` if the value should be returned, ``False`` otherwise.

        """
        if roles is not None:
            variables = [var for var in self.variables
                         if hasattr(var.tag, 'roles') and
                         bool(set(roles) & set(var.tag.roles))]
        else:
            variables = self.variables
        if predicate is not None:
            variables = [var for var in variables if predicate(var)]
        return variables

    def dict_of_inputs(self):
        """Return a mapping from an input name to the input."""
        return {var.name: var for var in self.inputs}

    def replace(self, replacements):
        """Replace certain variables in the computation graph.

        Parameters
        ----------
        replacements : dict
            The mapping from variables to be replaced to the corresponding
            substitutes.

        """
        return ComputationGraph(theano.clone(self.outputs,
                                             replace=replacements))

    def get_theano_function(self):
        """Create Theano function from the graph contained."""
        return theano.function(self.inputs, self.outputs,
                               updates=self.updates)


def apply_noise(graph, variables, level, rng=None):
    """Add Gaussian noise to certain variable of a computation graph.

    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The computation graph.
    varibles : Theano variables
        Variables to add noise to.
    level : float
        Noise level.
    rng : Theano random stream, optional
        The random stream to use. By default an RNG with seed equal to 1 is
        used.

    """
    if not rng:
        rng = RandomStreams(1)
    replace = {}
    for variable in variables:
        replace[variable] = (variable +
                             rng.normal(variable.shape, std=level))
    return graph.replace(replace)
