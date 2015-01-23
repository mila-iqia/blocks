"""Annotated computation graph management."""
import logging
from itertools import chain

import theano
from theano import Variable
from theano.gof import graph
from theano.gof.sched import make_dependence_cmp, sort_apply_nodes
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.utils import is_graph_input, is_shared_variable

logger = logging.getLogger(__name__)
dependence = make_dependence_cmp()


class ComputationGraph(object):
    """Encapsulates a managed Theano computation graph.

    Parameters
    ----------
    outputs : list of Theano variables
        The outputs of the computation graph.

    Attributes
    ----------
    inputs : list of Theano variables
        The inputs of the computation graph.
    outputs : list of Theano variables
        The outputs of the computations graph.

    """
    def __init__(self, outputs):
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = outputs
        self._get_variables()

    def _get_variables(self):
        application_calls = set()
        updates = []

        inputs = graph.inputs(self.outputs)
        sorted_apply_nodes = sort_apply_nodes([inputs], self.outputs,
                                              [dependence])
        seen = set()
        variables = [var for var in list(chain(
            *[apply_node.inputs for apply_node in sorted_apply_nodes]))
            if not (var in seen or seen.add(var))] + self.outputs
        inputs = [v for v in inputs if is_graph_input(v)]

        i = 0
        while i < len(variables):
            var = variables[i]
            if hasattr(var.tag, 'application_call'):
                application_call = var.tag.application_call
                if application_call not in application_calls:
                    variables = (
                        variables[:i + 1] +
                        [av for av in application_call.auxiliary_variables
                         if av not in variables] +
                        variables[i + 1:])
                    i += len(application_call.auxiliary_variables)
            i += 1

        self.updates = updates
        self.inputs = inputs
        self.variables = variables

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

    def get_shared_variables(self):
        """Returns all shared variables found in the computation graph."""
        return [var for var in self.variables if is_shared_variable(var)]

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
