"""Annotated computation graph management."""
import logging
from collections import OrderedDict
from enum import Enum
from inspect import isclass
from itertools import chain

import theano
from theano import Variable
from theano.gof import graph
from theano.gof.sched import make_dependence_cmp, sort_apply_nodes
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.utils import is_graph_input, is_shared_variable, dict_union

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
        updates = OrderedDict()

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
        seen = {'application_call': set(), 'annotation': set(), 'brick': set()}
        while i < len(variables):
            var = variables[i]
            for tag in ('application_call', 'annotation', 'brick'):
                annotation = getattr(var.tag, tag, None)
                if annotation and annotation not in seen[tag]:
                    seen[tag].add(annotation)
                    new_avs = [av for av in annotation.auxiliary_variables
                               if av not in variables]
                    variables = variables[:i + 1] + new_avs + variables[i + 1:]
                    i += len(new_avs)
                    updates = dict_union(updates, annotation.updates)
            i += 1

        self.variables = variables
        self.updates = updates

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


class VariableFilter(object):
    """Filters Theano variables based on a range of criteria.

    Parameters
    ----------
    roles : list of :class:`VariableRole` attributes
        Matches any attribute which has one of the roles given.
    bricks : list of :class:`Brick` classes or instances
        Matches any variable whose brick is either the given brick, or
        whose brick is of a given class

    """
    def __init__(self, roles=None, bricks=None):
        self.roles = roles
        self.bricks = bricks

    def __call__(self, variables):
        """Filter the given variables.

        Parameters
        ----------
        variables : list of Theano variables

        """
        if self.roles is not None:
            variables = [var for var in variables if
                         hasattr(var.tag, 'roles') and
                         bool(set(self.roles) & set(var.tag.roles))]
        if self.bricks is not None:
            filtered_variables = []
            for var in variables:
                if hasattr(var, 'brick'):
                    var_brick = var.brick
                elif hasattr(var, 'application_call'):
                    var_brick = var.application_call.brick
                else:
                    continue
                for brick in self.bricks:
                    if isclass(brick) and isinstance(var_brick, brick):
                        filtered_variables.append(var)
                        break
                    elif var_brick is brick:
                        filtered_variables.append(var)
                        break
            variables = filtered_variables
        return variables


class VariableRole(str, Enum):
    """A collection of constants referring to variable roles."""
    #: Any variable attached to a brick or application call
    AUXILIARY = 'auxiliary'
    #: A scalar variable which represents some cost or regularization penalty
    COST = 'cost'
    #: The input to a brick
    INPUT = 'input'
    #: The output of a brick
    OUTPUT = 'output'
    #: Any parameter of the model
    PARAMETER = 'parameter'
    #: The weights of a particular linear transformation
    WEIGHTS = 'weights'
    #: The biases added after a linear transformation
    BIASES = 'biases'

    @classmethod
    def add_role(cls, var, role):
        r"""Add a role to a given Theano variable.

        Some roles will imply others, using this helper function will make
        sure that these roles are also added.

        Parameters
        ----------
        var : Theano variable
            The variable to assign the new role to.
        role : attribute of :class:`VariableRole`

        Examples
        --------
        >>> from theano import tensor
        >>> W = tensor.matrix()
        >>> VariableRole.add_role(W, VariableRole.WEIGHTS)
        >>> print(*W.tag.roles)
        VariableRole.PARAMETER VariableRole.WEIGHTS

        """
        roles = getattr(var.tag, 'roles', [])
        if role not in roles:
            if role in (cls.WEIGHTS, cls.BIASES) and \
                    cls.PARAMETER not in roles:
                roles.append(cls.PARAMETER)
            roles.append(role)
            var.tag.roles = roles


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
        rng = MRG_RandomStreams(1)
    replace = {}
    for variable in variables:
        replace[variable] = (variable +
                             rng.normal(variable.shape, std=level))
    return graph.replace(replace)
