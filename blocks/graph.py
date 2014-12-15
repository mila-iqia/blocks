"""Annonated computation graph management."""

import logging

import theano
from theano import Variable
from theano.scalar import ScalarConstant
from theano.tensor import TensorConstant
from theano.tensor.sharedvar import SharedVariable
from theano.tensor.shared_randomstreams import RandomStreams

logger = logging.getLogger(__name__)


class ComputationGraph(object):
    """Encapsulates a managed Theano computation graph.

    Attributes
    ----------
    inputs : list of Theano variables
        The inputs of the computation graph.
    outputs : list of Theano variables
        The outputs of the computations graph.

    Parameters
    ----------
    outputs : list of Theano variables
        The outputs of the computation graph.

    """
    def __init__(self, outputs):
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = outputs
        self._get_variables()

    def _get_variables(self):
        def recursion(current):
            self.variables.add(current)
            if current.owner:
                owner = current.owner
                if owner not in self.applies:
                    if hasattr(owner.tag, 'updates'):
                        logger.debug("updates of {}".format(owner))
                        self.updates.extend(owner.tag.updates.items())
                    self.applies.add(owner)

                for inp in owner.inputs:
                    if inp not in self.variables:
                        recursion(inp)

        def is_input(variable):
            return (not variable.owner
                    and not isinstance(variable, SharedVariable)
                    and not isinstance(variable, TensorConstant)
                    and not isinstance(variable, ScalarConstant))

        self.variables = set()
        self.applies = set()
        self.updates = []
        for output in self.outputs:
            recursion(output)
        self.inputs = [v for v in self.variables if is_input(v)]

    def dict_of_inputs(self):
        """Return a mapping from an input name to the input."""
        return {var.name: var for var in self.inputs}

    def replace(self, replacements):
        """Replace certain variables in the computation graph."""
        return ComputationGraph(theano.clone(self.outputs,
                                             replace=replacements))

    def function(self):
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
    rng : Theano random stream
        The random stream to use.

    """
    if not rng:
        rng = RandomStreams(1)
    replace = {}
    for variable in variables:
        replace[variable] = (variable +
                             rng.normal(variable.shape, std=level))
    return graph.replace(replace)
