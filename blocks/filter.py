from inspect import isclass

from blocks.bricks.base import ApplicationCall, Brick


def get_annotation(var, cls):
    """A helper function to retrieve an annotation of a particular type.

    Notes
    -----
    This function returns the first annotation of a particular type. If
    there are multiple--there shouldn't be--it will ignore them.

    """
    for annotation in getattr(var.tag, 'annotations', []):
        if isinstance(annotation, cls):
            return annotation


def get_brick(var):
    """A helper function to retrieve the brick that created this variable.

    See :func:`get_annotation`.

    """
    return get_annotation(var, Brick)


def get_application_call(var):
    return get_annotation(var, ApplicationCall)


class VariableFilter(object):
    """Filters Theano variables based on a range of criteria.

    Parameters
    ----------
    roles : list of :class:`VariableRole` attributes, optional
        Matches any attribute which has one of the roles given.
    bricks : list of :class:`Brick` classes or instances. optional
        Matches any variable whose brick is either the given brick, or
        whose brick is of a given class.

    Notes
    -----
    Note that only auxiliary variables, parameters, inputs and outputs are
    tagged with the brick that created them. Other Theano variables that
    were created in the process of applying a brick will be filtered out.

    Examples
    --------
    >>> from blocks.bricks import MLP, Linear, Sigmoid, Identity
    >>> mlp = MLP(activations=[Identity(), Sigmoid()], dims=[20, 10, 20])
    >>> from theano import tensor
    >>> x = tensor.matrix()
    >>> y_hat = mlp.apply(x)
    >>> from blocks.graph import ComputationGraph, VariableRole
    >>> cg = ComputationGraph(y_hat)
    >>> from blocks.filter import VariableFilter
    >>> var_filter = VariableFilter(roles=[VariableRole.BIASES],
    ...                             bricks=[mlp.linear_transformations[0]])
    >>> var_filter(cg.variables)
    [b]

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
                var_brick = get_brick(var)
                if var_brick is None:
                    import ipdb; ipdb.set_trace()
                if var_brick is None:
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
