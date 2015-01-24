from inspect import isclass

from blocks.bricks.base import Brick


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
                try:
                    var_brick, = [annotation for annotation in
                                  getattr(var.tag, 'annotations', [])
                                  if isinstance(annotation, Brick)]
                except ValueError:
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
