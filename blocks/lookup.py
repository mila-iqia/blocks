"""Introduces Lookup brick."""

from theano import tensor

from blocks.bricks import Brick, DefaultRNG
from blocks.utils import shared_floatx_zeros, check_theano_variable


class LookupTable(DefaultRNG):
    """Incapsulates representations of a range of integers.

    Parameters
    ----------
    size : int
        The size of the lookup table, or in other words, one plus the
        maximum index for which a representation is contained.
    dim : int
        The dimensionality of representations.

    """
    @Brick.lazy_method
    def __init__(self, length, dim, weights_init, **kwargs):
        super(LookupTable, self).__init__(**kwargs)
        self.__dict__.update(**locals())
        del self.self
        del self.kwargs

    @property
    def W(self):
        return self.params[0]

    def _allocate(self):
        self.params.append(shared_floatx_zeros((self.length, self.dim),
                           name="W"))

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @Brick.apply_method
    def lookup(self, indices):
        """Perform lookup.

        Parameters
        ----------
        indices : Theano variable
            The indices of interest. The dtype must be integer.

        Returns
        -------
        output : Theano variable
            Representations for the indices of the query. Has :math:`k+1`
            dimensions, where :math:`k` is the number of dimensions of the
            `indices` parameter. The last dimension stands for the
            representation element.

        """

        check_theano_variable(indices, None, "int")
        output_ndim = indices.ndim + 1
        output_shape = self.dim * tensor.ones((output_ndim,),
                                              dtype=indices.dtype)
        output_shape = tensor.set_subtensor(output_shape[:indices.ndim],
                                            indices.shape)
        return self.W[indices.flatten()].reshape(output_shape,
                                                 ndim=output_ndim)
