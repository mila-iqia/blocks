"""Introduces Lookup brick."""
import numpy
import theano
from theano import tensor

from blocks.bricks import Initializable
from blocks.bricks.base import application, lazy
from blocks.utils import (check_theano_variable, shared_floatx_zeros,
                          shared_floatx)


class LookupTable(Initializable):
    """Encapsulates representations of a range of integers.

    Parameters
    ----------
    length : int
        The size of the lookup table, or in other words, one plus the
        maximum index for which a representation is contained.
    dim : int
        The dimensionality of representations.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    """
    has_bias = False

    @lazy
    def __init__(self, length, dim, **kwargs):
        super(LookupTable, self).__init__(**kwargs)
        self.length = length
        self.dim = dim

    @property
    def W(self):
        return self.params[0]

    def _allocate(self):
        self.params.append(shared_floatx_zeros((self.length, self.dim),
                           name='W'))

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @application
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
        output_shape = [indices.shape[i]
                        for i in range(indices.ndim)] + [self.dim]
        return self.W[indices.flatten()].reshape(output_shape)


class Hash(Initializable):
    has_bias = False

    @lazy
    def __init__(self, dim, bits, **kwargs):
        super(Hash, self).__init__(**kwargs)
        self.dim = dim
        self.bits = bits

    def _allocate(self):
        self.params = [shared_floatx(self.rng.normal(size=(self.bits,
                                                           self.dim + 1)))]

    @application
    def apply(self, W, indices=None):
        hash_vectors = self.params[0]
        if indices is not None:
            W = W[indices]
        W_norms = W.norm(2, axis=0)
        max_W_norm = W_norms.max()
        scaled_W = W / max_W_norm
        part_1 = tensor.dot(hash_vectors[:, :-1], scaled_W)
        part_2 = tensor.outer(hash_vectors[:, -1],
                              tensor.sqrt(1 - tensor.sqrt(W_norms /
                                                          max_W_norm)))
        mappings = part_1 + part_2
        signs = tensor.switch(mappings < 0, numpy.int64(0), numpy.int64(1)).T
        hashes = (signs * (2 ** tensor.arange(self.bits,
                                              dtype='int64'))).sum(axis=1)
        return hashes
