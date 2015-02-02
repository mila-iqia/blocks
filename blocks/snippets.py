"""Routines to build frequently used Theano expressions."""
from theano import tensor


def L2_norm(tensors):
    """Computes the total L2 norm of a set of tensors.

    Parameters
    ----------
    tensors : iterable of :class:`~theano.Variable`
        The tensors.

    """
    return tensor.sqrt(sum((t ** 2).sum() for t in tensors))
