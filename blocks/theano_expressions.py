"""Frequently used Theano expressions."""
from theano import tensor


def L2_norm(tensors):
    """Computes the total L2 norm of a set of tensors.

    Parameters
    ----------
    tensors : iterable of :class:`~theano.Variable`
        The tensors.

    """
    flattened = [tensor.as_tensor_variable(t).flatten() for t in tensors]
    flattened = [(t if t.ndim > 0 else tensor.alloc(t, 1))
                 for t in flattened]
    joined = tensor.join(0, *flattened)
    return tensor.sqrt(tensor.sqr(joined).sum())
