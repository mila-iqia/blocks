"""Frequently used Theano expressions."""
from theano import tensor


def l2_norm(tensors):
    """Computes the total L2 norm of a set of tensors.

    Converts all operands to :class:`~tensor.TensorVariable`
    (see :func:`~tensor.as_tensor_variable`).

    Parameters
    ----------
    tensors : iterable of :class:`~tensor.TensorVariable` (or compatible)
        The tensors.

    """
    flattened = [tensor.as_tensor_variable(t).flatten() for t in tensors]
    flattened = [(t if t.ndim > 0 else t.dimshuffle('x'))
                 for t in flattened]
    joined = tensor.join(0, *flattened)
    return tensor.sqrt(tensor.sqr(joined).sum())
