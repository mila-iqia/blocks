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


def hessian_times_vector(gradient, parameter, vector, r_op=False):
    """Return an expression for the Hessian times a vector.

    Parameters
    ----------
    gradient : :class:`~tensor.TensorVariable`
        The gradient of a cost with respect to `parameter`
    parameter : :class:`~tensor.TensorVariable`
        The parameter with respect to which to take the gradient
    vector : :class:`~tensor.TensorVariable`
        The vector with which to multiply the Hessian
    rop : bool, optional
        Whether to use :func:`~tensor.gradient.Rop` or not. Defaults to
        ``False``. Which solution is fastest normally needs to be
        determined by profiling.

    """
    if r_op:
        return tensor.Rop(gradient, parameter, vector)
    return tensor.grad(tensor.sum(gradient * vector), parameter)
