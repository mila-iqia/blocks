from theano import tensor

from blocks.bricks.base import Brick, application


class As2D(Brick):
    """Wraps a 2D application around reshapes to accomodate N-D tensors.

    The input is first flattened across all but its last axis, then the
    application method is called on the flattened input, and finally
    the output is reshaped so that it has the same number of dimensions
    as the input.

    Parameters
    ----------
    application_method : callable
        Some brick's :meth:`apply` method

    """
    def __init__(self, application_method, **kwargs):
        super(As2D, self).__init__(**kwargs)
        self.application_method = application_method
        self.children = [self.application_method.brick]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        shape, ndim = input_.shape, input_.ndim
        if ndim > 2:
            input_ = input_.reshape((tensor.prod(shape[:-1]), shape[-1]))
            output = self.application_method(input_)
            output = output.reshape(
                tensor.set_subtensor(shape[-1], output.shape[1]),
                ndim=ndim)
        else:
            output = self.application_method(input_)
        return output
