from theano.tensor.nnet.conv import conv2d, ConvOp
from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMax

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTERS, BIASES
from blocks.utils import shared_floatx_zeros


class Convolutional(Initializable):
    """Performs a 2D convolution.

    .. todo::

       Allow passing of image shapes for faster execution.

    Parameters
    ----------
    filter_size : tuple
        The height and width of the filter (also called *kernels*).
    num_filters : int
        Number of filters per channel.
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer. The filters are pooled over the
        channels.
    input_dim : tuple, optional
        The height and width of the input (image or feature map). If given,
        this will be passed to the Theano convolution operator, resulting
        in possibly faster execution times.
    step : tuple, optional
        The step (or stride) with which to slide the filters over the
        image. Defaults to (1, 1).
    border_mode : {'valid', 'full'}, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Defaults to 'valid'.

    """
    @lazy
    def __init__(self, filter_size, num_filters, num_channels, input_dim=None,
                 step=(1, 1), border_mode='valid', **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.input_dim = input_dim
        self.border_mode = border_mode
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.step = step

    def _allocate(self):
        W = shared_floatx_zeros((self.num_filters, self.num_channels) +
                                self.filter_size, name='W')
        add_role(W, FILTERS)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            b = shared_floatx_zeros((self.num_filters,), name='b')
            add_role(b, BIASES)
            self.params.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Perform the convolution.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            A 4D tensor with the axes representing batch size, number of
            channels, image height, and image width.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A 4D tensor of filtered images (feature maps) with dimensions
            representing batch size, number of filters, feature map height,
            and feature map width.

            The height and width of the feature map depend on the border
            mode. For 'valid' it is ``image_size - filter_size + 1`` while
            for 'full' it is ``image_shape + filter_size - 1``.

        """
        if self.use_bias:
            W, b = self.params
        else:
            W = self.params

        output = conv2d(
            input_, W, image_shape=(None,) + self.input_dim,
            subsample=self.step,
            border_mode=self.border_mode,
            filter_shape=((self.num_filters, self.num_channels) +
                          self.filter_size))
        if self.use_bias:
            output += b.dimshuffle('x', 0, 'x', 'x')
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return ((self.num_filters,) +
                    ConvOp.getOutputShape(self.input_dim[1:], self.filter_size,
                                          self.step, self.border_mode))
        return super(Convolutional, self).get_dim(name)


class MaxPooling(Initializable, Feedforward):
    """Max pooling layer.

    Parameters
    ----------
    pooling_size : tuple
        The height and width of the pooling region i.e. this is the factor
        by which your input's last two dimensions will be downscaled.
    step : tuple, optional
        The vertical and horizontal shift (stride) between pooling regions.
        By default this is equal to `pooling_size`. Setting this to a lower
        number results in overlapping pooling regions.
    input_dim : tuple, optional
        A tuple of integers representing the shape of the input. The last
        two dimensions will be used to calculate the output dimension.

    """
    @lazy
    def __init__(self, pooling_size, step=None, input_dim=None, **kwargs):
        super(MaxPooling, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.pooling_size = pooling_size
        self.step = step

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the pooling (subsampling) transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            An tensor with dimension greater or equal to 2. The last two
            dimensions will be downsampled. For example, with images this
            means that the last two dimensions should represent the height
            and width of your image.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A tensor with the same number of dimensions as `input_`, but
            with the last two dimensions downsampled.

        """
        output = max_pool_2d(input_, self.pooling_size, st=self.step)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return tuple(DownsampleFactorMax.out_shape(self.input_dim,
                                                       self.pooling_size,
                                                       st=self.step))


class ConvolutionalLayer(Sequence, Initializable):
    """A complete convolutional layer: Convolution, nonlinearity, pooling.

    .. todo::

       Mean pooling.

    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply in the detector stage (i.e. the
        nonlinearity before pooling.

    See :class:`Convolutional` and :class:`MaxPooling` for explanations of
    other parameters.

    Notes
    -----
    Uses max pooling.

    """
    def __init__(self, filter_size, num_filters, num_channels, pooling_size,
                 activation, conv_step=(1, 1), pooling_step=None,
                 border_mode='valid', input_dim=None, **kwargs):
        self.convolution = Convolutional(filter_size, num_filters,
                                         num_channels, input_dim=input_dim,
                                         step=conv_step,
                                         border_mode=border_mode)
        if input_dim is not None:
            pooling_input_dim = self.convolution.get_dim('output')
        else:
            pooling_input_dim = None
        self.pooling = MaxPooling(pooling_size, step=pooling_step,
                                  input_dim=pooling_input_dim)
        super(ConvolutionalLayer, self).__init__(
            application_methods=[self.convolution.apply, activation,
                                 self.pooling.apply], **kwargs)

        self.input_dim = input_dim

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return self.pooling.get_dim('output')
        return super(ConvolutionalLayer, self).get_dim(name)


class Flattener(Brick):
    """Flattens the input.

    It may be used to pass multidimensional objects like images or feature
    maps of convolutional bricks into bricks which allow only two
    dimensional input (batch, features) like MLP.

    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        batch_size = input_.shape[0]
        return input_.reshape((batch_size, -1))
