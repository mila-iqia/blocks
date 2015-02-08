from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTERS
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
        Number of input channels in the image e.g. 1 for grayscale and 3
        for RGB. The filters are pooled over the channels.
    step : tuple, optional
        The step (or stride) with which to slide the filters over the
        image. Defaults to (1, 1).
    border_mode : {'valid', 'full'}, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Defaults to 'valid'.

    """
    has_bias = False

    @lazy
    def __init__(self, filter_size, num_filters, num_channels,
                 step=(1, 1), border_mode='valid', **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.border_mode = border_mode
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.step = step

    def _allocate(self):
        filter_size_x, filter_size_y = self.filter_size
        W = shared_floatx_zeros((self.num_filters, self.num_channels,
                                 filter_size_x, filter_size_y), name='W')
        add_role(W, FILTERS)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')

    def _initialize(self):
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
        W, = self.params
        output = conv2d(
            input_, W, subsample=self.step, border_mode=self.border_mode,
            filter_shape=(self.num_filters,
                          self.num_channels) + self.filter_size)
        return output


class MaxPooling(Initializable, Feedforward):
    """Max pooling layer.

    Parameters
    ----------
    pooling_size : tuple
        The height and width of the pooling region i.e. this is the factor
        by which your input's last two dimensions will be downscaled.

    """
    @lazy
    def __init__(self, pooling_size, **kwargs):
        super(MaxPooling, self).__init__(**kwargs)
        self.pooling_size = pooling_size

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
        output = max_pool_2d(input_, self.pooling_size)
        return output


class ConvolutionalLayer(Sequence, Initializable):
    """A complete convolutional layer: Convolution, nonlinearity, pooling.

    Parameters
    ----------
    activation : :class:`.Application`
        The application method to apply in the detector stage (i.e. the
        nonlinearity before pooling.

    See :class:`Convolutional` and :class:`MaxPooling` for explanations of
    other parameters.

    """
    def __init__(self, filter_size, num_filters, num_channels, pooling_size,
                 activation, step=(1, 1), border_mode='valid', **kwargs):
        convolution = Convolutional(filter_size, num_filters,
                                    num_channels, step, border_mode)
        pooling = MaxPooling(pooling_size)
        super(ConvolutionalLayer, self).__init__(
            application_methods=[convolution.apply, activation, pooling.apply],
            **kwargs)


class Flattener(Brick):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        batch_size = input_.shape[0]
        return input_.reshape((batch_size, -1))
