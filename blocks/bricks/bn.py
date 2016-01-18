import collections

import numpy
from picklable_itertools.extras import equizip
from theano.tensor.nnet import bn

from ..graph import add_annotation
from ..initialization import Constant
from ..roles import (WEIGHT, BIAS, BATCH_NORM_POPULATION_MEAN,
                     BATCH_NORM_POPULATION_STDEV, BATCH_NORM_OFFSET,
                     BATCH_NORM_DIVISOR, add_role)
from ..utils import (shared_floatx_zeros, shared_floatx,
                     shared_floatx_nans)
from .base import lazy, application
from .sequences import Sequence, Feedforward, MLP
from .interfaces import RNGMixin


class BatchNormalization(RNGMixin, Feedforward):
    r"""Normalizes activations, parameterizes a scale and shift.

    Parameters
    ----------
    input_dim : int or tuple
        Shape of a single input example. It is assumed that a batch axis
        will be prepended to this.
    broadcastable : tuple, optional
        Tuple the same length as `input_dim` which specifies which of the
        per-example axes should be averaged over to compute means and
        standard deviations. For example, in order to normalize over all
        spatial locations in a `(batch_index, channels, height, width)`
        image, pass `(False, True, True)`.
    save_memory : bool, optional
        Use an implementation that stores less intermediate state and
        therefore uses less memory, at the expense of 5-10% speed. Default
        is `True`.
    weights_init : object, optional
        Initialization object to use for the learned scaling parameter
        ($\\gamma$ in [BN]_). By default, uses constant initialization
        of 1.
    biases_init : object, optional
        Initialization object to use for the learned shift parameter
        ($\\beta$ in [BN]_). By default, uses constant initialization of 0.

    Notes
    -----
    In order for trained models to behave sensibly immediately upon
    upon deserialization, by default, this brick runs in *inference* mode,
    using a population mean and population standard deviation (initialized
    to zeros and ones respectively) to normalize activations. It is
    expected that the user will adapt these during training in some
    fashion, independently of the training objective, e.g. by taking a
    moving average of minibatch-wise statistics.

    In order to *train* with batch normalization, one must obtain a
    training graph by transforming the original inference graph.  See
    :func:`batch_normalize`.

    This Brick accepts `weights_init` and `biases_init` arguments but is
    *not* an instance of :class:`~blocks.bricks.Initializable`, and will
    therefore not receive pushed initialization config from any parent
    brick. In almost all cases, you will probably want to stick with the
    defaults (unit scale and zero shift), but you can explicitly pass one
    or both initializers to override this.

    This has the necessary properties to be inserted into a
    :class:`blocks.bricks.conv.ConvolutionalSequence` as-is, in which case
    the `input_dim` should be omitted at construction, to be inferred from
    the layer below.

    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, broadcastable=None,
                 save_memory=True, weights_init=None,
                 biases_init=None, **kwargs):
        self.input_dim = input_dim
        self.broadcastable = broadcastable
        self.save_memory = save_memory
        self.weights_init = (Constant(1) if weights_init is None
                             else weights_init)
        self.biases_init = (Constant(0) if biases_init is None
                            else biases_init)
        super(BatchNormalization, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        mean = self.population_mean.copy(name='population_offset')
        stdev = self.population_stdev.copy(name='population_divisor')

        def annotate(var, role):
            add_role(var, role)
            add_annotation(var, self)
            add_annotation(var, application_call)

        annotate(mean, BATCH_NORM_OFFSET)
        annotate(stdev, BATCH_NORM_DIVISOR)

        # Heavy lifting is done by the Theano utility function.
        normalized = bn.batch_normalization(input_, self.W,
                                            self.b, mean, stdev,
                                            mode=('low_mem' if self.save_memory
                                                  else 'high_mem'))
        return normalized

    def _allocate(self):
        input_dim = ((self.input_dim,)
                     if not isinstance(self.input_dim, collections.Sequence)
                     else self.input_dim)
        broadcastable = (tuple(False for _ in range(len(input_dim)))
                         if self.broadcastable is None else self.broadcastable)
        if len(input_dim) != len(broadcastable):
            raise ValueError("input_dim and broadcastable must be same length")
        var_dim = ((1,) +  # batch axis
                   tuple(1 if broadcast else dim for dim, broadcast in
                         equizip(input_dim, broadcastable)))
        broadcastable = (True,) + broadcastable

        # "gamma", from the Ioffe & Szegedy manuscript.
        self._W = shared_floatx_nans(var_dim, name='batch_norm_scale',
                                     broadcastable=broadcastable)

        # "beta", from the Ioffe & Szegedy manuscript.
        self._b = shared_floatx_nans(var_dim, name='batch_norm_shift',
                                     broadcastable=broadcastable)
        add_role(self.W, WEIGHT)
        add_role(self.b, BIAS)
        self.parameters.append(self.W)
        self.parameters.append(self.b)

        # These aren't technically parameters, in that they should not be
        # learned using the same cost function as other model parameters.
        self.population_mean = shared_floatx_zeros(var_dim,
                                                   name='population_mean',
                                                   broadcastable=broadcastable)
        self.population_stdev = shared_floatx(numpy.ones(var_dim),
                                              name='population_stdev',
                                              broadcastable=broadcastable)
        add_role(self.population_mean, BATCH_NORM_POPULATION_MEAN)
        add_role(self.population_stdev, BATCH_NORM_POPULATION_STDEV)

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    def _initialize(self):
        self.biases_init.initialize(self.b, self.rng)
        self.weights_init.initialize(self.W, self.rng)

    # Needed for the Feedforward interface.
    @property
    def output_dim(self):
        return self.input_dim

    # The following properties allow for BatchNormalization bricks
    # to be used directly inside of a ConvolutionalSequence.
    @property
    def image_size(self):
        return self.input_dim[-2:]

    @image_size.setter
    def image_size(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (None,) + tuple(value)
        else:
            self.input_dim = (self.input_dim[0],) + tuple(value)

    @property
    def num_channels(self):
        return self.input_dim[0]

    @num_channels.setter
    def num_channels(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (value,) + (None, None)
        else:
            self.input_dim = (value,) + self.input_dim[-2:]

    def get_dim(self, name):
        if name in ('input', 'output'):
            return self.input_dim
        else:
            raise KeyError

    @property
    def num_output_channels(self):
        return self.num_channels


class SpatialBatchNormalization(BatchNormalization):
    """Convenient subclass for batch normalization across spatial inputs.

    Parameters
    ----------
    input_dim : int or tuple
        The input size of a single example. Must be length at least 2.
        It's assumed that the first axis of this tuple is a "channels"
        axis, which should not be summed over, and all remaining
        dimensions are spatial dimensions.

    Notes
    -----
    See :class:`BatchNormalization` for more details (and additional
    keyword arguments).

    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, **kwargs):
        if not isinstance(input_dim,
                          collections.Sequence) or len(input_dim) < 2:
            raise ValueError('expected input_dim to be length >= 2 '
                             '(channels, height, width)')
        broadcastable = (False,) + ((True,) * (len(input_dim) - 1))
        kwargs.setdefault('broadcastable', broadcastable)
        super(SpatialBatchNormalization, self).__init__(input_dim, **kwargs)


class BatchNormalizedMLP(MLP):
    """Convenient subclass for building an MLP with batch normalization.

    Parameters
    ----------
    save_memory : bool, optional
        See :class:`BatchNormalization`.

    Notes
    -----
    All other parameters are the same as :class:`~blocks.bricks.MLP`. Each
    activation brick is wrapped in a :class:`~blocks.bricks.Sequence`
    containing an appropriate :class:`BatchNormalization` brick and
    the activation that follows it.

    By default, the contained :class:`~blocks.bricks.Linear` bricks will
    not contain any biases, as they could be canceled out by the biases
    in the :class:`BatchNormalization` bricks being added. Pass
    `use_bias` with a value of `True` if you really want this for some
    reason.

    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, *args, **kwargs):
        save_memory = kwargs.pop('save_memory', True)
        activations = [
            Sequence([BatchNormalization(save_memory=save_memory).apply,
                      act.apply], name='batch_norm_activation_{}'.format(i))
            for i, act in enumerate(activations)
        ]
        # Batch normalization bricks incorporate a bias, so there's no
        # need for our Linear bricks to have them.
        kwargs.setdefault('use_bias', False)
        super(BatchNormalizedMLP, self).__init__(activations, dims, *args,
                                                 **kwargs)

    @property
    def save_memory(self):
        return self._save_memory

    @save_memory.setter
    def save_memory(self, value):
        self._save_memory = value
        for act in self.activations:
            assert isinstance(act.children[0], BatchNormalization)
            act.children[0].save_memory = value

    def _push_allocation_config(self):
        super(BatchNormalizedMLP, self)._push_allocation_config()
        # Do the extra allocation pushing for the BatchNormalization
        # bricks. They need as their input dimension the output dimension
        # of each linear transformation.  Exclude the first dimension,
        # which is the input dimension.
        for act, dim in equizip(self.activations, self.dims[1:]):
            act.children[0].input_dim = dim
