"""The blocks library for parameterized Theano ops"""
from functools import wraps

from blocks.bricks import SEPARATOR, UNDEF
from blocks.initialization import NdarrayInitialization
from blocks.utils import pack, unpack

BLOCK_PREFIX = 'block'


class Block(object):
    """Blocks are groups of bricks with a particular function.

    Bricks are atomic elements which can be reused to build a computational
    graph. Blocks are far more flexible: They can contain any number of
    bricks and blocks, in effect describing a tree-like structure that will
    help you manage your model. Some blocks will combine sets of bricks in
    intelligent ways, managing their life cycles for you.

    All blocks have names, which you can use to configure your model as a
    whole. Like bricks, blocks have `apply` methods which take Theano
    variables as input and output.

    Within a block, the names of its children (blocks and bricks) need to
    be unique.

    Parameters
    ----------
    name : str, optional
        The name of this brick. This can be used to filter the application
        of certain modifications by block names. By default the block
        receives the name of its class (lowercased). If this block is
        nested, the name is expected to be unique within its parent block.

    Attributes
    ----------
    children : list of objects
        List of Block and Brick instances which belong to this class.
        Blocks can be expected to be children of this class alone, while
        bricks can also be part of other blocks.

    """
    children = []

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}{}{}'.format(BLOCK_PREFIX, SEPARATOR, name)

    @staticmethod
    def apply_method(func):
        """Wraps block apply methods.

        Like bricks, the apply methods of blocks need to be wrapped. This
        is to make sure that Theano variables can be named and tagged in
        the appropriate ways.

        However, the wrapping is less comprehensive than bricks. Blocks are
        free to handle the allocation and initialization in any way they
        see fit (some blocks will do it for you, others won't).

        Parameters
        ----------
        func : method
            A method which takes Theano variables as an input, and returns
            the output of a block.

        """
        @wraps(func)
        def wrapped_apply(self, *inputs, **kwargs):
            inputs = list(inputs)
            for i, inp in enumerate(inputs):
                inputs[i] = inp.copy()
                inputs[i].owner = self
            outputs = pack(func(self, *inputs, **kwargs))
            for i, output in enumerate(outputs):
                outputs[i] = output.copy()
                outputs[i].owner = self
            return unpack(outputs)
        wrapped_apply._raw = func
        return wrapped_apply


class DaisyChain(Block):
    """A daisy-chain of bricks.

    This block takes a daisy chain of bricks and strings them together so
    that the input and output dimensions match.

    Parameters
    ----------
    layers : list of Brick instances
        Bricks are expected to have a ``input_dim`` and a ``output_dim``
        argument. The `output_dim` should already be set, the ``input_dim``
        will be set by this block, except for the first block.
    default_weights_init : object
        A :class:`NdarrayInitialization` object that will be used to
        initialize the weights of layers who don't have an initialization
        object already.
    default_biases_init : object
        A :class:`NdarrayInitialization` object to initialize the biases of
        layers which don't have biases_init set already.

    Notes
    -----

    If lazy initialization is enabled for some bricks, this block will not
    override these settings.

    Examples
    --------

    Consider a simple neural network with a single hidden layer and a
    softmax output layer.

    .. math::

       \mathbf{h} &= \\tanh(\mathbf{W}_0 \mathbf{x} + \mathbf{b}_0) \\\\
       \mathbf{y} &= \mathrm{softmax}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1)

    where :math:`\mathbf{x} \in \mathbb{R}^{30}`, :math:`\mathbf{h} \in
    \mathbb{R}^{20}` and :math:`\mathbf{y} \in \mathbb{R}^{10}`. We
    initialize the weight matrices by drawing from a standard normal
    distribution, and set the initial biases to zero.

    >>> from theano import tensor
    ... x = tensor.matrix()
    ... mlp = DaisyChain([LinearTanh(input_dim=30, output_dim=20),
    ...                   LinearSoftmax(output_dim=10)],
    ...                  default_weights_init=IsotropicGaussian(),
    ...                  default_biases_init=Constant(0))
    ... mlp.fprop(x)

    """
    def __init__(self, layers, default_weights_init=None,
                 default_biases_init=None):
        # Input validation
        if layers[0].input_dim is UNDEF:
            raise ValueError("First layer must have input dimension")
        if not all([layer.input_dim is UNDEF for layer in layers[1:]]):
            raise ValueError("All but the first layer must not have input_dim")
        if any([layer.weights_init is UNDEF for layer in layers]) and \
                default_weights_init is None:
            raise ValueError("Not all weights_init are set, and no default")
        if any([(layer.biases_init is UNDEF and layer.use_bias)
                for layer in layers]) and default_biases_init is None:
            raise ValueError("Not all biases_init are set, and no default")
        for init in [default_weights_init, default_biases_init]:
            if init is not None and \
                    not isinstance(init, NdarrayInitialization):
                raise ValueError("must be NdarrayInitialization instance")

        # Set the layer input dimensions
        output_dim = None
        for i, layer in enumerate(layers):
            if i > 0:
                layer.input_dim = output_dim
            if layer.weights_init is UNDEF:
                layer.weights_init = default_weights_init
            if layer.use_bias and layer.biases_init is None:
                layer.biases_init = default_biases_init
            output_dim = layer.output_dim
        self.children = layers

    @Block.apply_method
    def fprop(self, inp):
        """Forward-propogate the input through the MLP.

        Parameters
        ----------
        inp : Theano variable
            The input to the network.

        Returns
        -------
        output : Theano variable
            The output of the last layer of the MLP.

        """
        for layer in self.children:
            inp = layer.apply(inp)
        output = inp
        return output
