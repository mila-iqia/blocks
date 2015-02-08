"""Generic transformations with multiple inputs and/or outputs.

As bricks from this module are intended to serve as adapters, most of them
are lazy-only, i.e. can not be initialized with a single constructor call.

"""
import copy

from blocks.bricks import Initializable, Linear
from blocks.bricks.base import lazy, application


class Parallel(Initializable):
    """Apply similar transformations to several inputs.

    Given a prototype brick, a :class:`Parallel` brick makes several
    copies of it (each with its own parameters). At the application time
    every copy is applied to the respective input.

    >>> from theano import tensor
    >>> from blocks.initialization import Constant
    >>> x = tensor.matrix('x')
    >>> y = tensor.matrix('y')
    >>> parallel = Parallel(
    ...     input_names=['x', 'y'],
    ...     input_dims=dict(x=2, y=3), output_dims=dict(x=4, y=5),
    ...     weights_init=Constant(2))
    >>> parallel.initialize()
    >>> new_x, new_y = parallel.apply(x=x, y=y)
    >>> new_x.eval({x: [[1, 1]]}) # doctest: +ELLIPSIS
    array([[ 4.,  4.,  4.,  4.]]...
    >>> new_y.eval({y: [[1, 1, 1]]}) # doctest: +ELLIPSIS
    array([[ 6.,  6.,  6.,  6.,  6.]]...

    Parameters
    ----------
    input_names : list of str
        Input names.
    input_dims : dict
        Dictonary of input dimensions, keys are input names, values are
        dimensions.
    output_dims : dict
        Dictionary of output dimensions, keys are input names, values are
        dimensions of transformed inputs.
    prototype : :class:`~blocks.bricks.Feedforward`
        A transformation prototype. A copy will be created for every
        input.  If ``None``, a linear transformation without bias is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, input_names, input_dims, output_dims,
                 prototype=None, **kwargs):
        super(Parallel, self).__init__(**kwargs)
        if not prototype:
            prototype = Linear(use_bias=False)

        self.input_names = input_names
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.prototype = prototype

        self.transforms = []
        for name in input_names:
            self.transforms.append(copy.deepcopy(self.prototype))
            self.transforms[-1].name = "transform_{}".format(name)
        self.children = self.transforms

    def _push_allocation_config(self):
        for name, transform in zip(self.input_names, self.transforms):
            transform.input_dim = self.input_dims[name]
            transform.output_dim = self.output_dims[name]

    @application
    def apply(self, **kwargs):
        return [transform.apply(kwargs[name])
                for name, transform
                in zip(self.input_names, self.transforms)]

    @apply.property('inputs')
    def apply_inputs(self):
        return self.input_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.input_names


class Fork(Parallel):
    """Forks single input into multiple channels.

    Parameters
    ----------
    fork_names : list of str
        Names of the channels to fork.
    prototype : instance of :class:`.Brick`
        A prototype for the input-to-fork transformations. A copy will be
        created for every output channel.

    Attributes
    ----------
    input_dim : int
        The input dimension. Required for allocation.
    fork_dims : dict of (output_name, int) pairs
        The dimensions of the forks. Required for allocation.

    Notes
    -----
        Lazy initialization only.

    """
    def __init__(self, fork_names, prototype=None, **kwargs):
        super(Fork, self).__init__(fork_names, prototype=prototype, **kwargs)
        self.fork_names = fork_names

    def _push_allocation_config(self):
        self.input_dims = {name: self.input_dim for name in self.fork_names}
        self.output_dims = self.fork_dims
        super(Fork, self)._push_allocation_config()

    @application(inputs=['input_'])
    def apply(self, input_):
        return super(Fork, self).apply(**{name: input_
                                          for name in self.fork_names})

    @apply.property('outputs')
    def apply_outputs(self):
        return super(Fork, self).apply.outputs


class Mixer(Parallel):
    """Mixes a new channel with old ones.

    .. digraph:: mixer

       rankdir=TB;
       splines=line;
       subgraph cluster_0 {
         label="old_inputs";
         a[label=""]; b[label=""]; c[label=""];
       }
       subgraph cluster_1 {
         label="outputs"; labelloc=b;
         d[label=""]; e[label=""]; f[label=""];
       }
       a -> d;
       b -> e;
       c -> f;
       new_input -> d;
       new_input -> e;
       new_input -> f;

    Parameters
    ----------
    old_names : list of str
        The names of the old channels.
    new_name : list of str
        The new channel's name.

    Attributes
    ----------
    channel_dims : dict of (channel_name, int) pairs
        The dimensions of the channels. Required for allocation.

    """
    def __init__(self, old_names, new_name, prototype=None, **kwargs):
        super(Mixer, self).__init__(old_names, prototype=prototype, **kwargs)
        self.old_names = old_names
        self.new_name = new_name

    def _push_allocation_config(self):
        self.input_dims = {name: self.channel_dims[self.new_name]
                           for name in self.old_names}
        self.output_dims = {name: self.channel_dims[name]
                            for name in self.old_names}
        super(Mixer, self)._push_allocation_config()

    @application
    def apply(self, **kwargs):
        new = kwargs.pop(self.new_name)
        if not set(kwargs.keys()) == set(self.old_names):
            raise ValueError
        result = super(Mixer, self).apply(
            return_list=True, **{name: new for name in self.old_names})
        for i, name in enumerate(self.old_names):
            result[i] += kwargs[name]
        return result

    @apply.property('inputs')
    def apply_inputs(self):
        return [self.new_name] + self.old_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.old_names
