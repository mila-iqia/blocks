import copy

from blocks.bricks import Brick, lazy, application, MLP, Identity
from blocks.utils import update_instance


class Parallel(Brick):
    """Apply similar transformations to several channels.

    Parameters
    ----------
    channel_names : list of str
        Input names.
    input_dims : dict
        Dictonary of input dimensions, keys are channel names, values are
        dimensions.
    output_dims : dict
        Dictonary of output dimensions, keys are channel names, values are
        dimensions.
    prototype : :class:`Brick`
        A transformation prototype. A copy will be created for every channel.
        If ``None``, a linear transformation is used.

    """
    @lazy
    def __init__(self, channel_names, input_dims, output_dims,
                 prototype=None, weights_init=None, biases_init=None,
                 **kwargs):
        super(Parallel, self).__init__(**kwargs)
        update_instance(self, locals())

        if not self.prototype:
            self.prototype = MLP([Identity()], use_bias=False)
        self.transforms = []
        for name in self.channel_names:
            self.transforms.append(copy.deepcopy(self.prototype))
            self.transforms[-1].name = "transform_{}".format(name)
        self.children = self.transforms

    def _push_allocation_config(self):
        for name, transform in zip(self.channel_names, self.transforms):
            transform.dims[0] = self.input_dims[name]
            transform.dims[-1] = self.output_dims[name]

    def _push_initialization_config(self):
        for child in self.children:
            if self.weights_init:
                child.weights_init = self.weights_init
            if self.biases_init:
                child.biases_init = self.biases_init

    @application
    def apply(self, **kwargs):
        return [transform.apply(kwargs[name])
                for name, transform
                in zip(self.channel_names, self.transforms)]

    @apply.property('inputs')
    def apply_inputs(self):
        return self.channel_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.channel_names


class Fork(Parallel):
    """Forks single input into multiple channels.

    Parameters
    ----------
    fork_names : list of str
        Names of the channels to fork.
    prototype : instance of :class:`Brick`
        A prototype for the input-to-fork transformations. A copy will be
        created for every output channel.

    Notes
    -----
        Currently works only with lazy initialization
        (can not be initialized with a single constructor call).

    """
    def __init__(self, fork_names, prototype=None, **kwargs):
        super(Fork, self).__init__(fork_names, prototype=prototype, **kwargs)
        update_instance(self, locals())

    def _push_allocation_config(self):
        self.input_dims = {name: self.input_dim for name in self.fork_names}
        self.output_dims = self.fork_dims
        super(Fork, self)._push_allocation_config()

    @application(inputs='inp')
    def apply(self, inp):
        return super(Fork, self).apply(**{name: inp
                                          for name in self.fork_names})

    @apply.property('outputs')
    def apply_outputs(self):
        return super(Fork, self).apply.outputs
