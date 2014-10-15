from itertools import chain

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


# NOT WORKING RIGHT NOW


class Model(object):
    def __init__(self, input, output, cost_block):
        self.input = input
        self.output = output
        self.cost_block = cost_block
        self.params = list(chain(*self.collect_tag([output], 'params')))
        self.block_inputs = self.collect_input(
            [self.output],
            lambda name: (name is not None and '_input_' in name))
        self.monitor_channels = \
            list(chain(*self.collect_tag([output], 'monitor_channels')))

    def grad(self, y):
        return tensor.grad(self.cost(y),
                           self.params)

    def cost(self, y):
        return self.cost_block.apply(y, self.output)

    def apply_dropout(self, include_prob=0.5, scale=2):
        rng = MRG_RandomStreams()
        masks = [rng.binomial(p=include_prob, size=var.shape,
                              dtype=var.dtype) for var in self.block_inputs]
        dropouts = [var * mask * scale
                    for var, mask in zip(self.block_inputs, masks)]
        return theano.clone(self.output,
                            replace=dict(zip(self.block_inputs, dropouts)))

    @staticmethod
    def collect_tag(outputs, tag):
        rval = []
        seen = set()

        def _collect_tag(outputs):
            for output in outputs:
                if output in seen:
                    continue
                seen.add(output)
                if hasattr(output.tag, tag):
                    rval.append(getattr(output.tag, tag))
                owner = output.owner
                if owner is None or owner in seen:
                    continue
                seen.add(owner)
                inputs = owner.inputs
                _collect_tag(inputs)
        _collect_tag(outputs)
        return rval

    @staticmethod
    def collect_input(outputs, name=lambda name: True):
        rval = []
        seen = set()

        def _collect_input(outputs):
            for output in outputs:
                if output in seen:
                    continue
                seen.add(output)
                if name(output.name):
                    rval.append(output)
                owner = output.owner
                if owner is None or owner in seen:
                    continue
                seen.add(owner)
                inputs = owner.inputs
                _collect_input(inputs)
        _collect_input(outputs)
        return rval
