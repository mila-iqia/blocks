import numpy

import theano
from theano import tensor

from groundhog.mainLoop import MainLoop
from groundhog.trainer.SGD import SGD

from blocks.bricks import Brick, GatedRecurrent, Tanh
from blocks.sequence_generators import SequenceGenerator, SimpleReadout
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.groundhog.state import GroundhogState
from blocks.groundhog.model import GroundhogModel

floatX = theano.config.floatX


class Readout(SimpleReadout):

    def __init__(self):
        super(Readout, self).__init__(readout_dim=1,
                                        source_names=['states'])

    @Brick.apply_method
    def cost(self, readouts, outputs):
        return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)


class SeriesModel(GroundhogModel):

    def __init__(self):
        self.transition = GatedRecurrent(
            name="transition", activation=Tanh(), dim=10,
            weights_init=Orthogonal())
        self.generator = SequenceGenerator(
            Readout(), self.transition,
            weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
            name="generator")
        self.generator.initialize()

        self.x = tensor.tensor3('x')
        super(SeriesModel, self).__init__(
            [self.generator], self.generator.cost(self.x).sum())


class SeriesIterator(object):

    def __init__(self, rng, func, seq_len, batch_size):
        self.__dict__.update(**locals())
        del self.self

    def start(self, offset):
        pass

    def next(self):
        T = self.rng.uniform(0, self.seq_len, (self.batch_size,))

        x = numpy.zeros((self.seq_len, self.batch_size, 1), dtype=floatX)
        for i in range(self.seq_len):
            x[i, :, 0] = self.func(T + i)

        return dict(x=x)

    @property
    def next_offset(self):
        return 0


def main():
    seed = 1
    rng = numpy.random.RandomState(seed)
    batch_size = 10

    data = SeriesIterator(rng, lambda x : numpy.sin(x), 100, batch_size)
    model = SeriesModel()
    state = GroundhogState("sine", batch_size, 0.0001).as_dict()
    trainer = SGD(model, state, data)
    main_loop = MainLoop(data, None, None, model, trainer, state, None)

    main_loop.main()

if __name__ == "__main__":
    main()
