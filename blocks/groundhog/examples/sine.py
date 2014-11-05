import numpy
import argparse
from matplotlib import pyplot

import theano
from theano import tensor

from groundhog.mainLoop import MainLoop
from groundhog.trainer.SGD import SGD

from blocks.bricks import Brick, GatedRecurrent, Tanh
from blocks.model import Model
from blocks.sequence_generators import SequenceGenerator, SimpleReadout
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.groundhog import GroundhogIterator, GroundhogState, GroundhogModel

floatX = theano.config.floatX


class Readout(SimpleReadout):

    def __init__(self):
        super(Readout, self).__init__(readout_dim=1,
                                      source_names=['states'])

    @Brick.apply_method
    def cost(self, readouts, outputs):
        return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)


class SeriesModel(Model):

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


class SeriesIterator(GroundhogIterator):

    def __init__(self, rng, func, seq_len, batch_size):
        self.__dict__.update(**locals())
        del self.self

    def next(self):
        T = self.rng.uniform(0, self.seq_len, (self.batch_size,))

        x = numpy.zeros((self.seq_len, self.batch_size, 1), dtype=floatX)
        for i in range(self.seq_len):
            x[i, :, 0] = self.func(T + i)

        return dict(x=x)


def main():
    parser = argparse.ArgumentParser(
        "Case study of generating simple sequences with RNN")
    parser.add_argument("mode", choices=["train", "plot"])
    parser.add_argument("prefix", default="sine")
    args = parser.parse_args()

    model = SeriesModel()
    gh_model = GroundhogModel(model)

    if args.mode == "train":
        seed = 1
        rng = numpy.random.RandomState(seed)
        batch_size = 10

        data = SeriesIterator(rng, lambda x: numpy.sin(x), 100, batch_size)
        state = GroundhogState(args.prefix, batch_size, learning_rate=0.0001).as_dict()
        trainer = SGD(gh_model, state, data)
        main_loop = MainLoop(data, None, None, gh_model, trainer, state, None)
        main_loop.main()
    else:
        gh_model.load(args.prefix + "model.npz")
        sample = theano.function([], model.generator.generate(
            n_steps=100, batch_size=10, iterate=True))
        x = sample()[0]
        pyplot.plot(x[..., 0])
        pyplot.show()


if __name__ == "__main__":
    main()
