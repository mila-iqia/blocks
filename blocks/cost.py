from abc import ABCMeta

from theano import tensor

from blocks.bricks import Brick


class Cost(Brick):
    pass


class CostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (batch, features).
    """
    __metaclass__ = ABCMeta

    @Brick.apply_method
    def apply(self, y, y_hat):
        self.cost_matrix._raw(y, y_hat).sum(axis=1).mean()


class BinaryCrossEntropy(CostMatrix):
    @Brick.apply_method
    def cost_matrix(self, y, y_hat):
        cost = tensor.nnet.binary_crossentropy(y_hat, y)
        return cost


class AbsoluteError(CostMatrix):
    @Brick.apply_method
    def cost_matrix(self, y, y_hat):
        cost = tensor.abs(y - y_hat)
        return cost


class SquaredError(CostMatrix):
    @Brick.apply_method
    def apply(self, y, y_hat):
        cost = tensor.sqr(y - y_hat)
        return cost
