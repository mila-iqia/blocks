"""Objects for encapsulating parameter initialization strategies."""
from abc import ABCMeta, abstractmethod
import numpy as np
import theano


class NdarrayInitialization(object):
    """Base class specifying the interface for ndarray initialization."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def initialize(self, rng, shape):
        """
        Generate an initial set of parameters from a given distribution.

        Parameters
        ----------
        rng : object
            A `numpy.random.RandomState`.
        shape : tuple
            A shape tuple for the requested parameter array shape.

        Returns
        -------
        ndarray
            An ndarray with values drawn from the distribution specified by
            this object, of shape `shape`, with dtype
            `theano.config.floatX`.
        """
        pass


class Constant(NdarrayInitialization):
    """
    Initialize parameters to a constant. The constant may be a scalar or an
    array_like of any shape that is broadcastable with the requested
    parameter arrays.

    Parameters
    ----------
    constant : array_like
        The initialization value to use. Must be a scalar or an ndarray (or
        compatible object, such as a nested list) that has a shape that is
        broadcastable with any shape requested by `initialize`.
    """
    def __init__(self, constant):
        self._constant = np.asarray(constant)

    def initialize(self, rng, shape):
        dest = np.empty(shape, dtype=theano.config.floatX)
        dest[...] = self._constant
        return dest


class IsotropicGaussian(NdarrayInitialization):
    """
    Initialize parameters from an isotropic Gaussian distribution.

    Parameters
    ----------
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    """
    def __init__(self, mean=0, std=1):
        self._mean = mean
        self._std = std

    def initialize(self, rng, shape):
        m = rng.normal(self._mean, self._std, size=shape)
        return m.astype(theano.config.floatX)


class Uniform(NdarrayInitialization):
    """
    Initialize parameters from a uniform distribution.

    Parameters
    ----------
    mean : float, optional
        The mean of the uniform distribution (i.e. the center of mass for
        the density function); Defaults to 0.
    width : float, optional
        One way of specifying the range of the uniform distribution. The
        support will be [mean - width/2, mean + width/2]. **Exactly one**
        of `width` or `std` must be specified.
    std : float, optional
        An alternative method of specifying the range of the uniform
        distribution. Chooses the width of the uniform such that random
        variates will have a desired standard deviation. **Exactly one** of
        `width` or `std` must be specified.

    """
    def __init__(self, mean=0., width=None, std=None):
        if (width is not None) == (std is not None):
            raise ValueError("must specify width or std, "
                             "but not both")
        if std is not None:
            # Variance of a uniform is 1/12 * width^2
            self._width = np.sqrt(12) * std
        else:
            self._width = width
        self._mean = mean

    def initialize(self, rng, shape):
        w = self._width / 2
        m = rng.uniform(self._mean - w, self._mean + w, size=shape)
        return m.astype(theano.config.floatX)
