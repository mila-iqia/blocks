import os
import struct

import numpy
import theano

from blocks import config
from blocks.datasets import InMemoryDataset, lazy_properties
from blocks.datasets.schemes import SequentialScheme
from blocks.utils import update_instance


MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049


@lazy_properties('data')
class MNIST(InMemoryDataset):
    """The MNIST dataset of handwritten digits.

    .. todo::

       Right now this dataset always returns flattened images. In order to
       support e.g. convolutions and visualization, it needs to support the
       original 28 x 28 image format.

    .. todo::

       The data path is hardcoded right now. A similar approach to Pylearn2
       should be adopted, where the data path can be configured in one way
       or the other, so that it can seamlessly load datasets from one
       directory (e.g. /data/lisa/data).

    Parameters
    ----------
    which_set : 'train' or 'test'
        Whether to load the training set (60,000 samples) or the test set
        (10,000 samples). Note that MNIST does not have a validation set;
        usually you will create your own training/validation split using
        the start and stop arguments.
    start : int, optional
        The first example to load
    stop : int, optional
        The last example to load
    binary : bool, optional
        If ``True``, returns binary (black/white) images instead of
        grayscale. ``False`` by default.

    """
    sources = ('features', 'targets')

    def __init__(self, which_set, start=None, stop=None, binary=False,
                 **kwargs):
        if which_set not in ('train', 'test'):
            raise ValueError("MNIST only has a train and test set")
        num_examples = (stop if stop else 60000) - (start if start else 0)
        default_scheme = SequentialScheme(num_examples, 1)
        update_instance(self, locals())
        super(MNIST, self).__init__(**kwargs)

    def load(self):
        if self.which_set == 'train':
            data = 'train-images-idx3-ubyte'
            labels = 'train-labels-idx1-ubyte'
        elif self.which_set == 'test':
            data = 't10k-images-idx3-ubyte'
            labels = 't10k-labels-idx1-ubyte'
        data_path = os.path.join(config.data_path, 'mnist')
        X = read_mnist_images(
            os.path.join(data_path, data),
            'bool' if self.binary
            else theano.config.floatX)[self.start:self.stop]
        X = X.reshape((X.shape[0], numpy.prod(X.shape[1:])))
        y = read_mnist_labels(
            os.path.join(data_path, labels))[self.start:self.stop,
                                             numpy.newaxis]
        self.data = {'features': X, 'targets': y}

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError("MNIST does not have a state")
        return tuple(self.data[source][request] for source in self.sources)


def read_mnist_images(filename, dtype=None):
    """Read MNIST images from the original ubyte file format.

    Parameters
    ----------
    filename : str
        Filename/path from which to read images.

    dtype : 'float32', 'float64', or 'bool'
        If unspecified, images will be returned in their original
        unsigned byte format.

    Returns
    -------
    images : ndarray, shape (n_images, n_rows, n_cols)
        An image array, with individual examples indexed along the
        first axis and the image dimensions along the second and
        third axis.

    Notes
    -----
    If the dtype provided was boolean, the resulting array will
    be boolean with `True` if the corresponding pixel had a value
    greater than or equal to 128, `False` otherwise.

    If the dtype provided was a float dtype, the values will be mapped to
    the unit interval [0, 1], with pixel values that were 255 in the
    original unsigned byte representation equal to 1.0.

    """
    with open(filename, 'rb') as f:
        magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError("Wrong magic number reading MNIST image file")
        array = numpy.fromfile(f, dtype='uint8').reshape((number, rows, cols))
    if dtype:
        dtype = numpy.dtype(dtype)

        if dtype.kind == 'b':
            # If the user wants booleans, threshold at half the range.
            array = array >= 128
        elif dtype.kind == 'f':
            # Otherwise, just convert.
            array = array.astype(dtype)
            array /= 255.
        else:
            raise ValueError("Unknown dtype to convert MNIST to")
    return array


def read_mnist_labels(filename):
    """Read MNIST labels from the original ubyte file format.

    Parameters
    ----------
    filename : str
        Filename/path from which to read labels.

    Returns
    -------
    labels : ndarray, shape (nlabels,)
        A one-dimensional unsigned byte array containing the
        labels as integers.

    """
    with open(filename, 'rb') as f:
        magic, number = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError("Wrong magic number reading MNIST label file")
        array = numpy.fromfile(f, dtype='uint8')
    return array
