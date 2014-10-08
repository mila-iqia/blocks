"""The blocks library for parameterized Theano ops"""
import numpy as np

BLOCK_PREFIX = 'block'
DEFAULT_SEED = [2014, 10, 5]
SEPARATOR = '_'


class Block(object):
    """Blocks are groups of bricks with a particular function.

    Bricks are disjoint subsets of the computational graph, which can
    potentially be reused several times throughout the graph. Blocks
    collect connected groups of bricks into groups. However, unlike bricks,
    blocks can be nested. This allows you to organise your model in a
    tree-like structure of nested blocks. In order to keep the
    tree-structure well-defined, blocks cannot be reused.

    Like bricks, blocks have `apply` methods which take Theano variables as
    input and output. Blocks can perform a lot of operations for you, like
    inferring what the input dimensions should be, or combining bricks in
    otherwise complicated ways.

    Within a block, the names of its children (blocks and bricks) need to
    be unique.

    Parameters
    ----------
    name : str, optional
        The name of this brick. This can be used to filter the application
        of certain modifications by block names. By default the block
        receives the name of its class (lowercased). If this block is
        nested, the name is expected to be unique within its parent block.
    rng : object, optional
        A `numpy.random.RandomState` object. This RNG will be passed to all
        its children.
    initialize : bool, optional
        If `True` then the parameters of this brick will automatically be
        allocated and initialized by calls to the :meth:`allocate` and
        :meth:`initialize`. If `False` these methods need to be called
        manually after initializing. Defaults to `True`.

    Attributes
    ----------
    children : list of objects
        List of Block and Brick instances which belong to this class.
        Blocks can be expected to be children of this class alone, while
        bricks can also be part of other blocks.

    """
    def __init__(self, name=None, rng=None, initialize=True):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}{}{}'.format(BLOCK_PREFIX, SEPARATOR, name)
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.initialize = initialize
