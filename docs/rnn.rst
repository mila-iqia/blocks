Recurrent neural networks
=========================

.. warning::

    This section is very much work in progress!

Blocks offers native support for recurrent neural networks (RNNs).

Quickstart example
------------------

As a starting example, we'll be building an RNN which accumulates the input it
receives. The equation describing that RNN is

.. math:: \mathbf{h}_t = \mathbf{h}_{t-1} + \mathbf{x}_t

>>> import numpy
>>> import theano
>>> from theano import tensor
>>> from blocks import initialization
>>> from blocks.bricks import Identity
>>> from blocks.bricks.recurrent import SimpleRecurrent
>>> x = tensor.tensor3('x')
>>> recurrent = SimpleRecurrent(
...     dim=3, activation=Identity(), weights_init=initialization.Identity())
>>> recurrent.initialize()
>>> h = recurrent.apply(x)
>>> f = theano.function([x], h)
>>> print f(numpy.ones((3, 2, 3)))
[[[ 1.  1.  1.]
  [ 1.  1.  1.]]
<BLANKLINE>
 [[ 2.  2.  2.]
  [ 2.  2.  2.]]
<BLANKLINE>
 [[ 3.  3.  3.]
  [ 3.  3.  3.]]]

Let's modify that example so that the RNN accumulates two times the input it
receives:

.. math:: \mathbf{h}_t = \mathbf{h}_{t-1} + 2 \cdot \mathbf{x}_t

>>> from blocks.bricks import Linear
>>> doubler = Linear(
...     input_dim=3, output_dim=3, weights_init=initialization.Identity(2),
...     biases_init=initialization.Constant(0))
>>> doubler.initialize()
>>> h_doubler = recurrent.apply(doubler.apply(x))
>>> f = theano.function([x], h_doubler)
>>> print f(numpy.ones((3, 2, 3)))
[[[ 2.  2.  2.]
  [ 2.  2.  2.]]
<BLANKLINE>
 [[ 4.  4.  4.]
  [ 4.  4.  4.]]
<BLANKLINE>
 [[ 6.  6.  6.]
  [ 6.  6.  6.]]]

Note that in order to double the input we had to apply a :class:`.bricks.Linear`
brick to `x`, even though

.. math:: \mathbf{h}_t = f(\mathbf{V}\mathbf{h}_{t-1} + \mathbf{W}\mathbf{x}_t + \mathbf{b})

is what is usually thought of as the RNN equation. The reason why recurrent
bricks work that way is it allows greater flexibility and modularity:
:math:`\mathbf{W}\mathbf{x}_t` can be replaced by a whole neural network if we
want.

Initial states
--------------

Recurrent models all have in common that their initial state has to be
specified. However, in constructing our toy examples, we omitted to pass
:math:`\mathbf{h}_0` when applying the recurrent brick. What happened?

It turns out that recurrent bricks set that initial state to zero if it's not
passed as argument, which is a good sane default in most cases, but we can just
as well set it explicitly.

We will modify the starting example so that it accumulates the input it
receives, but starting from one instead of zero:

.. math:: \mathbf{h}_t = \mathbf{h}_{t-1} + \mathbf{x}_t, \quad \mathbf{h}_0 = 1

>>> h0 = tensor.matrix('h0')
>>> h = recurrent.apply(inputs=x, states=h0)
>>> f = theano.function([x, h0], h)
>>> print f(numpy.ones((3, 2, 3)), numpy.ones((2, 3)))
[[[ 2.  2.  2.]
  [ 2.  2.  2.]]
<BLANKLINE>
 [[ 3.  3.  3.]
  [ 3.  3.  3.]]
<BLANKLINE>
 [[ 4.  4.  4.]
  [ 4.  4.  4.]]]
