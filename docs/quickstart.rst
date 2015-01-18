Quickstart
==========

In this quick-start tutorial we will use the Blocks framework to train a
`multilayer perceptron`_ (MLP) to perform handwriting recognition on the `MNIST
handwritten digit database`_.

The Task
--------
MNIST is a dataset which consists of 70,000 handwritten digits. Each digit is a
grayscale image of 28 by 28 pixels. Our task is to classify each of the images
into one of the 10 categories representing the numbers from 0 to 9.

.. figure:: _static/mnist.png
   :align: center

   Sample MNIST digits

The Model
---------
We will train a simple MLP with a single hidden layer that uses the rectifier_
activation function. Our output layer will consist of a softmax_ function with
10 units; one for each class. Mathematically speaking, our model is parametrized
by the weight matrices :math:`\mathbf{W}_h` and :math:`\mathbf{W}_y`, and bias
vectors :math:`\mathbf{b}_h` and :math:`\mathbf{b}_y`. The rectifier activation
function is defined as

.. math:: \mathrm{ReLU}(\mathbf{x})_i = \max(0, \mathbf{x}_i)

and our softmax output function is defined

.. math:: \mathrm{softmax}(\mathbf{x})_i = \frac{e^{\mathbf{x}_i}}{\sum_{j=1}^n \mathbf{x}_j}

Hence, our complete model is

.. math:: f(\mathbf{x}) = \mathrm{softmax}(\mathbf{W}_y\mathrm{ReLU}(\mathbf{W}_h\mathbf{x} + \mathbf{b}_h) + \mathbf{b}_y)

Since the output of a softmax represents a categorical probability distribution
we can consider :math:`f(\mathbf{x}) = \hat p(\mathbf{y} \mid \mathbf{x})`,
where :math:`\mathbf{x}` is the 784-dimensional (28 × 28) input, and
:math:`\mathbf{y}` the probability distribution of it belonging to classes
:math:`i = 0,\dots,9`. We can train the parameters of our model by minimizing
the negative log-likelihood i.e.  the categorical cross-entropy between our
model's output and the target distribution. That is, we minimize the sum of

.. math:: - \log \sum_{i=0}^{10} p(\mathbf{y} = i) \hat p(\mathbf{y} = i \mid \mathbf{x})

over all examples. We do so by using `stochastic gradient descent`_ (SGD) on
mini-batches.

Building the model
------------------
Constructing the model with Blocks is very simple. We start by defining the
input variable using Theano.

.. tip::
   Want to follow along with the Python code? If you are using IPython, enable
   the `doctest mode`_ using the special ``%doctest_mode`` command so that you
   can copy-paste the examples below (including the ``>>>`` prompts) straight
   into the IPython interpreter.

>>> from theano import tensor
>>> x = tensor.matrix('features')

Note that we picked the name ``'features'`` for our input. This is important,
because the name needs to match the name of the data source we want to train on.
MNIST defines two data sources: ``'features'`` and ``'targets'``.

For the sake of this tutorial, we will go through building an MLP the long way.
For a much quicker way, skip right to the end of this section. We begin with
applying the linear transformations and activations.

>>> from blocks.bricks import Linear, Rectifier, Softmax
>>> input_to_hidden = Linear(name='input_to_hidden', input_dim=784, output_dim=100)
>>> h = Rectifier().apply(input_to_hidden.apply(x))
>>> hidden_to_output = Linear(name='hidden_to_output', input_dim=100, output_dim=10)
>>> y_hat = Softmax().apply(hidden_to_output.apply(h))

Blocks' uses "bricks" to build models. Bricks are parametrized Theano ops. What
this means is that we start by initializing them with certain parameters e.g.
``input_dim``. After initialization we can apply our bricks on Theano variables
to build the model we want.

Now that we have built our model, let's define the cost to minimize. For this,
we will need the Theano variable representing the target labels.

>>> y = tensor.lmatrix('targets')
>>> from blocks.bricks.cost import CategoricalCrossEntropy
>>> cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

That's it! But creating a simple MLP this way is rather cumbersome. In practice,
we would have simply used the :class:`~blocks.bricks.MLP` class.

>>> from blocks.bricks import MLP
>>> mlp = MLP(activations=[Rectifier(), Softmax()], dims=[784, 100, 10]).apply(x)

Training your model
-------------------
Besides helping you build models, Blocks also provides the main other features
needed to train a model. It has a set of training algorithms (like SGD), an
interface to datasets, and a training loop that allows you to monitoring and
control the training process.

We want to train our model on the training set of MNIST.

>>> from blocks.datasets.mnist import MNIST
>>> mnist = MNIST("train")

Datasets only provide an interface to the data. For actual training, we will
need to iterate over the data in minibatches. This is done by initiating a data
stream which makes use of a particular iteration scheme. We will use an
iteration scheme that iterates over our MNIST examples sequentially in batches
of size 256.

>>> from blocks.datasets import DataStream
>>> from blocks.datasets.schemes import SequentialScheme
>>> data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
...     num_examples=mnist.num_examples, batch_size=256))

As our algorithm we will use straightforward SGD with a fixed learning rate.

>>> from blocks.algorithms import GradientDescent, SteepestDescent
>>> algorithm = GradientDescent(cost=cost, step_rule=SteepestDescent(learning_rate=0.1))

That's all we need! We can use the :class:`~blocks.main_loop.MainLoop` to
combine all the different pieces. Let's train our model for a single epoch and
print the progress to see how it works.

>>> from blocks.main_loop import MainLoop
>>> from blocks.extensions import FinishAfter, Printing
>>> main_loop = MainLoop(model=mlp, data_stream=data_stream, algorithm=algorithm,
...                      extensions=[FinishAfter(after_n_epochs=1), Printing()])
>>> main_loop.run() # doctest: +SKIP
-------------------------------------------------------------------------------
BEFORE FIRST EPOCH
-------------------------------------------------------------------------------
Training status:
     iterations_done: 0
     epochs_done: 0
Log records from the iteration 0:
-------------------------------------------------------------------------------
AFTER ANOTHER EPOCH
-------------------------------------------------------------------------------
Training status:
     iterations_done: 235
     epochs_done: 1
Log records from the iteration 235:
     training_finish_requested: True
-------------------------------------------------------------------------------
TRAINING HAS BEEN FINISHED:
-------------------------------------------------------------------------------
Training status:
     iterations_done: 235
     epochs_done: 1
Log records from the iteration 235:
     training_finish_requested: True
     training_finished: True

.. _multilayer perceptron: https://en.wikipedia.org/wiki/Multilayer_perceptron
.. _MNIST handwritten digit database: http://yann.lecun.com/exdb/mnist/
.. _rectifier: https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29
.. _softmax: https://en.wikipedia.org/wiki/Softmax
.. _stochastic gradient descent: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
.. _doctest mode: http://ipython.org/ipython-doc/dev/interactive/tips.html#run-doctests
