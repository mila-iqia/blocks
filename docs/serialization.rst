Serialization
=============

The ability to save models and their training progress is important for two
reasons:

1. Neural nets can take days or even weeks to train. If training is
   interrupted during this time, it is important that we can continue from
   where we left off.
2. We need the ability to save models in order to share them with others or save
   them for later use or inspection.

These two goals come with differing requirements, which is why Blocks implements
two serialization methods.

Pickling the training loop
--------------------------

.. warning::

   Due to the complexity of serializing a Python objects as large as the main
   loop, pickling will sometimes fail because it exceeds the default maximum
   recursion depth set in Python. Please make sure that you always have backup
   of your pickled main loop before resuming training.

The first approach used is to pickle the entire :class:`main loop
<.MainLoop>`, effectively serializing the exact state of the model as well
as training. Techncially there are some difficulties with this approach:

* Some Python objects cannot be pickled e.g. file handles, generators,
  dynamically generated classes, nested classes, etc.
* The pickling of Theano objects can be problematic.
* We do not want to serialize the training data kept in memory, since this can
  be prohibitively large.

Blocks addresses these problems by using a pickling extension called Dill,
avoiding certain data structures such as generators and nested classes (see the
:ref:`developer guidelines <serialization_guidelines>`), and by overriding the
pickling behavior of datasets.

However, in general you should not rely on this serialization approach for the
long term saving of your models. Problems that remain are

* Unpickling depends on the libraries used being unchanged. This means that if
  you updated Blocks, Theano, etc. to a new version where the interface has
  changed, loading your training progress will fail.
* The unpickling of Theano objects can be problematic, especially when
  transferring from GPU to CPU or vice versa.
* It is not possible on Python 2 to unpickle objects that were pickled in Python
  3.

.. note::

   On the long term, we plan to serialize the log, data stream, and the rest of
   the main loop separately. This way you can e.g. perform plotting without
   needing to deserialize the Theano model.

Parameter saving
----------------

The second method used by Blocks is intended to be more cross-platform, and a
safer way of storing models for a longer period of time. This method:

* Stores the parameters in a binary NumPy file (``.npz``)
* Serializes the log
* Serializes the data stream

When resuming training, the model is reconstructed after which the parameters
can be reloaded from the NumPy file. The training log and data stream are loaded
as well, allowing the training to continue. However, this method makes no effort
to try and store the exact state of training. This means that:

* Training algorithms that are stateful e.g. those that use moving averages or
  keep any sort of history that is not part of the log (ADADELTA, momentum,
  L-BFGS, etc.) will reset.
* Training extensions will be reset as well.
* You will need to reconstruct the Theano graph before the parameters are
  reloaded. This means that you will need the original script.
