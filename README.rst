.. image:: https://coveralls.io/repos/bartvm/blocks/badge.svg?branch=master
   :target: https://coveralls.io/r/bartvm/blocks?branch=master

.. image:: https://travis-ci.org/bartvm/blocks.svg?branch=master
   :target: https://travis-ci.org/bartvm/blocks

.. image:: https://readthedocs.org/projects/blocks/badge/?version=latest
   :target: https://blocks.readthedocs.org/

.. image:: https://landscape.io/github/bartvm/blocks/master/landscape.svg
   :target: https://landscape.io/github/bartvm/blocks/master

Blocks
======
Blocks is a framework that helps you build neural network models on top of
Theano. Currently it supports and provides:

* Constructing parametrized Theano operations, called "bricks"
* Pattern matching to select variables and bricks in large models
* A pipeline for loading and iterating over training data
* Algorithms to optimize your model
* Automatic creation of monitoring channels (*limited support*)
* Application of graph transformations, such as dropout (*limited support*)

In the feature we also hope to support:

* Saving and resuming of training
* Monitoring and analyzing values during training progress (on the training set
  as well as on test sets)
* Dimension, type and axes-checking

Please see the documentation_ for more information.

If you want to contribute, please make sure to read the `developer guidelines`_.

.. _documentation: http://blocks.readthedocs.org
.. _developer guidelines: http://blocks.readthedocs.org/en/latest/developer_guidelines.html
