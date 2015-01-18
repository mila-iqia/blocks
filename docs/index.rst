.. image:: https://coveralls.io/repos/bartvm/blocks/badge.svg?branch=master
   :target: https://coveralls.io/r/bartvm/blocks?branch=master

.. image:: https://travis-ci.org/bartvm/blocks.svg?branch=master
   :target: https://travis-ci.org/bartvm/blocks

.. image:: https://readthedocs.org/projects/blocks/badge/?version=latest
   :target: https://blocks.readthedocs.org/

|

Welcome to Blocks's documentation!
==================================
Blocks is a framework that helps you build neural network models on top of
Theano. Currently it supports and provides:

* Constructing parametrized Theano operations, called "bricks"
* Pattern matching to select variables and bricks in large models
* A pipeline for loading and iterating over training data
* Algorithms to optimize your model
* Automatic creation of monitoring channels (*limited support*)
* Application of graph transformations, such as dropout (*limited support*)

In the future we also hope to support:

* Saving and resuming of training
* Monitoring and analyzing values during training progress (on the training set
  as well as on test sets)
* Dimension, type and axes-checking

.. warning::
   Blocks is a new project which is still under development. As such, certain
   (all) parts of the framework are subject to change.

.. note::
   That said, if you are interested in using Blocks and run into any problems,
   don't hesitate to file bug reports, feature requests, or simply ask for help,
   by `making a GitHub issue`_.

.. _making a GitHub issue: https://github.com/bartvm/blocks/issues/new

Getting started
---------------
.. toctree::
   setup
   quickstart

In-depth
--------
.. toctree::
   bricks_overview
   configuration
   developer_guidelines

API Reference
-------------
.. toctree::
   bricks
   initialization
   datasets
   utils
   serialization
   graph

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
