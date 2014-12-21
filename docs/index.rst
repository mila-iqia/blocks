.. image:: https://coveralls.io/repos/bartvm/blocks/badge.png?branch=master
   :target: https://coveralls.io/r/bartvm/blocks?branch=master

.. image:: https://travis-ci.org/bartvm/blocks.svg?branch=master
   :target: https://travis-ci.org/bartvm/blocks

.. image:: https://readthedocs.org/projects/blocks/badge/?version=latest
   :target: https://blocks.readthedocs.org/
   :alt: Documentation Status

|

Welcome to Blocks's documentation!
==================================

Blocks is a framework that helps you build neural network models on top of
Theano. It also helps you manage your model by doing error-checking, creating
monitoring channels, and allowing for easy configuration of your model. Features
include:

* Dimension, type and axes-checking
* Automatic creation of monitoring channels
* Easy pattern matching to select the bricks you want in large graphs
* Lazy initialization of bricks
* Application of graph transformations, such as dropout

Table of contents
-----------------

.. toctree::

   getting_started
   blocks
   initialization
   utils
   serialization
   graph
   developer_guidelines

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
