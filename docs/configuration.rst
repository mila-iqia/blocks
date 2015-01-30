Configuration
=============

Blocks allows module-wide configuration values to be set using a YAML_
configuration file and `environment variables`_. Environment variables
override the configuration file which in its turn overrides the defaults.

The configuration is read from ``~/.blocksrc`` if it exists. A custom
configuration file can be used by setting the ``BLOCKS_CONFIG`` environment
variable. A configuration file is of the form:

.. code-block:: yaml

   data_path: /home/user/datasets

Which could be overwritten by using environment variables:

.. code-block:: bash

   BLOCKS_DATA_PATH=/home/users/other_datasets python

If a setting is not configured and does not provide a default, a
:class:`ConfigurationError` is raised when it is accessed.

Configuration values can be accessed as attributes of ``blocks.config``.

    >>> from blocks import config
    >>> print(config.data_path) # doctest: +SKIP
    '~/datasets'

The following configurations are supported:

.. option:: data_path

   The path where dataset files are stored. Can also be set using the
   environment variable ``BLOCKS_DATA_PATH``.

.. option:: default_seed

   The seed used when initializing random number generators (RNGs) such as NumPy
   ``RandomState`` objects as well as Theano's ``RandomStreams``objects. Must be
   an integer. By default this is set to 1.

.. _YAML: http://yaml.org/
.. _environment variables:
   https://en.wikipedia.org/wiki/Environment_variable

