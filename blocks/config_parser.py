"""Module level configuration.

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

.. _YAML: http://yaml.org/
.. _environment variables:
   https://en.wikipedia.org/wiki/Environment_variable

"""
import logging
import os

import yaml

logger = logging.getLogger(__name__)

NOT_SET = object()


class ConfigurationError(Exception):
    pass


class Configuration(object):
    def __init__(self):
        if 'BLOCKS_CONFIG' in os.environ:
            yaml_file = os.environ['BLOCKS_CONFIG']
        else:
            yaml_file = os.path.expanduser('~/.blocksrc')
        if os.path.isfile(yaml_file):
            with open(yaml_file) as f:
                self.yaml_settings = yaml.safe_load(f)
        else:
            self.yaml_settings = {}
        self.config = {}

    def __getattr__(self, key):
        if key not in self.config:
            raise ConfigurationError("Unknown configuration: {}".format(key))
        config = self.config[key]
        if config['env_var'] is not None and config['env_var'] in os.environ:
            value = os.environ[config['env_var']]
        elif key in self.yaml_settings:
            value = self.yaml_settings[key]
        else:
            value = config['default']
        if value is NOT_SET:
            raise ConfigurationError("Configuration not set and no default "
                                     "provided: {}.".format(key))
        return config['type'](value)

    def add_config(self, key, type, default=NOT_SET, env_var=None):
        """Add a configuration setting.

        Parameters
        ----------
        key : str
            The name of the configuration setting. This must be a valid
            Python attribute name i.e. alphanumeric with underscores.
        type : function
            A function such as ``float``, ``int`` or ``str`` which takes
            the configuration value and returns an object of the correct
            type.  Note that the values retrieved from environment
            variables are always strings, while those retrieved from the
            YAML file might already be parsed. Hence, the function provided
            here must accept both types of input.
        default : object, optional
            The default configuration to return if not set. By default none
            is set and an error is raised instead.
        env_var : str, optional
            The environment variable name that holds this configuration
            value. If not given, this configuration can only be set in the
            YAML configuration file.

        """
        self.config[key] = {'default': default,
                            'env_var': env_var,
                            'type': type}

config = Configuration()
config.add_config('data_path', env_var='BLOCKS_DATA_PATH', type=str)
