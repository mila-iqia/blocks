"""Module-wide configuration settings.

Blocks allows module-wide configuration values to be set using a YAML
configuration file and environment variables. Environment variables
override the configuration file which in its turn overrides the defaults
(if they exist).

The configuration file to used can be set using the ``BLOCKS_CONFIG``
environment variable. Otherwise, the file ``~/.blocksrc`` is used if it
exists.

If a setting is not configured and does not provide a default, a
:class:`ConfigurationError` is raised when it is accessed.

Configuration values can be accessed as attributes of ``blocks.config``.

    >>> from blocks import config >>> print config.data_path

"""
import logging
import os

import yaml

logger = logging.getLogger(__name__)

NOT_SET = object()


class ConfigurationError(Exception):
    pass


class Configuration(object):
    def __init__(self, yaml_file=None):
        if 'BLOCKS_CONFIG' in os.environ:
            yaml_filename = os.environ['BLOCKS_CONFIG']
        else:
            yaml_filename = os.path.expanduser('~/.blocksrc')
        if os.path.isfile(yaml_filename):
            yaml_file = open(yaml_filename)
            self.yaml_settings = yaml.safe_load(yaml_file)
        else:
            self.yaml_settings = {}
        self.config = {}

    def __getattr__(self, key):
        if key not in self.config:
            raise ConfigurationError("Unknown configuration: {}".format(key))
        config = self.config[key]
        if config['env_var'] in os.environ:
            value = os.environ[config['env_var']]
        elif key in self.yaml_settings:
            value = self.yaml_settings[key]
        else:
            value = config['default']
        if value is NOT_SET:
            raise ConfigurationError("Configuration not set: {}.".format(key))
        elif type is None:
            return value
        else:
            return type(value)

    def add_config(self, key, default=NOT_SET, env_var=None, type=None):
        self.config[key] = {'default': default,
                            'env_var': env_var,
                            'type': type}

config = Configuration()

config.add_argument('data_path', env_var='BLOCKS_DATA_PATH', type=str)
