"""Module level configuration."""
import logging
import os

import yaml

logger = logging.getLogger(__name__)

NOT_SET = object()


class ConfigurationError(Exception):
    pass


class Configuration(object):
    def __init__(self):
        self.config = {}

    def load_yaml(self):
        if 'BLOCKS_CONFIG' in os.environ:
            yaml_file = os.environ['BLOCKS_CONFIG']
        else:
            yaml_file = os.path.expanduser('~/.blocksrc')
        if os.path.isfile(yaml_file):
            with open(yaml_file) as f:
                for key, value in yaml.safe_load(f).items():
                    if key not in self.config:
                        raise ValueError("Unrecognized config in YAML: {}"
                                         .format(key))
                    self.config[key]['yaml'] = value

    def __getattr__(self, key):
        if key == 'config' or key not in self.config:
            raise AttributeError
        config = self.config[key]
        if 'value' in config:
            value = config['value']
        elif 'env_var' in config and config['env_var'] in os.environ:
            value = os.environ[config['env_var']]
        elif 'yaml' in config:
            value = config['yaml']
        elif 'default' in config:
            value = config['default']
        else:
            raise ConfigurationError("Configuration not set and no default "
                                     "provided: {}.".format(key))
        return config['type'](value)

    def __setattr__(self, key, value):
        if key != 'config' and key in self.config:
            self.config[key]['value'] = value
        else:
            super(Configuration, self).__setattr__(key, value)

    def add_config(self, key, type_, default=NOT_SET, env_var=None):
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
        self.config[key] = {'type': type_}
        if env_var is not None:
            self.config[key]['env_var'] = env_var
        if default is not NOT_SET:
            self.config[key]['default'] = default

config = Configuration()

# Define configuration options
config.add_config('data_path', type_=str, env_var='BLOCKS_DATA_PATH')
config.add_config('default_seed', type_=int, default=1)

config.load_yaml()
