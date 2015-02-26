import logging
import os
import sys
from functools import wraps
from importlib import import_module
from unittest.case import SkipTest

from six import StringIO

import blocks
from blocks import config


def silence_printing(test):
    @wraps(test)
    def wrapper(*args, **kwargs):
        stdout = sys.stdout
        sys.stdout = StringIO()
        logger = logging.getLogger(blocks.__name__)
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            test(*args, **kwargs)
        finally:
            sys.stdout = stdout
            logger.setLevel(old_level)
    return wrapper


def skip_if_not_available(modules=None, datasets=None, configurations=None):
    """Raises a SkipTest exception when requirements are not met.

    Parameters
    ----------
    modules : list
        A list of strings of module names. If one of the modules fails to
        import, the test will be skipped.
    datasets : list
        A list of strings of folder names. If the data path is not
        configured, or the folder does not exist, the test is skipped.
    configurations : list
        A list of of strings of configuration names. If this configuration
        is not set and does not have a default, the test will be skipped.

    """
    if modules is None:
        modules = []
    if datasets is None:
        datasets = []
    if configurations is None:
        configurations = []
    for module in modules:
        try:
            import_module(module)
        except Exception:
            raise SkipTest
        if module == 'bokeh':
            ConnectionError = import_module(
                'requests.exceptions').ConnectionError
            session = import_module('bokeh.session').Session()
            try:
                session.execute('get', session.base_url)
            except ConnectionError:
                raise SkipTest

    if datasets and not hasattr(config, 'data_path'):
        raise SkipTest
    for dataset in datasets:
        if not os.path.exists(os.path.join(config.data_path, dataset)):
            raise SkipTest
    for configuration in configurations:
        if not hasattr(config, configuration):
            raise SkipTest
