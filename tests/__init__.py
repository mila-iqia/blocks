import os
import sys
import shutil
from functools import wraps
from unittest import SkipTest

from six import StringIO


def silence_printing(test):
    @wraps(test)
    def wrapper(*args, **kwargs):
        stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            test(*args, **kwargs)
        finally:
            sys.stdout = stdout
    return wrapper


def temporary_files(*files):
    def wrap_test(test):
        @wraps(test)
        def wrapped_test(*args, **kwargs):
            if any(os.path.exists(file_) for file_ in files):
                raise SkipTest
            try:
                test(*args, **kwargs)
            finally:
                for file_ in files:
                    if os.path.exists(file_):
                        rm = (shutil.rmtree if os.path.isdir(file_)
                              else os.remove)
                        rm(file_)
        return wrapped_test
    return wrap_test
