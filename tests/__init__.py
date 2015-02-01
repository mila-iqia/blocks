import os
import sys
import shutil
from functools import update_wrapper


class Discarder(object):
    def write(self, text):
        pass


def silence_printing(test):
    def wrapper(*args, **kwargs):
        stdout = sys.stdout
        sys.stdout = Discarder()
        try:
            test(*args, **kwargs)
        finally:
            sys.stdout = stdout
    update_wrapper(wrapper, test)
    return wrapper


def temporary_files(*files):
    def wrap_test(test):
        def wrapped_test(*args, **kwargs):
            if any(os.path.exists(file_) for file_ in files):
                raise IOError
            try:
                test(*args, **kwargs)
            finally:
                for file_ in files:
                    if os.path.exists(file_):
                        rm = (shutil.rmtree if os.path.isdir(file_)
                              else os.remove)
                        rm(file_)
        update_wrapper(wrapped_test, test)
        return wrapped_test
    return wrap_test
