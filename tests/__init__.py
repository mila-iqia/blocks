import os
from functools import update_wrapper


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
                        os.remove(file_)
        update_wrapper(wrapped_test, test)
        return wrapped_test
    return wrap_test
