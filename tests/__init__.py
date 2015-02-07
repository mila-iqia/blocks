import sys
from functools import wraps

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
