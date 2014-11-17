from __future__ import absolute_import, print_function

import doctest
import importlib
import pkgutil

import blocks
import blocks.bricks


def setup(testobj):
    # Not importing unicode_literal because it gives problems
    # If needed, see https://dirkjan.ochtman.nl/writing/2014/07/06/
    # single-source-python-23-doctests.html for a solution
    testobj.globs['absolute_import'] = absolute_import
    testobj.globs['print_function'] = print_function


def load_tests(loader, tests, ignore):
    # This function loads doctests from all submodules and runs them
    # with the __future__ imports necessary for Python 2
    for _, module, _ in pkgutil.walk_packages(path=blocks.__path__,
                                              prefix=blocks.__name__ + '.'):
        try:
            tests.addTests(doctest.DocTestSuite(
                module=importlib.import_module(module), setUp=setup,
                optionflags=doctest.IGNORE_EXCEPTION_DETAIL))
        except:
            pass
    return tests
