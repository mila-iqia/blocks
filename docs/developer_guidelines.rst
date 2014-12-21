Developer guidelines
====================

We want to encourage everyone to contribute to the development of Blocks. To
ensure the codebase is of high quality, we ask all new developers to have a
quick read through these rules to make sure that any code you contribute will be
easy to merge!

Code style
----------

Blocks follows the `PEP8 style guide`_ closely, so please make sure you are
familiar with it. Our `Travis CI buildbot`_ runs flake8_ as part of every build,
which checks for PEP8 compliance (using the pep8_ tool) and for some common
coding erros using pyflakes_. You might want to install and run flake8_ on your
code before submitting a PR to make sure that your build doesn't fail because of
e.g. a bit of extra whitespace.

Note that passing flake8_ does not necessarily mean that your code is PEP8
compliant! Some guidelines which aren't checked by flake8_:

* Imports `should be grouped`_ into standard library, third party, and local
  imports with a blank line in between groups.
* Variable names should be explanatory and unambiguous.

There are also some style guideline decisions that were made specifically for
Blocks:

* Do not rename imports i.e. do not use ``import theano.tensor as T`` or
  ``import numpy as np``.
* Direct imports, ``import ...``, preceed ``from ... import ...`` statements.
* Imports are otherwise listed alphabetically.
* Don't recycle variable names (i.e. don't use the same variable name to refer
  to different things in a particular part of code), especially when they are
  arguments to functions.

.. _PEP8 style guide: https://www.python.org/dev/peps/pep-0008/
.. _Travis CI buildbot: https://travis-ci.org/bartvm/blocks
.. _flake8: https://pypi.python.org/pypi/flake8
.. _pep8: https://pypi.python.org/pypi/pep8
.. _pyflakes: https://pypi.python.org/pypi/pyflakes
.. _should be grouped: https://www.python.org/dev/peps/pep-0008/#imports

Docstrings
----------

Blocks follows the `NumPy docstring standards`_. For a quick introduction, have
a look at the NumPy_ or Napoleon_ examples of compliant docstrings. A few common
mistakes to avoid:

* There is no line break after the opening quotes (``"""``).
* There is an empty line before the closing quotes (``"""``).
* The summary should not be more than one line.

The docstrings are formatted using reStructuredText_, and can make use of all
the formatting capabilities this provides. They are rendered into HTML
documentation using the `Read the Docs`_ service. After code has been merged,
please ensure that documentation was built successfully and that your docstrings
rendered as you intended by looking at the `online documentation`_, which is
automatically updated.

Writing doctests_ is encouraged, and they are run as part of the test suite.
They should be written to be Python 3 compliant.

.. _NumPy docstring standards: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _NumPy: https://github.com/numpy/numpy/blob/master/doc/example.py
.. _Napoleon: http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_numpy.html
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _doctests: https://docs.python.org/2/library/doctest.html
.. _Read the Docs: https://readthedocs.org/
.. _online documentation: http://blocks.readthedocs.org/

Unit testing
------------

Blocks uses unit testing to ensure that individual parts of the library behave
as intended. It's also essential in ensuring that parts of the library are not
broken by proposed changes.

All new code should be accompanied by extensive unit tests. Whenever a pull
request is made, the full test suite is run on `Travis CI`_, and pull requests
are not merged until all tests pass. Coverage analysis is performed using
coveralls_. Please make sure that at the very least your unit tests cover the
core parts of your committed code. In the ideal case, all of your code should be
unit tested.

The test suite can be executed locally using nose2_ [#]_.

.. [#] For all tests but the doctests, nose_ can also be used.

.. _Travis CI: https://travis-ci.org/bartvm/blocks
.. _coveralls: https://coveralls.io/r/bartvm/blocks
.. _nose2: https://readthedocs.org/projects/nose2/
.. _nose: http://nose.readthedocs.org/en/latest/

Python 2 and 3
--------------

Blocks aims to be both Python 2 and Python 3 compliant using a single code-base,
without using 2to3_. There are many online resources which discuss the writing
compatible code. For a quick overview see `the cheatsheet from Python
Charmers`_.  For non-trivial cases, we use the six_ compatibility library.

Documentation should be written to be Python 3 compliant.

.. _2to3: https://docs.python.org/2/library/2to3.html
.. _the cheatsheet from Python Charmers: http://python-future.org/compatible_idioms.html
.. _six: https://pythonhosted.org/six/
