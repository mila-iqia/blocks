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
a bit of extra whitespace.

Note that passing flake8_ does not necessarily mean that your code is PEP8
compliant. Some guidelines which aren't checked by flake8_:

* Imports `should be grouped`_ into standard library, third party, and local
  imports.
* Variable names should be explanatory. Please avoid using abbreviations, unless
  there is an abbreviation which is truly universal and saves a significant case
  of space. For example, use ``variable`` instead of ``var``, ``allocate``
  instead of ``alloc``, ``initialization`` instead of ``init``, etc.

.. _PEP8 style guide: https://www.python.org/dev/peps/pep-0008/
.. _Travis CI buildbot: https://travis-ci.org/bartvm/blocks
.. _flake8: https://pypi.python.org/pypi/flake8
.. _pep8: https://pypi.python.org/pypi/pep8
.. _pyflakes: https://pypi.python.org/pypi/pyflakes
.. _should be grouped: https://www.python.org/dev/peps/pep-0008/#imports
