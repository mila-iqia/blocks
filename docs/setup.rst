Installation
============
The easiest way to install Blocks using the Python package manager ``pip``.
Blocks isn't listed yet on the Python Package Index (PyPI), so you will have to
grab it directly from GitHub.

.. code-block:: bash

   pip install --upgrade --no-deps git+git://github.com/bartvm/blocks.git --user

If you have administrative rights, remove ``--user`` to install the package
system-wide. The ``--no-deps`` flag is there to make sure that ``pip`` doesn't
try to update NumPy and Scipy, possibly overwriting the optimised version on
your system with a newer but slower version.

If you want to update Blocks, simply repeat the command above to pull the latest
version from GitHub.

Requirements
------------
Blocks' only requirements are Theano and six. We develop using the bleeding-edge
version of Theano, so be sure to follow the `relevant installation
instructions`_ to make sure that your Theano versino is up to date.

.. _relevant installation instructions: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions
