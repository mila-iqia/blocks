Installation
============
The easiest way to install Blocks using the Python package manager pip.  Blocks
isn't listed yet on the Python Package Index (PyPI), so you will have to grab
it directly from GitHub.

.. code-block:: bash

   pip install --upgrade git+git://github.com/bartvm/blocks.git#egg=blocks --user

If you want to make sure that you can run the tests and/or use the plotting
integration with Bokeh_, install the extra requirements as well.

.. code-block:: bash

   pip install --upgrade git+git://github.com/bartvm/blocks.git#egg=blocks[plot,test] --user

If you have administrative rights, remove ``--user`` to install the package
system-wide. If you want to update Blocks, simply repeat one of the commands
above to pull the latest version from GitHub.

.. warning::

   Pip may try to update your versions of NumPy and SciPy if they are outdated.
   However, pip's versions might not be linked to an optimized BLAS
   implementation. To prevent this from happening use the ``--no-deps`` flag
   when installing Blocks and install the dependencies manually, making sure
   that you install NumPy and SciPy using your system's package manager (e.g.
   ``apt-get`` or ``yum``), or use a Python distribution like Anaconda_.

Requirements
------------
Blocks' requirements are

* Theano_, for pretty much everything
* dill_, for training progress serialization
* PyYAML_, to parse the configuration file
* six_, to support both Python 2 and 3 with a single codebase

There are also some optional requirements

* nose2_, to run the test suite
* Bokeh_, for live plotting of your training

We develop using the bleeding-edge version of Theano, so be sure to follow the
`relevant installation instructions`_ to make sure that your Theano version is
up to date.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _nose2: https://nose2.readthedocs.org/en/latest/
.. _PyYAML: http://pyyaml.org/wiki/PyYAML
.. _Bokeh: http://bokeh.pydata.org/
.. _dill: https://github.com/uqfoundation/dill
.. _Theano: http://deeplearning.net/software/theano/
.. _six: http://pythonhosted.org/six/
.. _relevant installation instructions: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions
