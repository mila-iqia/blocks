Pull request workflow
=====================

Blocks development takes place on GitHub_; developers (including project
leads!) add new features by sending `pull requests`_ from their personal
fork (we operate on the so-called `fork & pull`_ model).

.. _GitHub: http://github.com/
.. _pull requests: https://help.github.com/articles/using-pull-requests/
.. _fork & pull: https://help.github.com/articles/using-pull-requests/#fork--pull

This page serves as a "quick reference" for the recommended pull request
workflow. It assumes you are working on a UNIX-like environment with Git
already installed. It is **not** intended to be an exhaustive tutorial
on Git; there are many of those available.

Before you begin
================

Create a GitHub account
~~~~~~~~~~~~~~~~~~~~~~~
If you don't already have one, you should 
`create yourself a GitHub account <https://github.com/join>`_.

Fork the Blocks repository
~~~~~~~~~~~~~~~~~~~~~~~~~~
Once you've set up your account and logged in, you should fork the Blocks
repository to your account by clicking the "Fork" button on the
`official repository's web page <https://github.com/bartvm/blocks>`_.
More information on forking is available in `the GitHub documentation`_.

.. _the GitHub documentation: https://help.github.com/articles/fork-a-repo/

Clone from your fork
~~~~~~~~~~~~~~~~~~~~
In the side bar of your newly created fork of the Blocks repository, you should
see a field that says **HTTPS clone URL** above it. Copy that to your clipboard
and run, at the terminal,

.. bash::
    git clone CLONE_URL

where ``CLONE_URL`` is the URL you copied from your GitHub fork.

If you're doing a lot of development GitHub you should look into setting up
`SSH key authentication <https://help.github.com/categories/ssh/>`_.

Add the official Blocks repository as a remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. bash::
    git remote add upstream https://github.com/bartvm/blocks.git

You only need to do this once.

Beginning a pull request
========================

Verify that origin points to your fork
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. bash::
    git remote -v \|grep origin

Should display two lines. The URLs therein should contain your GitHub username.

Update your upstream remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. bash::
    git fetch upstream

Create a new branch for your pull request based on ``upstream/master``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. bash::
    git checkout -b my_branch_name_for_my_cool_feature upstream/master

Obviously you'll probably want to choose a better branch name.

Make modifications, stage them, and commit them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Repeat until satisfied:

* Make some modifications to the code
* Stage them using `git add` (`git add -p` is particularly useful)
* `git commit` them, alternately `git reset` to undo staging by `git add`.

Push the branch to your fork
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. bash::
    git push -u origin my_branch_name_for_my_cool_feature

Send a pull request
~~~~~~~~~~~~~~~~~~~
This can be done from the GitHub web interface for your fork. See
`this documentation from GitHub`_ for more information.

**Give your pull request an appropriate title** which makes it obvious what
the content is. If it is intended to resolve a specific issue, put "Fixes
#*NNN*." in the pull request description field, where *NNN* is the issue
number.

.. _this documentation from GitHub: https://help.github.com/articles/using-pull-requests/#initiating-the-pull-request
