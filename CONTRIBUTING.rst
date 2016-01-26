Contributing
============

How to contribute
-----------------

Users can contribute to this project by using it and giving feedback,
reporting bugs, or suggesting changes. Users with development experience
users who would like to start learning how to develop software can are
welcome to contribute code.

Contributing feedback, bug reports, etc.
----------------------------------------

The best way of giving feedback is the use of the `issue tracker on
Github <https://github.com/ceholden/yatsm/issues>`__ as this tracker is
open and publicly visible, is relatively well formed, and integrates
well as all of the development is done on Github. Emailing me is not an
efficient way of making suggestions or reporting bugs because it limits
the number of people who may engage in or benefit by the conversation.

Bug reports are not helpful if there is not sufficient information to
reproduce the bug. Sufficient information may include, but is not
limited to:

-  The exact command or API reference you entered to produce the problem
-  The full log, including the entire stack traceback, of the command or
   API reference that is causing the problem
-  The configuration file used for your dataset and run parameters
-  The version of YATSM

   -  ``python -c 'import yatsm; print(yatsm.__version__)'``
   -  If you are using a development build (i.e., if you cloned
      ``master`` and did not check out a tag), please include the SHA-1
      hash of the last commit

-  The version of dependencies used by YATSM

   -  ``pip freeze``

Please also include a description of what you were trying to accomplish,
what you were expecting to happen, and how what did happen differed from
your expectation. These directions are probably excessively prescriptive
so keep these in mind but use your best judgment.

Feedback in the form of a feature suggestion may be less formally
stated, but example implementations are always welcome in any form to
help further your explanation.

Contributing code
-----------------

The best way of contributing code is to fork this project, make the
changes you wish to include, and then submit a "pull request" (PR) via
Github's PR interface.

Conventions
~~~~~~~~~~~

The YATSM project follows a few conventions when writing code:

1. Write your code to be compatible with Python 2.7 and 3.5+. The
   ``six`` module is a dependency for this project, so use it if you
   need to.
2. All public functions, classes, methods, etc. must have useful and
   informative docstrings. This project uses `the Google format for
   docstrings <https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`__
   as they are, in our opinion, the most readable when inspecting in
   IPython and are now parseable by Sphinx.
3. All code contributions must adhere to the `Python style guide,
   PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ as much as
   possible. Exceptions may be made for line lengths above 79
   characters, but there is likely no reason to go beyond 100
   characters.

   -  This project generally adheres to the `Google Python style
      guide <https://google.github.io/styleguide/pyguide.html>`__ as way
      of providing another example.

4. All new code contributions should come with a set of corresponding
   tests that exercise the code addition.
5. All code contributions need to pass both the existing and added
   tests. See the ``.travis.yaml`` file for how automated testing on
   `Travis CI <https://travis-ci.org/ceholden/yatsm>`__ is performed.

Example
~~~~~~~

1. For this project

   -  Click on the "Fork" button near the top of this page. This button
      creates a copy of the YATSM project under your own account that
      you have write access to

2. Clone this copy to your machine:

   ::

       $ git clone git@github.com:YOUR_USERNAME/yatsm.git
       $ cd yatsm/

3. Create a branch for your bug fix or feature addition. As a
   convention, it is useful to name this branch as a reference to the
   issue that describes the bug or to the feature that you wish to add.
   For example, one might name the branch ``issue34``,
   ``feat/io_abstract``, etc:

   ::

       $ git checkout -b feat/my_new_feature

   Make your changes on this branch.

4. Continue your work while tracking significant, notable chunks of
   changes as individual commits using Git:

   ::

       $ git add -p your_modified_files
       $ git commit -m "This is a description of the changes made"

   Push your code changes to your forked copy of YATSM:

   ::

       $ git push -u origin feat/my_new_feature

5. When you are done, submit a "pull request" (PR) using Github's
   interface to send your contributions to YATSM's maintainers for
   review.

Code of conduct
===============

Try not to be an ass and cite scientific work when making contributions
that relate to a published algorithm.
