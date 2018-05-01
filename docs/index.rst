PyMaid
======

:Release: |version|
:Date: |today|

PyMaid is a Python 3 package for working with CATMAID data.
It allows you to fetch, analyse and plot data from a CATMAID server.
The package is under heavy development, I strongly recommend
watching its `Github repository <https://github.com/schlegelp/PyMaid>`_.
Also, make sure that your ``pymaid.__version__`` is up-to-date (see
**Release** above).

Features
--------

* wrappers for CATMAID's API to fetch data
* custom neuron objects that perform on-demand data fetching
* 2D (matplotlib) and 3D (vispy or plotly) plotting of neurons
* virtual neuron surgery (cutting, stitching, pruning, rerooting)
* R bindings (e.g. for libraries nat, nat.nblast and elmr)
* interface with Blender 3D
* import/export from/to SWC
* and oh so much more...

Contribute
----------

Source Code: https://github.com/schlegelp/PyMaid

Issue Tracker: https://github.com/schlegelp/PyMaid/issues

Support
-------

If you have any questions, please don't hesitate: pms70@cam.ac.uk

License
---------
PyMaid is licensed under the GNU GPL v3+ license

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   source/install
   source/intro.ipynb
   source/neurons
   source/fetching_data
   source/plotting.ipynb
   source/morph_analysis.ipynb
   source/connectivity_analysis
   source/blender
   source/rmaid_doc
   source/user_stats
   source/api
   source/troubleshooting
   examples/examples_index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

