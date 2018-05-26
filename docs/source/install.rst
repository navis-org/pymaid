Install
=======

PyMaid requires Python 3.4 or higher. It heavily depends on other
scientific packages (e.g. `scipy` and `numpy`). If you do not already
have a Python environment configured on your computer, please see the
instructions for installing the full `scientific Python stack
<https://scipy.org/install.html>`_.

.. note::
   If you are on Windows I strongly recommend installing a scientific Python
   distribution that comes with many of the key dependencies preinstalled:
   `Anaconda <https://www.continuum.io/downloads>`_,
   `Enthought Canopy <https://www.enthought.com/products/canopy/>`_,
   `Python(x,y) <http://python-xy.github.io/>`_,
   `WinPython <https://winpython.github.io/>`_, or
   `Pyzo <http://www.pyzo.org/>`_.
   If you already use one of these Python distribution, please refer to their
   online documentation on how to install additional packages.

Quick install
-------------

PyMaid is **NOT** listed in the Python Packaging Index (PyPI). There is a
`pymaid` package on PyPI but that is something else! Hence, you will have to
install from `Github <https://github.com/schlegelp/PyMaid>`_. To get the
most recent version please use:

::
   pip install git+git://github.com/schlegelp/pymaid@master

See `here <https://pip.pypa.io/en/stable/installing/>`_ how to get PIP.

Depending on your default Python version you may have to specify that you want
PyMaid to be installed for Python 3:

::
   pip3 install git+git://github.com/schlegelp/pymaid@master

.. important::
   One of the dependencies ``pyoctree`` requires ``numpy`` to be installed. If
   pip fails with ``ImportError: No module named 'numpy'`` you have to manually
   install numpy first by running ``pip install numpy``. Then retry installing
   PyMaid via ``pip``.

.. note::
   The :mod:`pymaid.rmaid` module requires `rpy2 <https://rpy2.readthedocs.io>`_.
   As ``rpy2`` installation fails if no R is installed, it is not a default
   dependency. If you want to use the interface between R's nat, catnat and elmr
   you have have to install ``rpy2`` manually *after* R has been set up.

Installing from source
----------------------

1. Download the source (tar.gz file) from
 https://github.com/schlegelp/PyMaid/tree/master/dist

2. Unpack and change directory to the source directory
 (the one with setup.py).

3. Run :samp:`python setup.py install` to build and install

Requirements
------------

PyMaid relies on scientific Python packages to do its job.
On Linux and MacOS these packages will be installed automatically
when you install pymaid but on Windows you may have to tinker around
to get them to work. Your best bet is to use a scientific Python
distribution such as `Anaconda <https://www.continuum.io/downloads>`_
which should come with "batteries included".

`NumPy <http://www.numpy.org/>`_
  Provides matrix representation of graphs and is used in some graph
  algorithms for high-performance matrix computations.

`Pandas <http://pandas.pydata.org/>`_
  Provides advanced dataframes and indexing.

`Vispy <http://vispy.org/>`_
  Used to visualise neurons in 3D. This requires you to have *one* of
  the supported `backends <http://vispy.org/installation.html#backend-requirements>`_
  installed. During automatic installation PyMaid will try installing the
  PyQt5 backend to fullfil this requirement.

`Plotly <https://plot.ly/python/getting-started/>`_
  Used to visualise neurons in 3D. Alternative to Vispy based on WebGL.

`NetworkX <https://networkx.github.io>`_
  Graph analysis tool written in pure Python. This is the standard library
  used by PyMaid.

`SciPy <http://scipy.org>`_
  Provides sparse matrix representation of graphs and many scientific
  computing tools.

`Matplotlib < http://matplotlib.sourceforge.net/>`_
  Essential for all 2D plotting.

`Seaborn <https://seaborn.pydata.org>`_
  Provides additional plotting.

`tqdm <https://pypi.python.org/pypi/tqdm>`_
  Neat progress bars.

`PyPNG <https://pythonhosted.org/pypng/>`_
  Generates PNG images. Used for taking screenshot from 3D viewer. Install
  from PIP: ``pip install pypng``.

`PyOctree <https://pypi.python.org/pypi/pyoctree/>`_ (optional)
  Generates octrees from meshes to compute ray casting. Used to check if
  objects are within volume.

`Rpy2 <https://rpy2.readthedocs.io/en/version_2.8.x/overview.html#installation>`_ (optional)
  Provides interface with R. This allows you to use e.g. R packages from
  https://github.com/jefferis and https://github.com/alexanderbates. Note that
  this package is not installed automatically as it would fail if R is not
  already installed on the system. You have to install Rpy2 manually!

`Shapely <https://shapely.readthedocs.io/en/latest/>`_ (optional)
  This is used to get 2D outlines of CATMAID volumes.


Speed: iGraph vs NetworkX
-------------------------

By default PyMaid uses the `NetworkX <https://networkx.github.io>`_ graph
library for most of the computationally expensive function. NetworkX is
written in pure Python, well maintained and easy to install.

If you need that extra bit of speed, consider manually installing
`iGraph <http://igraph.org/>`_. It is written in C and therefore very fast. If
available, PyMaid will try using iGraph over NetworkX. iGraph is difficult to
install though because you have to install the C core first and then its
Python bindings, ``python-igraph``.

