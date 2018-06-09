Install
=======

Installation instructions come in two flavors:

1. **Quick install**: if you know what you are doing.
2. **Installation 101** : step-by-step instructions.

Quick install
-------------

If you don't already have it, get `PIP <https://pip.pypa.io/en/stable/installing/>`_.

PyMaid is **NOT** listed in the Python Packaging Index (PyPI). There is a
`pymaid` package on PyPI but that is something else! Hence, you will have to
install from `Github <https://github.com/schlegelp/PyMaid>`_. To get the
most recent version use:

::

   pip install git+git://github.com/schlegelp/pymaid@master


There are two (optional) dependencies that you might want to install manually:
pyoctree and rpy2 (see Requirements below). The latter is only relevant if
you intend to use pymaid's R bindings.


Installing from source
**********************
Instead of using PIP to install from Github, you can also install manually:

1. Download the source (tar.gz file) from
   https://github.com/schlegelp/PyMaid/tree/master/dist

2. Unpack and change directory to the source directory
   (the one with ``setup.py``).

3. Run ``python setup.py install`` to build and install


Installation 101
----------------

1. Check if Python 3 is installed and install if necessary. Linux and Mac 
   should come with Python distribution(s) but you need to figure out if 
   you have Python 2, Python 3 or both. Open a terminal, type in ``python`` 
   and press enter:

    a) You get something along the lines of ``command not found``, see below
       note on installing Python 3.
    b) A Python console opens but it says e.g. ``Python 2.7.14``. Exit the Python 
       console (``exit()``) and try ``python3``. If this works, your default
       distribution is Python 2.x. That's fine but you have to be careful to
       specify which Python you want packages to be installed for. Concretely,
       you need to replace ``pip install ...`` with ``pip3 install ...``.
       If ``python3`` throws ``command not found``, see below note on installing 
       Python 3.
    c) A Python 3 console opens. Proceed with step 2.

2. Get the Python package manager `PIP <https://pip.pypa.io/en/stable/installing/>`_. 
   Try ``pip`` in a terminal and if that throws an error, go to above URL to
   download and install pip. In a nutshell:
    a) Open above URL
    b) Download the ``get-pip.py`` to your Downloads folder by right-clicking
       it and selecting `Save File As`.
    c) Open a terminal, navigate to your Downloads folder (e.g.
       ``cd Downloads``) and run ``python get-pip.py``.

3. Install pymaid and its dependencies. Open a terminal and run 
   ``pip install git+git://github.com/schlegelp/pymaid@master`` to install
   the most recent version from Github. Remember to use ``pip3`` instead if
   your default distribution is Python 2. This *should* take care of all
   required dependencies. If anything fails, find the culprit in below 
   Requirements and install the dependency manually before attempting to
   install pymaid again.

4. Go to *Introduction* and try some of the basic tutorials.

.. note::
   There are two (optional) dependencies that you might want to install manually:
   pyoctree and rpy2 (see Requirements below). The latter is only relevant if
   you intend to use pymaid's R bindings.

.. note::
   **Installing Python 3**: 

   On **Linux** and **OSX (Mac)**, simply visit e.g. https://www.python.org and 
   download + install Python 3.4 or later.
   
   On **Windows**, things are bit more tricky. While pymaid is written in pure
   Python, some of its dependencies are written in C for speed and need to be 
   compiled - which a pain on Windows. I strongly recommend installing a 
   scientific Python distribution that comes with "batteries included". 
   `Anaconda <https://www.continuum.io/downloads>`_ is a widespread solution
   that comes with its own package manager ``conda``.


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
  `PyQt5 <http://pyqt.sourceforge.net/Docs/PyQt5/installation.html>`_ backend 
  to fullfil this requirement.

`Plotly <https://plot.ly/python/getting-started/>`_
  Used to visualise neurons in 3D. Alternative to Vispy based on WebGL.

`NetworkX <https://networkx.github.io>`_
  Graph analysis tool written in pure Python. This is the standard library
  used by PyMaid.

`SciPy <http://scipy.org>`_
  Provides sparse matrix representation of graphs and many scientific
  computing tools.

`Matplotlib <http://matplotlib.sourceforge.net/>`_
  Essential for all 2D plotting.

`Seaborn <https://seaborn.pydata.org>`_
  Used e.g. for its color palettes.

`tqdm <https://pypi.python.org/pypi/tqdm>`_
  Neat progress bars.

`PyPNG <https://pythonhosted.org/pypng/>`_
  Generates PNG images. Used for taking screenshot from 3D viewer. Install
  from PIP: ``pip install pypng``.

`PyOctree <https://pypi.python.org/pypi/pyoctree/>`_ (optional)
  Provides octrees from meshes to perform ray casting. Used to check if
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

