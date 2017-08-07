Install
=======

PyMaid requires Python 3.3 or higher. It heavily depends on other
scientific packages (e.g. `scipy`). If you do not already
have a Python environment configured on your computer, please see the
instructions for installing the full `scientific Python stack
<https://scipy.org/install.html>`_. 

.. note::
   If you are on Windows and want to install optional packages (e.g., `scipy`),
   then you will need to install a Python distribution such as
   `Anaconda <https://www.continuum.io/downloads>`_,
   `Enthought Canopy <https://www.enthought.com/products/canopy/>`_,
   `Python(x,y) <http://python-xy.github.io/>`_,
   `WinPython <https://winpython.github.io/>`_, or
   `Pyzo <http://www.pyzo.org/>`_.
   If you use one of these Python distribution, please refer to their online
   documentation.

Quick install
-------------

PyMaid is *not yet* listed in the Python Packaging Index but you can install
the current version from `Github <https://github.com/schlegelp/PyMaid>`_ using:

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
   PyMaid via pip.

.. note::
   The :mod:`pymaid.rmaid` module requires `rpy2 <https://rpy2.readthedocs.io>`_.
   As ``rpy2`` installation fails if no R is installed, it is not a default 
   dependency and has to be installed manually *after* R has been set up.

Installing from source
----------------------

1. Download the source (tar.gz file) from
 https://github.com/schlegelp/PyMaid/tree/master/dist

2. Unpack and change directory to the source directory
 (it should have the files README.txt and setup.py).

3. Run :samp:`python setup.py install` to build and install

Requirements 
------------

PyMaid heavily relies on scientific Python packages to do its job. 
On Linux and MacOS these packages will be installed automatically
but on Windows you may have to tinker around to get them to work.
Your best bet is to use a scientific Python distribution such
as `Anaconda <https://www.continuum.io/downloads>`_ which has
most of these preinstalled. 


NumPy
*****
Provides matrix representation of graphs and is used in some graph algorithms for high-performance matrix computations.

  - Download: http://scipy.org/Download

Pandas
******
Provides advanced dataframes and indexing.

	- Download: http://pandas.pydata.org/getpandas.html

Vispy
*****
Used to visualise neurons in 3D. This requires you to have *one* of 
the supported `backends <http://vispy.org/installation.html#backend-requirements>`_ 
installed. During automatic installation PyMaid will try installing the PyQt5 
backend to fullfil this requirement.

  - Download: http://vispy.org/installation.html

Plotly
******
Used to visualise neurons in 2D. Alternative to Vispy based on WebGL.

  - Download: https://plot.ly/python/getting-started/

iGraph
******
Provides fast analysis of graphs (networks and neurons)

  - Download: http://igraph.org/python/#downloads

SciPy
*****
Provides sparse matrix representation of graphs and many numerical scientific tools.

  - Download: http://scipy.org/Download

Matplotlib
**********
Provides flexible drawing of graphs.

  - Download: http://matplotlib.sourceforge.net/

tqdm
****
Neat progress bar

  - Download: https://pypi.python.org/pypi/tqdm

PyOctree
********
Generates octrees from meshes to compute ray casting. Used to check if objects are within volume.

  - PyPi: https://pypi.python.org/pypi/pyoctree/
  - Github: https://github.com/mhogg/pyoctree

Rpy2
****
Provides interface with R. This allows you to use e.g. R packages from https://github.com/jefferis and https://github.com/alexanderbates. Note that this package is not installed automatically as it would fail if R is not already installed on the system. You have to install Rpy2 manually!

  - Download: https://rpy2.readthedocs.io/en/version_2.8.x/overview.html#installation

