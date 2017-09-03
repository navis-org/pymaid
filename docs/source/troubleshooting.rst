Troubleshooting
===============

Installation
------------

Problem:
++++++++
PyOctree fails compiling because of 'fopenmp' 

Solution: 
+++++++++
1. Download and extract the PyOctree Github `repository <https://github.com/mhogg/pyoctree>`_. 
2. Open ``setup.py`` and set ``BUILD_ARGS['mingw32'] = [ ]`` and ``LINK_ARGS['unix'] = [ ]``
3. Open a terminal, navigate to the directory containing ``setup.py`` and run ``python setup.py install`` (if your default Python is 2.7, use ``python3``)


Plotting
--------

Problem:
++++++++
3D plots using VisPy only use one quarter of the canvas.

Solution:
++++++++++
Try installing the developer version from GitHub (https://github.com/vispy/vispy). As one-liner: ``git clone https://github.com/vispy/vispy.git && cd vispy && python setup.py install --user``

