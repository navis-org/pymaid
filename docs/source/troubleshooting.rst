Troubleshooting
===============

Installation
------------
*Problem:* PyOctree fails compiling on MacOS. 
*Solution:* I've encountered problems with installing PyOctree on an Anaconda distribution. In that case, try changing the default compiler to GCC.


Plotting
--------

*Problem:* 3D plots using VisPy only use one quarter of the canvas.
*Solution:* Try installing the developer version from GitHub (https://github.com/vispy/vispy). As one-liner: ``git clone https://github.com/vispy/vispy.git && cd vispy && python setup.py install --user``

