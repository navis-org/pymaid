Troubleshooting
===============

Installation
------------

Problem:
++++++++
PyOctree fails compiling because of ``fopenmp``.

Solution: 
+++++++++
1. Download and extract the PyOctree Github `repository <https://github.com/mhogg/pyoctree>`_. 
2. Open ``setup.py`` and set ``BUILD_ARGS['mingw32'] = [ ]`` and ``LINK_ARGS['unix'] = [ ]``
3. Open a terminal, navigate to the directory containing ``setup.py`` and run ``python setup.py install`` (if your default Python is 2.7, use ``python3``)


Plotting
--------

Problem:
++++++++
3D plots with VisPy as backend use only one quarter of the canvas.

Solution:
++++++++++
Try installing the developer version from GitHub (https://github.com/vispy/vispy). As one-liner::

    git clone https://github.com/vispy/vispy.git && cd vispy && python setup.py install --user

Problem:
++++++++
3D plots using Plotly are too small and all I can see is a chunk of legend.

Solution:
++++++++++
Sometimes plotly does not scale the plot correctly. The solution is to play
around with the ``width`` parameter::

    fig = pymaid.plot3d(neurons, backend='plotly', width=1200)


Jupyter
-------

Problem:
++++++++
Instead of a progress bar, I get some odd message (e.g. ``Hbox(children=...``)
when using pymaid in a Jupyter notebook.

Solution:
+++++++++
You probably have `ipywidgets <ipywidgets.readthedocs.io>`_ not installed or
not configured properly. One work-around is to force pymaid to use standard
progress bars using :func:`pymaid.set_pbars`::
        
    pymaid.set_pbars(jupyter=False)
