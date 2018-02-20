Plotting
********

Pymaid contains functions for 2D and 3D plotting of neurons, synapses and networks. These functions represent wrappers for `matplotlib <http://www.matplotlib.org>`_ for 2D, `vispy <http://www.vispy.org>`_ and `plotly <http://plot.ly>`_ for 3D.

.. note::
   If you are experiencing issues when using vispy
   as backend, you should try installing the dev
   version (currently 0.6.0dev0) directly from
   `Github <https://github.com/vispy/vispy>`_.
   The version installed from PIP is 0.5.2.

Plotting Neurons
================

Neuron objects, :class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList`, as well as nblast results, :class:`~pymaid.rmaid.nbl_results`, have built-in methods that call :func:`~pymaid.plot3d` or :func:`~pymaid.plot2d`.

2D Plotting
-----------
This uses matplotlib to generate 2D plots. The big advantage is that you can save these plots as vector graphics. Unfortunately, matplotlib's capabilities regarding 3D data are limited. The main problem is that depth (z) is at most simulated by trying to layer objects according to their z-order rather than doing proper rendering. You have several options to deal with this: see `method` parameter in :func:`pymaid.plot2d`. It is important to be aware of this issue as e.g. neuron A might be plotted in front of neuron B even though it is actually spatially behind it. The more busy your plot and the more neurons intertwine, the more likely this is to happen.

>>> import pymaid
>>> import matplib.pyplot as plt
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org',
...                              'HTTP_USER' ,
...                              'HTTP_PASSWORD',
...                              'TOKEN' )
>>> nl = pymaid.CatmaidNeuronList([123456, 567890])
>>> # Plot using standard parameters
>>> fig, ax = nl.plot2d()
>>> plt.show()
>>> # Plot using matplotlib's 3D functions
>>> fig, ax = nl.plot2d( method='3d_complex' )
>>> # Change from default frontal view to lateral view
>>> ax.azim = 0
>>> plt.show()
>>> # Render 3D rotation
>>> for i in range(0,360,10):
>>>   ax.azim = i
>>>   plt.savefig('frame_{0}.png'.format(i), dpi=200)

Plotting volumes
+++++++++++++++

>>> # Retrieve volume
>>> lh = pymaid.get_volume('LH_R')
>>> # Set color and alpha
>>> lh['color'] = (1,0,0,.5)
>>> fig, ax = nl.plot2d([ 123456,56789,lh ] )
>>> plt.show()

3D Plotting
-----------
For 3D plots, we are using either Vispy or Plotly to render neurons and volumes. The default backend is Vispy.

>>> import pymaid
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org',
...                              'HTTP_USER' ,
...                              'HTTP_PASSWORD',
...                              'TOKEN' )
>>> nl = pymaid.CatmaidNeuronList([123456, 567890], remote_instance = rm)
>>> # Plot using standard parameters
>>> nl.plot3d()
>>> # Save screenshot
>>> pymaid.screenshot('screenshot.png', alpha = True)

The canvas persistent and survives simply closing the window. Calling :func:`~pymaid.plot3d` again will add objects to the canvas and open it again.

>>> # Add another set of neurons
>>> nl2 = pymaid.CatmaidNeuronList([987675,543210], remote_instance = rm)
>>> nl2.plot3d()
>>> # To clear canvas either pass parameter when plotting...
>>> nl2.plot3d(clear3d=True)
>>> # ... or call function to clear
>>> pymaid.clear3d()
>>> # To wipe canvas from memory
>>> pymaid.close3d()

By default, calling :func:`~pymaid.plot3d` uses the vispy backend and does not plot connectors. By passing **kwargs, we can change that behavior:

>>> fig = nl.plot3d( backend = 'plotly', connectors = True )
2017-07-18 21:22:27,192 - pymaid - INFO - Generating traces...
2017-07-18 21:22:45,504 - pymaid - INFO - Traced done.
2017-07-18 21:22:45,505 - pymaid - INFO - Done. Plotted 4000 nodes and 320 connectors
2017-07-18 21:22:45,505 - pymaid - INFO - Use plotly.offline.plot(fig, filename="3d_plot.html") to plot. Optimised for Google Chrome.
>>> # Fig is a dictionary that plotly turns into a WebGL file
>>> from plotly import offline as poff
>>> poff.plot( fig )

.. note::
   Vispy itself uses either one of these backends:
   Qt, GLFW,SDL2, Wx, or Pyglet. By default, pymaid
   installs and sets PyQt5 as vispy's backend. If
   you need to change that use e.g. ``vispy.use(app='PyQt4')``

Navigating the 3D viewer
++++++++++++++++++++++++

1. Rotating: Hold left mousebutton
2. Zooming: Use the mousewheel or left+right-click and drag
3. Panning: Hold left mousebutton + shift
4. Perspective: Hold left and right mousbutton + shift

Adding volumes
++++++++++++++

:func:`~pymaid.plot3d` allows plotting of volumes (e.g. neuropil meshes). It's very straight forward to use meshes directly from you Catmaid Server:
there is a custom class for Catmaid Volumes, :class:`pymaid.Volume` which has some neat methods - check out its reference.

>>> import pymaid
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org',
...                              'HTTP_USER' ,
...                              'HTTP_PASSWORD',
...                              'TOKEN' )
>>> nl = pymaid.CatmaidNeuronList([123456, 567890], remote_instance = rm)
>>> # Plot volumes without specifying color
>>> nl.plot3d( ['v13.LH_R', 'v13_LH_L'] )
>>> # Provide colors
>>> vols = [ pymaid.get_volume('v13.LH_R', color=(255,0,0,.5)),
...  		 pymaid.get_volume('v13.LH_L', color=(0,255,0,.5)) ]
>>> nl.plot3d( vols )

You can also pass your own custom volumes as dictionarys:

>>> cust_vol = pymaid.volume( my_volumes = dict (
...            				vertices = [ (1,2,1),(5,6,7),(8,6,4) ],
...           				faces = [ (0,1,2) ],
...							name = 'custom volume',
...           				color = (255,0,0)
...            ) )
>>> nl.plot3d( cust_vol )

Plotting Networks
=================

:func:`~pymaid.plot_network` is a wrapper to plot networks using plotly. It's rather slow for large-ish graphs though

>>> import pymaid
>>> import plotly.offline as poff
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org',
...                              'HTTP_USER' ,
...                              'HTTP_PASSWORD',
...                              'TOKEN' )
>>> pns = pymaid.get_skids_by_annotation('PN right')
>>> partners = pymaid.get_partners( pns )
>>> all_skeleton_ids = pns + partners.skeleton_id.tolist()
>>> fig = pymaid.plot_network( all_skeleton_ids, remote_instance = rm )
>>> poff.plot(fig)

Reference
=========

.. autosummary::
    :toctree: generated/

    ~pymaid.plot3d
    ~pymaid.plot2d
    ~pymaid.plot1d
    ~pymaid.plot_network
    ~pymaid.clear3d
    ~pymaid.close3d
    ~pymaid.get_canvas
    ~pymaid.screenshot
    ~pymaid.Volume
