Plotting
********

Pymaid contains functions for 2 and 3D plotting of neurons, synapses and networks. These functions are all part of the :mod:`pymaid.plot` module and represent wrappers for `matplotlib <http://www.matplotlib.org>`_ for 2D, `vispy <http://www.vispy.org>`_ and `plotly <http://plot.ly>`_ for 3D.

Plotting Neurons
================

Neuron classes ( :class:`pymaid.core.CatmaidNeuron` and :class:`pymaid.core.CatmaidNeuronList`) as well as nblast results (:class:`pymaid.rmaid.nbl_results`) have built-in modules that call :func:`pymaid.plot.plot3d` or :func:`pymaid.plot.plot2d`.

2D Plotting:
------------

>>> from pymaid import core, pymaid
>>> import matplib.pyplot as plt
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org', 
...                              'HTTP_USER' , 
...                              'HTTP_PASSWORD', 
...                              'TOKEN' )
>>> nl = core.CatmaidNeuronList([123456, 567890], remote_instance = rm)
>>> #Plot using standard parameters
>>> fig, ax = nl.plot2d()
2017-07-25 14:56:08,701 - pymaid.plot - INFO - Done. Use matplotlib.pyplot.show() to show plot.
>>> plt.show()

Adding volumes:
+++++++++++++++

:func:`pymaid.plot.plot2d` has some built-in outlines for the **adult Drosophila** brain project: ``brain``, ``MB``, ``LH``, ``AL``, ``SLP``, ``SIP``, ``CRE``

>>> fig, ax = nl.plot2d( brain = (.8,.8,.8), 
...                      MB = (.3,.9,.3) )
>>> plt.show()

3D Plotting:
------------

>>> from pymaid import core, pymaid
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org', 
...                              'HTTP_USER' , 
...                              'HTTP_PASSWORD', 
...                              'TOKEN' )
>>> nl = core.CatmaidNeuronList([123456, 567890], remote_instance = rm)
>>> #Plot using standard parameters
>>> nl.plot3d()

By default, calling :func:`pymaid.plot.plot3d` uses the vispy backend and does not plot connectors. By passing **kwargs, we can change that behavior:

>>> fig = nl.plot3d( backend = 'plotly', connectors = True )
2017-07-18 21:22:27,192 - pymaid.plot - INFO - Generating traces...
2017-07-18 21:22:45,504 - pymaid.plot - INFO - Traced done.
2017-07-18 21:22:45,505 - pymaid.plot - INFO - Done. Plotted 4000 nodes and 320 connectors
2017-07-18 21:22:45,505 - pymaid.plot - INFO - Use plotly.offline.plot(fig, filename="3d_plot.html") to plot. Optimised for Google Chrome.
>>> #Fig is a dictionary that plotly turns into a WebGL file
>>> from plotly import offline as poff
>>> poff.plot( fig )

.. note::
   Vispy itself uses either one of these backends: 
   Qt, GLFW,SDL2, Wx, or Pyglet. By default, pymaid
   installs and sets PyQt5 as vispy's backend. If
   you need to change that use e.g. ``vispy.use(app='PyQt4')``

Adding volumes:
+++++++++++++++

:func:`pymaid.plot.plot3d` allows plotting of volumes (e.g. neuropil meshes). It's very straight forward to use meshes directly from you Catmaid Server:

>>> from pymaid import plot, pymaid
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org', 
...                              'HTTP_USER' , 
...                              'HTTP_PASSWORD', 
...                              'TOKEN' )
>>> nl = core.CatmaidNeuronList([123456, 567890], remote_instance = rm)
>>> #Plot volumes without specifying color
>>> nl.plot3d( volumes = ['v13.LH_R', 'v13_LH_L'] )
>>> #Provide colors
>>> nl.plot3d( volumes = {'v13.LH_R':(255,0,0), 'v13_LH_L':(0,255,0)} )

You can also pass your own custom volumes as dictionarys:

>>> cust_vol = dict( my_volumes = dict (
...            verts = [ (1,2,1),(5,6,7),(8,6,4) ],
...            faces = [ (0,1,2) ],
...            color = (255,0,0)
...            ) )
>>> nl.plot3d( volumes = cust_vol )

Plotting Networks
=================

:func:`pymaid.plot.plot_network` is a wrapper to plot networks using plotly. It's rather slow for large-ish graphs though

>>> from pymaid import plot, pymaid, core
>>> import plotly.offline as poff
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org', 
...                              'HTTP_USER' , 
...                              'HTTP_PASSWORD', 
...                              'TOKEN' )
>>> pymaid.remote_instance = rm
>>> pns = pymaid.get_skids_by_annotation('PN right')
>>> partners = pymaid.get_partners( pns )
>>> all_skeleton_ids = pns + partners.skeleton_id.tolist()
>>> fig = plot.plot_network( all_skeleton_ids, remote_instance = rm )
>>> poff.plot(fig)
