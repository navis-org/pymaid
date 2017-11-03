Blender interface
*****************

Pymaid comes with an interface to import neurons into `Blender 3D <https://www.blender.org>`_: :mod:`pymaid.b3d`

.. note::
   Blender's Python console does not show all outputs. Please check the terminal
   if you experience issues. In Windows simply go to `Help` >> `Toggle System 
   Console`. In MacOS, right-click Blender in Finder >> `Show Package Contents` 
   >> `MacOS` >> `blender`.

Installation
============

Blender comes with its own Python 3.5, so you need to install PyMaid specifically for this distribution in order to use it within Blender.

There are several ways to install additional packages for Blender's built-in Python. The easiest way is probably this:

1. Download `PIPs <https://pip.pypa.io/en/stable/installing/>`_ get-pip.py and save e.g. in your downloads directory
2. run get-pip.py from Blender Python console:

>>> with open('/Users/YOURNAME/Downloads/get-pip.py') as source_file:
...     exec(source_file.read())

3. Then use pip to install any given package. Here, we install as Scipy an example:

>>> import pip
>>> pip.main(['install','git+git://github.com/schlegelp/pymaid@master'])

Alternatively run Blender's Python from a Terminal. In MacOS do:

1. Make sure Blender is in your Applications folder
2. Right click on Blender icon -> **Show Package Contents**
3. Navigate to **Contents/Resources/2.XX/python/bin** and run **python3.5m** by drag&dropping it into a Terminal
4. Try above steps from the Python shell 

Quickstart
==========

:mod:`pymaid.b3d` provides a simple handler that let's you add, select and manipulate neurons from within the Blender terminal. Try this from within Blender's console:

>>> import pymaid
>>> # The b3d is not automatically loaded when importing the pymaid package
>>> from pymaid import b3d
>>> rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
>>> # Fetch a bunch of neurons
>>> nl = pymaid.get_neuron( 'annotation: glomerulus DA1' )
>>> # Initialise handler
>>> handler = b3d.handler()
>>> # Load neurons into scene
>>> handler.add( nl )
>>> # Colorise neurons
>>> handler.colorize()
>>> # Change thickness of all neurons
>>> handler.neurons.bevel( .02 )
>>> # Select subset
>>> subset = handle.select( nl[:10] )
>>> # Make subset red
>>> subset.color(1,0,0)
>>> # Change color of presynapses to green
>>> handle.presynapses.color(0,1,0)
>>> # Show only connectors
>>> handle.connectors.hide_others()
>>> # Clear all objects
>>> handler.clear()


