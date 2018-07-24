pymaid: neuron analysis toolbox
===============================

.. raw:: html

   <div class="container-fluid">
      <div class="row">
         <div class="col-md-6">

Pymaid is a Python library for **visualisation** and **analysis** of **neuron data**
generated with `CATMAID <http://catmaid.readthedocs.io/en/stable/>`_. It allows you to
fetch, analyse and plot neuron morpholgy and connectivity from a CATMAID server.

The package is stable but I recommend watching its
`Github repository <https://github.com/schlegelp/PyMaid>`_ for updates.
Make sure that your ``pymaid.__version__`` is up-to-date.

For a brief introduction to the library, you can read the
:ref:`tutorial <tutorial>`. Visit the
:ref:`installation page <installing>` page to see how to download the
package. You can browse the :ref:`example gallery <example_gallery>` and
:ref:`API reference <api>`  to see what you can do with pymaid.

Pymaid is licensed under the GNU GPL v3+ license. The Source code is hosted
at `Github <https://github.com/schlegelp/PyMaid>`_. This is also where
`issues <https://github.com/schlegelp/PyMaid/issues>`_ are being tracked.
If you have any questions, please don't hesitate: pms70@cam.ac.uk


.. raw:: html

         </div>
         <div class="col-md-3">
            <div class="panel panel-default">
               <div class="panel-heading">
                  <h3 class="panel-title">Contents</h3>
               </div>
               <div class="panel-body">

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
   source/tiles.ipynb
   source/user_stats
   source/api
   source/troubleshooting
   examples/examples_index

.. raw:: html

               </div>
            </div>
         </div>
         <div class="col-md-3">
            <div class="panel panel-default">
               <div class="panel-heading">
                  <h3 class="panel-title">Features</h3>
               </div>
               <div class="panel-body">

* fetch data from CATMAID server
* 2D (matplotlib) and 3D (vispy or plotly) plotting of neurons
* virtual neuron surgery (cutting, stitching, pruning, rerooting)
* R bindings (e.g. for libraries nat, nat.nblast and elmr)
* interface with Blender 3D
* import/export from/to SWC
* process EM image data
* and oh so much more...

.. raw:: html

               </div>
            </div>
         </div>
      </div>
   </div>

