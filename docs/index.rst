pymaid: a Python-CATMAID interface
==================================

.. raw:: html

   <div class="container-fluid">
      <div class="row">
         <div class="col-lg-6">

``pymaid`` (short for "Python-CATMAID") is a Python library for fetching,
analyzing and visualizing data generated with
`CATMAID <http://catmaid.readthedocs.io/en/stable/>`_.

``pymaid`` is built on top of `navis <https://navis.readthedocs.io/en/latest/>`_
and is fully compatible with its functions.

For a brief introduction to the library, you can read the
:ref:`tutorial <tutorial>`. Visit the
:ref:`installation page <installing>` page to see how to download the
package. You can browse the :ref:`example gallery <example_gallery>` and
:ref:`API reference <api>` to learn what you can do with pymaid.

Pymaid is licensed under the GNU GPL v3+ license. The source code is hosted
at `Github <https://github.com/navis-org/pymaid>`_. Feedback, feature requests
and bug reports are very welcome and best placed in
`issues <https://github.com/navis-org/pymaid/issues>`_.
If you have any questions, please don't hesitate: pms70@cam.ac.uk

The package is stable but I recommend watching its
`Github repository <https://github.com/navis-org/pymaid>`_ for updates.
Make sure that your ``pymaid.__version__`` is up-to-date and check out the
:ref:`release notes <whats_new>`.

.. raw:: html

         </div>
         <div class="col-lg-3">
            <div class="panel panel-default">
               <div class="panel-heading">
                  <h3 class="panel-title">Contents</h3>
               </div>
               <div class="panel-body">

.. toctree::
   :maxdepth: 1

   source/whats_new
   source/install
   source/intro.ipynb
   examples/examples_index
   source/troubleshooting
   source/api


.. raw:: html

               </div>
            </div>
         </div>
         <div class="col-lg-3">
            <div class="panel panel-default">
               <div class="panel-heading">
                  <h3 class="panel-title">Features</h3>
               </div>
               <div class="panel-body">

* fetch data directly from CATMAID server
* write data (e.g. annotations, tags, neurons) to the server
* fully compatible with `navis <https://navis.readthedocs.io/en/latest/>`_
* high-level functions to analyze e.g. connectivity

.. raw:: html

               </div>
            </div>
         </div>
      </div>
   </div>
