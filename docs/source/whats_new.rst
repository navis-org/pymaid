.. _whats_new:

What's new?
===========

.. list-table::
   :widths: 7 7 86
   :header-rows: 1

   * - Version
     - Date
     -
   * - 0.88
     - 29/07/18
     - - data caching for faster queries, see :doc:`caching demo <data_caching>`
       - new function: :func:`~pymaid.smooth_neuron`
       - :func:`~pymaid.resample_neuron` now resamples radius too
       - :func:`~pymaid.guess_radius` interpolation now takes distance along spines into account
       - :func:`~pymaid.despike_neuron` is now able to catch spikes that consist of multiple nodes
       - :func:`~pymaid.cable_length` is now deprecated
       - general improvements to docstrings
   * - 0.87
     - 20/07/18
     - - :func:`~pymaid.get_team_contributions` now takes link creation into account
       - :func:`~pymaid.get_time_invested` should be way faster now
       - :func:`~pymaid.geodesic_matrix` now returns a SparseDataFrame to save memory
       - addded :func:`pymaid.CatmaidNeuron.to_dataframe` method
       - general improvements and docstrings
   * - 0.86
     - 16/07/18
     - - arithmetric operations with CatmaidNeuron/Lists will now warn if skeleton IDs match but neuron objects are not identical. See :doc:`here <neuronlist_math>` for explanation.
       - fixed a bug when using regex to query for neurons that led to duplicate skeleton IDs being returned
   * - 0.85
     - 13/07/18
     - - fixed a series of critical bugs in :func:`~pymaid.plot3d`, :func:`pymaid.Volume.combine`, :func:`~pymaid.cut_neuron`, :func:`pymaid.CatmaidNeuronList.remove_duplicates`,  :func:`~pymaid.get_skid_from_treenode` and :func:`~pymaid.neuron2json`
       - :func:`~pymaid.cut_neuron` now accepts multiple cut nodes
       - improved depth coloring in :func:`~pymaid.plot2d`
       - added depth coloring to :func:`~pymaid.plot2d` with method '3d' - see :doc:`here <depth_coloring>` for examples