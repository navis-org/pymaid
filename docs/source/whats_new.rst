.. _whats_new:

What's new?
===========

.. list-table::
   :widths: 7 7 86
   :header-rows: 1

   * - Version
     - Date
     -
   * - 2.1.0
     - 04/04/22
     - With this release we mainly follow some renamed functions in ``navis`` but
       we also make pymaid play more nicely with public CATMAID instances and
       of course fix a couple bugs.
       One important thing to mention is that the default `max_threads` for
       :class:`pymaid.CatmaidInstance` is now 10 (down from 100). This also
       applies to :func:`pymaid.connect_catmaid`. If your internet can handle
       more connections, feel free to up it back to 100.
   * - 2.0.0
     - 21/11/20
     - This release marks a huge break in the way pymaid works: it is now
       fully dependent on and compatible with `navis <https://navis.readthedocs.io/en/latest/>`_.
       - :class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList` are now subclasses of ``navis.TreeNeuron`` and ``navis.NeuronList``, respectively
       - all functions not specific to CATMAID have been removed in favour of the equivalent ``navis`` function
       - for consistency, all use of ``treenode`` have been replace with ``node``: for example, the ``treenode_id`` column in node tables is now ``node_id``
       - API docs and tutorials have been updated to use navis functions instead
   * - 1.1.0
     - 4/4/20
     - - changed argument names, types and order for :class:`CATMAIDInstance` to facilitate connecting to public Catmaid servers
       - was: ``(server, authname, authpassword, authtoken, ...)``, now is ``(server_url, token, http_user=None, http_pw=None, ...)``
   * - 1.0.1
     - 23/1/20
     - - changed to semantic versioning: `major.minor.patch`
       - new functions: :func:`~pymaid.find_first_branchpoint`, :func:`~pymaid.set_nodes_reviewed`, :func:`~pymaid.prune_by_length`, :func:`~pymaid.get_skeleton_change`
       - plotting now accepts matplotlib colormaps
       - tons of small improvements and bugfixes
   * - 0.101
       to
       0.103
     - 25/09/19
     - - teach :func:`~pymaid.upload_neuron` and :func:`~pymaid.transfer_neuron` to make use of new ``source`` fields in CATMAID
       - pymaid will now warn if run in Jupyter lab without plotly renderer extension
       - renamed function ``to_dotproduct()`` to :func:`~pymaid.to_dotprops`
       - auto-detect if no display available
       - various improvements and bugfixes
   * - 0.100
     - 03/09/19
     - - improve :func:`~pymaid.get_nodes_in_volume`
       - add optional headless mode (set env variable `PYMAID_HEADLESS=TRUE`)
   * - 0.99
     - 12/08/19
     - - use ``ujson`` library if available for faster unpacking
       - new function: :func:`~pymaid.update_node_confidence`, :func:`~pymaid.get_connectivity_counts`
       - various improvements and bug fixes
   * - 0.98
     - 21/06/19
     - - new functions: :func:`~pymaid.join_nodes`, :func:`~pymaid.link_connector`, :func:`~pymaid.join_skeletons`, :func:`~pymaid.replace_skeleton`, :func:`~pymaid.link_connector`, :func:`~pymaid.delete_nodes`, :func:`~pymaid.add_connector`
       - reworked ``get_nodes_by_tag()`` and renamed to :func:`~pymaid.find_nodes`
   * - 0.97
     - 21/06/19
     - - new functions: :func:`~pymaid.upload_volume`, :func:`~pymaid.shared_partners`
       - improved :func:`~pymaid.upload_neuron`, :func:`~pymaid.from_swc`, :func:`~pymaid.plot2d` and more
       - fixes for :func:`~pymaid.remove_annotations`, :func:`~pymaid.get_neuron` and more
   * - 0.96
     - 22/05/19
     - - fixed bug in :func:`~pymaid.plot3d` using plotly
   * - 0.95
     - 17/05/19
     - - new function: :func:`~pymaid.get_connectors_in_bbox`
       - new multi-ray option for :func:`~pymaid.in_volume` for complicated meshes
       - other improvements: :func:`~pymaid.from_swc`
       - many bugfixes
   * - 0.94
     - 09/04/19
     - - started reworking vispy plot3d: in brief, will try reducing the number of shader programs running
       - new functions: :func:`~pymaid.break_fragments`, :func:`~pymaid.heal_fragmented_neuron`, :func:`~pymaid.update_radii`, :func:`~pymaid.get_neuron_id`, :func:`~pymaid.rmaid.neuron2dps`
       - :class:`~pymaid.Volumes` now allow multiplication and division - will apply to vertex coordinates
       - improved: :func:`~pymaid.from_swc`, :func:`~pymaid.to_swc`, :func:`~pymaid.predict_connectivity`, :func:`~pymaid.stitch_neurons`, :func:`~pymaid.reroot_neuron`, :func:`~pymaid.upload_neuron`
       - fixes in :func:`~pymaid.delete_neuron`, :func:`~pymaid.rename_neurons`, :func:`~pymaid.get_history`, :func:`~pymaid.split_axon_dendrite`, :func:`~pymaid.CatmaidNeuronList.remove_duplicates`
       - updated to networkx 2.2
   * - 0.93
     - 05/02/19
     - - various improvements to the Blender interface ``pymaid.b3d``
       - improved :func:`~pymaid.predict_connectivity`
       - new functions to import/transfer neurons to/between Catmaid instances: :func:`pymaid.upload_neuron` and :func:`pymaid.transfer_neuron`
       - new function :func:`pymaid.sparseness` to calculate lifetime sparseness
       - tons of bug fixes
   * - 0.92
     - 06/11/18
     - - new pymaid.Volume methods: ``to_csv`` and ``from_csv``
       - new functions: :func:`~pymaid.add_meta_annotations`, :func:`~pymaid.remove_meta_annotations`, :func:`~pymaid.get_annotated`
       - some under-the-hood changes following change in CATMAID's API
       - general bug fixes and improvements
   * - 0.91
     - 31/10/18
     - - new CatmaidInstance attributes to get info on your server: ``catmaid_version``, ``available_projects`` and ``image_stacks``
       - new functions: :func:`~pymaid.shorten_name`, :func:`~pymaid.get_user_stats`, :func:`~pymaid.intersection_matrix`, :func:`~pymaid.get_node_location`
       - various improvements and bugfixes
   * - 0.90
     - 20/09/18
     - - vispy 3d viewer overhaul: prettier, better picking, new shortcuts
       - indexing of :class:`~pymaid.CatmaidNeuronList` via ``.skid[]`` now returns results in order of query
       - new function: :func:`~pymaid.find_nodes`
       - new function: :func:`~pymaid.connection_density`
       - improved :func:`~pymaid.split_axon_dendrite`
       - improved :func:`~pymaid.to_swc` and :func:`~pymaid.from_swc`
       - improved :ref:`neuronlist math and comparisons <neuronlist_math>`
       - :func:`~pymaid.plot2d` and :func:`~pymaid.plot3d` now accept lists of colors
       - :func:`~pymaid.has_soma` is now much faster
       - faster neuron import in :ref:`blender_3d`
       - improved docstrings
       - various bugfixes
   * - 0.89
     - 14/08/18
     - - new function: :func:`~pymaid.cytoscape.watch_network` constantly pushes updates Cytoscape
       - new function: :func:`~pymaid.get_nth_partners` returns neurons connected via n hops
       - by default, :func:`~pymaid.plot3d` now chooses the backend automatically: vispy for terminal sessions, plotly for Jupyter notebook/lab
       - :func:`~pymaid.get_skids_by_annotation` now accepts negative search criteria
       - :func:`~pymaid.from_swc` now imports multiple SWCs at a time
       - major improvements to caching system
       - by default, progress bars will now vanish after completion
       - followed changes in CATMAID API regarding node tables
       - various bugfixes
   * - 0.88
     - 29/07/18
     - - data caching for faster queries, see :doc:`caching demo <data_caching>`
       - new function: :func:`~pymaid.smooth_neuron`
       - :func:`~pymaid.resample_neuron` now resamples radius too
       - :func:`~pymaid.guess_radius` interpolation now takes distance along spines into account
       - :func:`~pymaid.despike_neuron` is now able to catch spikes that consist of multiple nodes
       - :func:`~pymaid.calc_cable` is now deprecated
       - general improvements to docstrings
   * - 0.87
     - 20/07/18
     - - :func:`~pymaid.get_team_contributions` now takes link creation into account
       - :func:`~pymaid.get_time_invested` should be way faster now
       - :func:`~pymaid.geodesic_matrix` now returns a SparseDataFrame to save memory
       - added :func:`pymaid.CatmaidNeuron.to_dataframe` method
       - general improvements and docstrings
   * - 0.86
     - 16/07/18
     - - arithmetric operations with CatmaidNeuron/Lists will now warn if skeleton IDs match but neuron objects are not identical. See :doc:`here <neuronlist_math>` for explanation.
       - fixed a bug when using regex to query for neurons that led to duplicate skeleton IDs being returned
   * - 0.85
     - 13/07/18
     - - fixed a series of critical bugs in :func:`~pymaid.plot3d`, :func:`pymaid.Volume.combine`, :func:`~pymaid.cut_neuron`, :func:`pymaid.CatmaidNeuronList.remove_duplicates`,  :func:`~pymaid.get_skid_from_node` and :func:`~pymaid.neuron2json`
       - :func:`~pymaid.cut_neuron` now accepts multiple cut nodes
       - improved depth coloring in :func:`~pymaid.plot2d`
       - added depth coloring to :func:`~pymaid.plot2d` with method '3d' - see :doc:`here <depth_coloring>` for examples
