.. _api:

API Reference
=============

.. _api_fetch:

Fetching data
+++++++++++++
This section contains functions to pull data from a CATMAID server.

Neurons
-------
Functions related to searching for neurons and fetching their 3D skeletons:

.. autosummary::
    :toctree: generated/

    pymaid.find_neurons
    pymaid.get_arbor
    pymaid.get_names
    pymaid.get_neuron
    pymaid.get_neuron_id
    pymaid.get_neurons_in_volume
    pymaid.get_neuron_list
    pymaid.get_skids_by_annotation
    pymaid.get_skids_by_name

Annotations
-----------
Functions to fetch annotations:

.. autosummary::
    :toctree: generated/

    pymaid.get_annotated
    pymaid.get_annotations
    pymaid.get_annotation_details
    pymaid.get_user_annotations

Treenodes
----------
Functions to fetch treenodes and connectors:

.. autosummary::
    :toctree: generated/

    pymaid.find_treenodes
    pymaid.get_connectors_in_bbox
    pymaid.get_skid_from_treenode
    pymaid.get_node_details
    pymaid.get_node_location
    pymaid.get_treenode_table
    pymaid.get_treenode_info

Tags
----
Functions to fetch node tags:

.. autosummary::
    :toctree: generated/

    pymaid.get_label_list
    pymaid.get_node_tags

Connectivity
------------
Functions to fetch connectivity data:

.. autosummary::
    :toctree: generated/

    pymaid.adjacency_matrix
    pymaid.adjacency_from_connectors
    pymaid.cn_table_from_connectors
    pymaud.get_connectivity_counts
    pymaid.get_connectors
    pymaid.get_connector_details
    pymaid.get_connectors_between
    pymaid.get_connector_links
    pymaid.get_connectors_in_bbox
    pymaid.get_edges
    pymaid.get_partners
    pymaid.get_partners_in_volume
    pymaid.get_nth_partners
    pymaid.get_paths

.. _api_userstats:

User stats
----------
Functions to fetch user stats:

.. autosummary::
    :toctree: generated/

    pymaid.get_contributor_statistics
    pymaid.get_history
    pymaid.get_logs
    pymaid.get_transactions
    pymaid.get_team_contributions
    pymaid.get_time_invested
    pymaid.get_user_list
    pymaid.get_user_contributions

Volumes
-------
Functions to fetch volumes (meshes):

.. autosummary::
    :toctree: generated/

    pymaid.get_volume

Image data (tiles)
------------------
Functions to fetch and process image data. Note that this is not imported at
top level but has to be imported explicitly::

  >>> from pymaid import tiles
  >>> help(tiles.crop_neuron)

.. autosummary::
    :toctree: generated/

    pymaid.tiles.LoadTiles
    pymaid.tiles.crop_neuron

.. _api_misc:

Misc
----
Functions to fetch miscellaneous data:

.. autosummary::
    :toctree: generated/

    pymaid.clear_cache
    pymaid.has_soma
    pymaid.get_cable_lengths
    pymaid.get_review
    pymaid.get_review_details
    pymaid.url_to_coordinates

.. _api_upload:

Uploading data
++++++++++++++
Functions to push data to a CATMAID server. Use these with caution!

Neurons
-------
Upload, rename, move or delete neurons:

.. autosummary::
    :toctree: generated/

    pymaid.delete_neuron
    pymaid.differential_upload
    pymaid.push_new_root
    pymaid.rename_neurons
    pymaid.replace_skeleton
    pymaid.transfer_neuron
    pymaid.update_radii
    pymaid.upload_neuron

Annotations
-----------
Edit neuron annotations:

.. autosummary::
    :toctree: generated/

    pymaid.add_annotations
    pymaid.add_meta_annotations
    pymaid.remove_annotations
    pymaid.remove_meta_annotations

Treenodes
----------
Edit treenodes:

.. autosummary::
    :toctree: generated/

    pymaid.add_treenode
    pymaid.delete_nodes
    pymaid.move_nodes
    pymaid.update_node_confidence

Connectivity
------------
Edit connectors and connector links:

.. autosummary::
    :toctree: generated/

    pymaid.add_connector
    pymaid.link_connector

Tags
----
Edit tags:

.. autosummary::
    :toctree: generated/

    pymaid.add_tags
    pymaid.delete_tags

Volumes
-------
Upload volumes:

.. autosummary::
    :toctree: generated/

    pymaid.upload_volume

CatmaidInstance
+++++++++++++++
Methods of the remote CatmaidInstance object interfacing with CATMAID server:

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidInstance
    pymaid.CatmaidInstance.copy
    pymaid.CatmaidInstance.clear_cache
    pymaid.CatmaidInstance.fetch
    pymaid.CatmaidInstance.load_cache
    pymaid.CatmaidInstance.make_url
    pymaid.CatmaidInstance.setup_cache
    pymaid.CatmaidInstance.save_cache

.. _api_neurons:

CatmaidNeuron/List
++++++++++++++++++
Neuron/List objects representing neurons and lists thereof:

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuron
    pymaid.CatmaidNeuronList

CatmaidNeuron/List methods
--------------------------
Methods common to both CatmaidNeurons and CatmaidNeuronLists:

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuron.copy
    pymaid.CatmaidNeuron.downsample
    pymaid.CatmaidNeuron.plot3d
    pymaid.CatmaidNeuron.plot2d
    pymaid.CatmaidNeuron.plot_dendrogram
    pymaid.CatmaidNeuron.prune_by_strahler
    pymaid.CatmaidNeuron.prune_by_volume
    pymaid.CatmaidNeuron.prune_distal_to
    pymaid.CatmaidNeuron.prune_proximal_to
    pymaid.CatmaidNeuron.prune_by_longest_neurite
    pymaid.CatmaidNeuron.reroot
    pymaid.CatmaidNeuron.reload
    pymaid.CatmaidNeuron.resample
    pymaid.CatmaidNeuron.summary
    pymaid.CatmaidNeuron.from_swc
    pymaid.CatmaidNeuron.to_swc

CatmaidNeuronList-specific
--------------------------
Methods specific to CatmaidNeuronLists:

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuronList.to_selection
    pymaid.CatmaidNeuronList.from_selection
    pymaid.CatmaidNeuronList.has_annotation
    pymaid.CatmaidNeuronList.head
    pymaid.CatmaidNeuronList.tail
    pymaid.CatmaidNeuronList.itertuples
    pymaid.CatmaidNeuronList.remove_duplicates
    pymaid.CatmaidNeuronList.sample
    pymaid.CatmaidNeuronList.summary
    pymaid.CatmaidNeuronList.mean
    pymaid.CatmaidNeuronList.sum
    pymaid.CatmaidNeuronList.sort_values

Volumes
-------
Methods of Volume object representing CATMAID meshes:

.. autosummary::
    :toctree: generated/

    pymaid.Volume
    pymaid.Volume.combine
    pymaid.Volume.from_csv
    pymaid.Volume.plot3d
    pymaid.Volume.resize
    pymaid.Volume.to_csv
    pymaid.Volume.to_2d
    pymaid.Volume.to_trimesh


.. _api_plot:

Plotting
++++++++
Functions for plotting.

.. autosummary::
    :toctree: generated/

    pymaid.plot3d
    pymaid.plot2d
    pymaid.plot1d
    pymaid.plot_network
    pymaid.plot_history
    pymaid.clear3d
    pymaid.close3d
    pymaid.get_viewer
    pymaid.screenshot

Vispy 3D viewer
---------------
Methods of vispy 3D viewer:

.. autosummary::
    :toctree: generated/

    pymaid.Viewer
    pymaid.Viewer.add
    pymaid.Viewer.clear
    pymaid.Viewer.close
    pymaid.Viewer.colorize
    pymaid.Viewer.set_colors
    pymaid.Viewer.hide_neurons
    pymaid.Viewer.unhide_neurons
    pymaid.Viewer.screenshot
    pymaid.Viewer.show


.. _api_morph:

Neuron Morphology
+++++++++++++++++
Functions to analyse and manipulate neuron morphology.

Manipulation
------------
Change neuron morphology:

.. autosummary::
    :toctree: generated/

    pymaid.average_neurons
    pymaid.break_fragments
    pymaid.cut_neuron
    pymaid.despike_neuron
    pymaid.guess_radius
    pymaid.heal_fragmented_neuron
    pymaid.longest_neurite
    pymaid.prune_by_strahler
    pymaid.reroot_neuron
    pymaid.remove_tagged_branches
    pymaid.smooth_neuron
    pymaid.split_axon_dendrite
    pymaid.split_into_fragments
    pymaid.stitch_neurons
    pymaid.subset_neuron
    pymaid.time_machine
    pymaid.tortuosity
    pymaid.union_neurons

Resampling
----------
Resample neurons:

.. autosummary::
    :toctree: generated/

    pymaid.downsample_neuron
    pymaid.resample_neuron

Analysis
--------
Various morphology metrics:

.. autosummary::
    :toctree: generated/

    pymaid.arbor_confidence
    pymaid.bending_flow
    pymaid.calc_cable
    pymaid.classify_nodes
    pymaid.find_main_branchpoint
    pymaid.flow_centrality
    pymaid.segregation_index
    pymaid.strahler_index

Distances
---------
Functions to work with (geodesic -> "along-the-arbor") distances:

.. autosummary::
    :toctree: generated/

    pymaid.cable_overlap
    pymaid.distal_to
    pymaid.dist_between
    pymaid.geodesic_matrix

Intersection
------------
Functions to query whether points intersect with a given volume:

.. autosummary::
    :toctree: generated/

    pymaid.in_volume
    pymaid.intersection_matrix

.. _api_con:

Connectivity
++++++++++++
Various functions to work with connectivity data.

Graphs
------
Turn neurons or connectivity into iGraph or networkX objects:

.. autosummary::
    :toctree: generated/

    pymaid.neuron2nx
    pymaid.neuron2igraph
    pymaid.neuron2KDTree
    pymaid.network2nx
    pymaid.network2igraph

Predicting connectivity
-----------------------
Function to predict connectivity:

.. autosummary::
    :toctree: generated/

    pymaid.predict_connectivity

Adjacency matrices
------------------
Function to generate or manipulate adjacency matrices:

.. autosummary::
    :toctree: generated/

    pymaid.adjacency_matrix
    pymaid.group_matrix

Analyses
--------
Functions to analyse connectivity:

.. autosummary::
    :toctree: generated/

    pymaid.cluster_by_connectivity
    pymaid.cluster_by_synapse_placement
    pymaid.ClustResults
    pymaid.connection_density
    pymaid.sparseness

Plotting network
----------------
Functions to plot networks:

.. autosummary::
    :toctree: generated/

    pymaid.plot_network

Filtering
---------
Functions to filter connectivity data:

.. autosummary::
    :toctree: generated/

    pymaid.filter_connectivity
    pymaid.shared_partners

Import/Export
+++++++++++++
Functions to import and export neuron objects:

.. autosummary::
    :toctree: generated/

    pymaid.from_swc
    pymaid.json2neuron
    pymaid.neuron2json
    pymaid.to_swc

.. _api_interfaces:

Interfaces
++++++++++
Interfaces with various external tools. These modules have to be imported
explicitly as they are not imported at top level. For example::

   >>> from pymaid import b3d
   >>> h = b3d.handler()

.. _api_b3d:

Blender API
-----------
Functions to be run inside `Blender 3D <https://www.blender.org/>`_ and import
CATMAID data (see Examples)

The interface is realised through a :class:`~pymaid.b3d.handler` object. It
is used to import objects and facilitate working with them programmatically
once they are imported.

.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler

Objects
~~~~~~~
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.add
    pymaid.b3d.handler.clear
    pymaid.b3d.handler.select
    pymaid.b3d.handler.hide
    pymaid.b3d.handler.unhide

Materials
~~~~~~~~~
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.color
    pymaid.b3d.handler.colorize
    pymaid.b3d.handler.emit
    pymaid.b3d.handler.use_transparency
    pymaid.b3d.handler.alpha
    pymaid.b3d.handler.bevel

Selections
~~~~~~~~~~
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.select

    pymaid.b3d.object_list.set
    pymaid.b3d.object_list.select
    pymaid.b3d.object_list.color
    pymaid.b3d.object_list.colorize
    pymaid.b3d.object_list.emit
    pymaid.b3d.object_list.use_transparency
    pymaid.b3d.object_list.alpha
    pymaid.b3d.object_list.bevel
    pymaid.b3d.object_list.hide
    pymaid.b3d.object_list.unhide
    pymaid.b3d.object_list.hide_others
    pymaid.b3d.object_list.render
    pymaid.b3d.object_list.delete
    pymaid.b3d.object_list.to_json


Cytoscape API
-------------
Functions to use `Cytoscape <https://cytoscape.org/>`_ via the cyREST API.

.. autosummary::
    :toctree: generated/

    pymaid.cytoscape.generate_network
    pymaid.cytoscape.get_client
    pymaid.cytoscape.watch_network

R interface (rMAID)
-------------------
Bundle of functions to use R libraries.

.. autosummary::
    :toctree: generated/
    pymaid.rmaid.data2py
    pymaid.rmaid.dotprops2py
    pymaid.rmaid.get_neuropil
    pymaid.rmaid.init_rcatmaid
    pymaid.rmaid.nblast
    pymaid.rmaid.nblast_allbyall
    pymaid.rmaid.NBLASTresults
    pymaid.rmaid.neuron2py
    pymaid.rmaid.neuron2dps
    pymaid.rmaid.neuron2r

Utility
+++++++
Various utility functions.

.. autosummary::
    :toctree: generated/

    pymaid.eval_skids
    pymaid.set_pbars
    pymaid.set_loggers
    pymaid.shorten_name
