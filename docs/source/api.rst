.. _api:

API Reference
=============

.. _api_fetch:

Connecting to a Server
++++++++++++++++++++++
Connections to servers are represented by a ``CatmaidInstance`` object. You
can either initialize it directly or store credentials as environmental
variables and use :func:`~pymaid.connect_catmaid` to do that for you.

.. autosummary::
    :toctree: generated/

    pymaid.connect_catmaid
    pymaid.CatmaidInstance
    pymaid.CatmaidInstance.copy
    pymaid.CatmaidInstance.clear_cache
    pymaid.CatmaidInstance.fetch
    pymaid.CatmaidInstance.load_cache
    pymaid.CatmaidInstance.make_url
    pymaid.CatmaidInstance.setup_cache
    pymaid.CatmaidInstance.save_cache

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
    pymaid.get_origin
    pymaid.get_skids_by_annotation
    pymaid.get_skids_by_name
    pymaid.get_skids_by_origin
    pymaid.get_skeleton_change

Annotations
-----------
Functions to fetch annotations:

.. autosummary::
    :toctree: generated/

    pymaid.get_annotated
    pymaid.get_annotations
    pymaid.get_annotation_details
    pymaid.get_user_annotations

Nodes
-----
Functions to fetch nodes and connectors:

.. autosummary::
    :toctree: generated/

    pymaid.find_nodes
    pymaid.get_connectors_in_bbox
    pymaid.get_skid_from_node
    pymaid.get_node_details
    pymaid.get_nodes_in_volume
    pymaid.get_node_location
    pymaid.get_node_table
    pymaid.get_node_info

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
    pymaid.get_connectivity_counts
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
    pymaid.get_user_stats

Volumes
-------
Functions to fetch volumes (meshes):

.. autosummary::
    :toctree: generated/

    pymaid.get_volume

Landmarks
---------
Functions to fetch data about landmarks.

.. autosummary::
    :toctree: generated/

    pymaid.get_landmarks
    pymaid.get_landmark_groups

Reconstruction samplers
-----------------------
Functions for reconstruction samplers:

.. autosummary::
    :toctree: generated/

    pymaid.get_sampler
    pymaid.get_sampler_domains
    pymaid.get_sampler_counts

Image data (tiles)
------------------
Functions to fetch and process image data. Note that this is not imported at
top level but has to be imported explicitly::

  >>> from pymaid import tiles
  >>> help(tiles.crop_neuron)

.. autosummary::
    :toctree: generated/

    pymaid.tiles.TileLoader
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
    pymaid.get_import_info
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
    pymaid.join_skeletons
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

Nodes
-----
Edit nodes:

.. autosummary::
    :toctree: generated/

    pymaid.add_node
    pymaid.delete_nodes
    pymaid.join_nodes
    pymaid.move_nodes
    pymaid.set_nodes_reviewed
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

.. _api_neurons:

CatmaidNeuron/List
++++++++++++++++++
Neuron/List objects representing neurons and lists thereof:

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuron
    pymaid.CatmaidNeuronList

:class:`~pymaid.CatmaidNeuron` are a subclasses of
<navis `https://navis.readthedocs.io/en/latest/`>_ ``TreeNeuron`` and
as such can be used with ``navis`` functions.

CatmaidNeuron/List methods
--------------------------
Methods common to both ``CatmaidNeurons`` and ``CatmaidNeuronLists``:

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuron.copy
    pymaid.CatmaidNeuron.downsample
    pymaid.CatmaidNeuron.plot3d
    pymaid.CatmaidNeuron.plot2d
    pymaid.CatmaidNeuron.prune_by_strahler
    pymaid.CatmaidNeuron.prune_by_volume
    pymaid.CatmaidNeuron.prune_distal_to
    pymaid.CatmaidNeuron.prune_proximal_to
    pymaid.CatmaidNeuron.prune_by_longest_neurite
    pymaid.CatmaidNeuron.prune_twigs
    pymaid.CatmaidNeuron.reroot
    pymaid.CatmaidNeuron.reload
    pymaid.CatmaidNeuron.resample
    pymaid.CatmaidNeuron.summary

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

.. _api_morph:

Working with Neurons
++++++++++++++++++++
As said previously, :class:`~pymaid.CatmaidNeuron` can be used with
<navis `https://navis.readthedocs.io/en/latest/`>_. This includes functions
to manipulate (e.g. prune, subset, resample), analyze (e.g. strahler index,
synapse flow) or plot neurons.

In addition, ``pymaid`` has a few more CATMAID-specific functions:

Morphology
----------
.. autosummary::
    :toctree: generated/

    pymaid.remove_tagged_branches
    pymaid.time_machine
    pymaid.prune_by_length
    pymaid.union_neurons

Analysis
--------
Various morphology metrics:

.. autosummary::
    :toctree: generated/

    pymaid.arbor_confidence

Predicting connectivity
-----------------------
Function to predict connectivity:

.. autosummary::
    :toctree: generated/

    pymaid.predict_connectivity

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

Filtering
---------
Functions to filter connectivity data:

.. autosummary::
    :toctree: generated/

    pymaid.filter_connectivity
    pymaid.shared_partners

.. _api_utility:

Utility
+++++++
Various utility functions.

.. autosummary::
    :toctree: generated/

    pymaid.eval_skids
    pymaid.set_pbars
    pymaid.set_loggers
    pymaid.shorten_name
