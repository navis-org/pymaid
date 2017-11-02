Overview
========

This page will give you a quick overview about PyMaid functions. 

Currently, PyMaid has a few thousand lines of code and this code is spread over a few logical modules:
:mod:`pymaid.pymaid`, :mod:`pymaid.core`, :mod:`pymaid.igraph_catmaid`, :mod:`pymaid.plotting`, :mod:`pymaid.cluster`,
:mod:`pymaid.morpho`, :mod:`pymaid.rmaid`, :mod:`pymaid.user_stats` and :mod:`pymaid.b3d`. 

You can import them explicitly...

>>> from pymaid import cluster
>>> results = cluster.cluster_by_synapse_placement(some_neurons)

but for the most part you will just import the whole package. This, conveniently, flattens the namespace 
and (almost) all functions are then available from the base level (i.e. ``pymaid.function``).

>>> import pymaid
>>> results = pymaid.cluster_by_synapes_placement(some_neurons)


Neuron and NeuronList objects
------------------------------

Neurons and lists of neurons are represented by 
:class:`~pymaid.core.CatmaidNeuron` and :class:`~pymaid.core.CatmaidNeuronList`.

They can be minimally initialized with just skeleton IDs but usually you would
get them returned from :func:`~pymaid.pymaid.get_neuron`.

Data (e.g. nodes, connectors, name, review status, annotation) are retrieved/calculated on-demand the first time they are **explicitly** requested. 
CatmaidNeuronList also allows indexing similar to pandas DataFrames 
(see examples).

Selection of attributes:

- ``skeleton_id``: neurons' skeleton ID	
- ``neuron_name``: neurons' name
- ``nodes``: treenode table as pandas DataFrame
- ``connectors``: connector table pandas DataFrame
- ``tags``: node tags (dict)
- ``annotations``: list of neurons' annotations
- ``cable_length``: cable length(s) in nm
- ``review_status``: review status of neuron(s)
- ``soma``: treenode ID of soma (if applicable)
- ``slabs``: slabs (linear segments)
- ``igraph``: iGraph representation of the neuron


Selection of class methods:

- :func:`~pymaid.core.CatmaidNeuron.plot3d`: create 3D plot of neuron(s)
- :func:`~pymaid.core.CatmaidNeuron.plot2d`: create 2D plot of neuron(s)
- :func:`~pymaid.core.CatmaidNeuron.plot_dendrogram`: plot dendogram of neuron
- :func:`~pymaid.core.CatmaidNeuron.prune_by_strahler`: prune neuron by strahler index
- :func:`~pymaid.core.CatmaidNeuron.prune_by_volume`: prune neuron in- or outside of a volume
- :func:`~pymaid.core.CatmaidNeuron.prune_distal_to`: cut off nodes distal to a given treenode
- :func:`~pymaid.core.CatmaidNeuron.prune_proximal_to`: cut off nodes proximal to a given treenode
- :func:`~pymaid.core.CatmaidNeuron.reroot`: reroot neuron to given node
- :func:`~pymaid.core.CatmaidNeuron.reload`: reload neuron(s) from server
- :func:`~pymaid.core.CatmaidNeuron.summary`: pandas DataFrame with basic parameters of neuron(s)
- :func:`~pymaid.core.CatmaidNeuron.downsample`: downsample neuron(s)
- :func:`~pymaid.core.CatmaidNeuron.copy`: returns deep copy of the object
- :func:`~pymaid.core.CatmaidNeuron.from_swc`: creates CatmaidNeuron from swc file
- :func:`~pymaid.core.CatmaidNeuronList.to_json`: saves neuronlist as json that can be opend in CATMAID's selection widget


List of PyMaid functions
------------------------

Functions to retrieve data from server:

- :class:`~pymaid.pymaid.CatmaidInstance`: this class is used you set up and store the connection to your CATMAID server
- :func:`~pymaid.pymaid.add_annotations`: use to add annotation(s) to neuron(s)
- :func:`~pymaid.pymaid.add_tags`: add tags of treenodes or connectors
- :func:`~pymaid.pymaid.delete_tags`: delete tags of treenodes or connectors
- :func:`~pymaid.pymaid.delete_neuron`: delete entire neurons
- :func:`~pymaid.pymaid.get_arbor`: similar to get_neuron but more detailed information on connectors
- :func:`~pymaid.pymaid.get_annotations`: get annotations of a set of neurons (annotation only)
- :func:`~pymaid.pymaid.get_annotation_details`: get detailed annotations for a set of neurons (includes user and timestamp)
- :func:`~pymaid.pymaid.get_connectors`: get connectors (synapses, abutting and/or gap junctions) for set of neurons
- :func:`~pymaid.pymaid.get_connector_details`: get details for connector (i.e. all neurons connected to it)
- :func:`~pymaid.pymaid.get_connectors_between`: get connectors connecting two sets of neurons
- :func:`~pymaid.pymaid.get_contributor_statistics`: get contributors (nodes, synapses, etc) for a set of neurons
- :func:`~pymaid.pymaid.get_edges`: get edges (connections) between sets of neurons
- :func:`~pymaid.pymaid.get_history`: retrieve project history similar to the project statistics widget
- :func:`~pymaid.pymaid.get_logs`: get what the log widged shows (merges, splits, etc.)
- :func:`~pymaid.pymaid.get_label_list`: get a list of all treenode labels (tags) in the project
- :func:`~pymaid.pymaid.get_names`: retrieve names of a set of skeleton IDs
- :func:`~pymaid.pymaid.get_neuron`: get neuron skeleton(s) - i.e. what the 3D viewer in CATMAID shows
- :func:`~pymaid.pymaid.get_neurons_in_volume`: get neurons in a defined box volume
- :func:`~pymaid.pymaid.get_neuron_list`: retrieve neurons that fit certain criteria (e.g. user, size, dates)
- :func:`~pymaid.pymaid.get_node_user_details`: get details (creator, edition time, etc.) for individual nodes
- :func:`~pymaid.pymaid.get_partners`: retrieve connected partners for a list of neurons
- :func:`~pymaid.pymaid.get_partners_in_volume`: retrieve connected partners for a list of neurons within a given Catmaid volume
- :func:`~pymaid.pymaid.get_paths`: get possible paths between two sets of neurons
- :func:`~pymaid.pymaid.get_review`: get review status for set of neurons
- :func:`~pymaid.pymaid.get_review_details`: get review status (reviewer + timestamp) for each individual node
- :func:`~pymaid.pymaid.get_skids_by_annotation`: get skeleton IDs that are annotated with a given annotation
- :func:`~pymaid.pymaid.get_skids_by_name`: get skeleton IDs of neurons with given names
- :func:`~pymaid.pymaid.get_node_tags`: get tags of a set of treenodes or connectors
- :func:`~pymaid.pymaid.get_treenode_info`: retrieve info (i.e. skeleton ID) for a set of treenodes
- :func:`~pymaid.pymaid.get_treenode_table`: retrieve treenode table for given neurons
- :func:`~pymaid.pymaid.get_user_annotations`: get list of annotations used by given user(s)
- :func:`~pymaid.pymaid.get_user_list`: get list of users in the project
- :func:`~pymaid.pymaid.get_volume`: get volume (verts + faces) of CATMAID volumes
- :func:`~pymaid.pymaid.rename_neurons`: use to rename neurons (careful!)
- :func:`~pymaid.pymaid.url_to_coordinates`: generate urls to coordinates


Wrappers to use igraph:

- :func:`~pymaid.igraph_catmaid.cluster_nodes_w_synapses`: uses iGraph's `shortest_paths_dijkstra` to cluster nodes with synapses
- :func:`~pymaid.igraph_catmaid.dist_from_root`: calculates geodesic (along-the-arbor) distances for nodes to root node
- :func:`~pymaid.igraph_catmaid.matrix2graph`: generates iGraph representation from adjacency matrix
- :func:`~pymaid.igraph_catmaid.network2graph`: generates iGraph representation from set of neurons
- :func:`~pymaid.igraph_catmaid.neuron2graph`: generates iGraph representation of neuron morphology

Functions to plot neurons:

- :func:`~pymaid.plotting.plot2d`: generates 2D plots of neurons
- :func:`~pymaid.plotting.plot3d`: uses either `Vispy <http://vispy.org>`_ or `Plotly <http://plot.ly>`_ to generate 3D plots of neurons
- :func:`~pymaid.plotting.plot_network`: uses iGraph and `Plotly <http://plot.ly>`_ to generate network plots
- :func:`~pymaid.plotting.clear3d`: clear 3D canvas
- :func:`~pymaid.plotting.close3d`: close 3D canvas and wipe from memory
- :func:`~pymaid.plotting.screenshot`: save screenshot

Functions for clustering:

- :func:`~pymaid.cluster.adjacency_matrix`: create a Pandas dataframe containing the adjacency matrix for two sets of neurons
- :func:`~pymaid.cluster.cluster_by_connectivity`: returns distance matrix based on connectivity similarity (Jarrell et al., 2012)
- :func:`~pymaid.cluster.group_matrix`: groups matrix by columns or rows - use to e.g. collapse connectivity matrix into groups of neurons
- :func:`~pymaid.cluster.cluster_xyz`: cluster points (synapses, nodes) based on eucledian distance
- :func:`~pymaid.cluster.cluster_by_synapse_placement`: hierarchical clustering of neurons based on synapse placement

Functions for morphological analyses:

- :func:`~pymaid.morpho.arbor_confidence`: calculates confidence along the arbor
- :func:`~pymaid.morpho.cable_within_distance`: calculate cable between pairs of neurons that is within given distance
- :func:`~pymaid.morpho.calc_cable`: calculate cable length of given neuron
- :func:`~pymaid.morpho.calc_strahler_index`: calculate strahler index for each node
- :func:`~pymaid.morpho.calc_flow_centrality`: implementation of synapse flow centrality algorithm by Schneider-Mizell et al., 2016
- :func:`~pymaid.morpho.classify_nodes`: adds a new column to a neuron's dataframe categorizing each node as branch, slab, leaf or root
- :func:`~pymaid.morpho.cut_neuron`: cut neuron at a node or node tag
- :func:`~pymaid.morpho.distal_to`: use this to check spatial relation of nodes within a neuron
- :func:`~pymaid.morpho.downsample_neuron`: takes skeleton data and reduces the number of nodes while preserving synapses, branch points, etc.
- :func:`~pymaid.morpho.flow_centrality_split`: split neuron into axon, dendrite and primary neurite based on synapse flow centrality
- :func:`~pymaid.morpho.in_volume`: test if points are within given CATMAID volume
- :func:`~pymaid.morpho.longest_neurite`: prunes neuron to its longest neurite
- :func:`~pymaid.morpho.prune_by_strahler`: prunes the neuron by strahler index
- :func:`~pymaid.morpho.reroot_neuron`: reroot neuron to a specific node
- :func:`~pymaid.morpho.synapse_root_distances`: similar to :func:`pymaid.igraph_catmaid.dist_from_root` but does not use iGraph
- :func:`~pymaid.morpho.stitch_neurons`: stitch neurons back together


Interface with R (nat, rcatmaid, etc.):

- :func:`~pymaid.rmaid.init_rcatmaid`: initialize connection with Catmaid server in R
- :func:`~pymaid.rmaid.data2py`: wrapper to convert R data to Python 
- :func:`~pymaid.rmaid.nblast`: wrapper to nblast a set neurons against external database
- :func:`~pymaid.rmaid.nblast_allbyall`: wrapper to nblast a set of neurons against each other
- :func:`~pymaid.rmaid.neuron2py`: converts R neuron and neuronlist objects to Pymaid neurons
- :func:`~pymaid.rmaid.neuron2r`: converts Pymaid neuron and list of neurons to R neuron and neuronlist objects, respectively

Functions to analyse user stats:

- :func:`~pymaid.user_stats.get_time_invested`: calculate the time users have spent working on a set of neurons
- :func:`~pymaid.user_stats.get_user_contributions`: returns contributions per user for a set of neurons