Overview
========

This page will give you a quick overview about PyMaid functions.


Neuron and NeuronList objects
------------------------------

Neurons and lists of neurons are represented by :class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList`.

They can be minimally initialized with just skeleton IDs but usually you would
get them returned from :func:`~pymaid.get_neuron`.

Data (e.g. nodes, connectors, name, review status, annotation) are retrieved/calculated on-demand the first time they are **explicitly** requested. CatmaidNeuronList also allows indexing similar to pandas DataFrames (see examples).

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
- ``segments``: list of linear segments 
- ``graph``: NetworkX graph representation of the neuron
- ``igraph``: iGraph representation of the neuron (if available)
- ``dps``: Dotproduct representation of the neuron


Selection of class methods:

- :func:`~pymaid.CatmaidNeuron.plot3d`: create 3D plot of neuron(s)
- :func:`~pymaid.CatmaidNeuron.plot2d`: create 2D plot of neuron(s)
- :func:`~pymaid.CatmaidNeuron.plot_dendrogram`: plot dendogram of neuron
- :func:`~pymaid.CatmaidNeuron.prune_by_strahler`: prune neuron by strahler index
- :func:`~pymaid.CatmaidNeuron.prune_by_volume`: prune neuron in- or outside of a volume
- :func:`~pymaid.CatmaidNeuron.prune_distal_to`: cut off nodes distal to a given treenode
- :func:`~pymaid.CatmaidNeuron.prune_proximal_to`: cut off nodes proximal to a given treenode
- :func:`~pymaid.CatmaidNeuron.reroot`: reroot neuron to given node
- :func:`~pymaid.CatmaidNeuron.reload`: reload neuron(s) from server
- :func:`~pymaid.CatmaidNeuron.summary`: pandas DataFrame with basic parameters of neuron(s)
- :func:`~pymaid.CatmaidNeuron.resample`: resample neuron(s) to given resolution
- :func:`~pymaid.CatmaidNeuron.downsample`: downsample/simplify neuron(s)
- :func:`~pymaid.CatmaidNeuron.copy`: returns deep copy of the object
- :func:`~pymaid.CatmaidNeuron.from_swc`: creates CatmaidNeuron from swc file
- :func:`~pymaid.CatmaidNeuronList.to_json`: saves neuronlist as json that can be opend in CATMAID's selection widget
- :func:`~pymaid.CatmaidNeuronList.from_json`: create a neuronlist from a CATMAID json selection

See :class:`~pymaid.CatmaidNeuron` or ``help(pymaid.CatmaidNeuron)`` for complete list.

List of PyMaid functions
------------------------

Fetching data from server
*************************

- :class:`~pymaid.CatmaidInstance`: Base class to set up and store the connection to your CATMAID server

Neurons:

- :func:`~pymaid.get_neuron`: returns CatmaidNeuron/Lists of neuron(s)
- :func:`~pymaid.delete_neuron`: delete entire neurons
- :func:`~pymaid.find_neurons`: search for neurons based on a variety of parameters
- :func:`~pymaid.get_arbor`: similar to get_neuron but more detailed information on connectors
- :func:`~pymaid.get_neurons_in_volume`: returns neurons in a defined box volume
- :func:`~pymaid.get_neuron_list`: retrieve neurons that fit certain criteria (e.g. user, size, dates)
- :func:`~pymaid.get_skids_by_annotation`: returns skeleton IDs that are annotated with a given annotation
- :func:`~pymaid.get_skids_by_name`: returns skeleton IDs of neurons with given names
- :func:`~pymaid.rename_neurons`: use to rename neurons (careful!)
- :func:`~pymaid.get_names`: retrieve names of a set of skeleton IDs

Annotations:

- :func:`~pymaid.add_annotations`: use to add annotation(s) to neuron(s)
- :func:`~pymaid.get_annotations`: returns annotations of a set of neurons (annotation only)
- :func:`~pymaid.get_annotation_details`: returns detailed annotations for a set of neurons (includes user and timestamp)
- :func:`~pymaid.get_user_annotations`: returns list of annotations used by given user(s)

Treenodes:

- :func:`~pymaid.get_treenode_table`: retrieve treenode table for given neurons
- :func:`~pymaid.get_treenode_info`: retrieve info (i.e. skeleton ID) for a set of treenodes
- :func:`~pymaid.get_skid_from_treenode`: returns the skeleton which a treenode belongs to
- :func:`~pymaid.get_node_user_details`: returns details (creator, edition time, etc.) of individual nodes

Tags:

- :func:`~pymaid.get_label_list`: returns list of all treenode labels (tags) in the project
- :func:`~pymaid.add_tags`: add tags of treenodes or connectors
- :func:`~pymaid.delete_tags`: delete tags of treenodes or connectors
- :func:`~pymaid.get_node_tags`: returns tags of a set of treenodes or connectors

Connectivity:

- :func:`~pymaid.get_connectors`: returns connectors (synapses, abutting and/or gap junctions) for set of neurons
- :func:`~pymaid.get_connector_details`: returns details for connector (i.e. all neurons connected to it)
- :func:`~pymaid.get_connectors_between`: returns connectors connecting two sets of neurons
- :func:`~pymaid.get_edges`: returns edges (connections) between sets of neurons
- :func:`~pymaid.get_partners`: returns connectivity table for a set of neurons
- :func:`~pymaid.get_partners_in_volume`: retrieve connected partners for a list of neurons within a given Catmaid volume
- :func:`~pymaid.get_paths`: returns possible paths between two sets of neurons

User stats:

- :func:`~pymaid.get_user_list`: returns list of users in the project
- :func:`~pymaid.get_history`: retrieve project history similar to the project statistics widget
- :func:`~pymaid.get_time_invested`: calculate the time users have spent working on a set of neurons
- :func:`~pymaid.get_user_contributions`: returns contributions per user for a set of neurons
- :func:`~pymaid.get_contributor_statistics`: returns contributors (nodes, synapses, etc) for a set of neurons
- :func:`~pymaid.get_logs`: returns what the log widged shows (merges, splits, etc.)

Volumes:

- :func:`~pymaid.get_volume`: returns volume (verts + faces) of CATMAID volumes

Misc: 

- :func:`~pymaid.url_to_coordinates`: generate urls to coordinates
- :func:`~pymaid.get_review`: returns review status for set of neurons
- :func:`~pymaid.get_review_details`: returns review status (reviewer + timestamp) for each individual node


Higher functions
****************

Graph utils:

- :func:`~pymaid.matrix2graph`: generates iGraph representation from adjacency matrix
- :func:`~pymaid.network2graph`: generates iGraph representation from set of neurons
- :func:`~pymaid.neuron2nx`: generates NetworkX representation of neuron morphology
- :func:`~pymaid.neuron2igraph`: generates iGraph representation of neuron morphology
- :func:`~pymaid.distal_to`: use this to check spatial relation of nodes within a neuron
- :func:`~pymaid.dist_between`: point-to-point geodesic (along the arbor) distance
- :func:`~geodesic_matrix`: all-by-all geodesic distance matrix
- :func:`~pymaid.reroot_neuron`: reroot neuron to a specific node
- :func:`~pymaid.longest_neurite`: prunes neuron to its longest neurite
- :func:`~pymaid.cut_neuron`: cut neuron at a node or node tag
- :func:`~pymaid.classify_nodes`: adds a new column to a neuron's dataframe categorizing each node as branch, slab, leaf or root

Resampling:

- :func:`~pymaid.resample_neuron`: resample neuron to given resolution
- :func:`~pymaid.downsample_neuron`: takes skeleton data and reduces the number of nodes while preserving synapses, branch points, etc.

Functions to plot neurons:

- :func:`~pymaid.plot2d`: generates 2D plots of neurons
- :func:`~pymaid.plot3d`: uses either `Vispy <http://vispy.org>`_ or `Plotly <http://plot.ly>`_ to generate 3D plots of neurons
- :func:`~pymaid.plot_network`: uses iGraph and `Plotly <http://plot.ly>`_ to generate network plots
- :func:`~pymaid.clear3d`: clear 3D canvas
- :func:`~pymaid.close3d`: close 3D canvas and wipe from memory
- :func:`~pymaid.screenshot`: save screenshot

Connectivity tools:

- :func:`~pymaid.adjacency_matrix`: create a Pandas dataframe of adjacency matrix for two sets of neurons
- :func:`~pymaid.group_matrix`: groups matrix by columns or rows - use to e.g. collapse connectivity matrix into groups of neurons
- :func:`~pymaid.calc_overlap`: calculate cable between pairs of neurons that is within given distance
- :func:`~pymaid.filter_connectivity`: filter connectivity based on volumes or pruned neurons
- :func:`~pymaid.predict_connectivity`: predict connectivity based on potential connections

Functions for clustering:

- :func:`~pymaid.cluster_by_connectivity`: returns distance matrix based on connectivity similarity (Jarrell et al., 2012)
- :func:`~pymaid_xyz`: cluster points (synapses, nodes) based on eucledian distance
- :func:`~pymaid.cluster_by_synapse_placement`: hierarchical clustering of neurons based on synapse placement

Functions for morphological analyses:

- :func:`~pymaid.arbor_confidence`: calculates confidence along the arbor
- :func:`~pymaid.calc_cable`: calculate cable length of given neuron
- :func:`~pymaid.calc_segregation_index`: calculate segregation index (polarity) based on Schneider-Mizell et al., 2016
- :func:`~pymaid.calc_strahler_index`: calculate strahler index for each node
- :func:`~pymaid.calc_bending_flow`: variation of synapse flow centrality
- :func:`~pymaid.calc_flow_centrality`: implementation of synapse flow centrality algorithm by Schneider-Mizell et al. (2016)
- :func:`~pymaid.in_volume`: test if points are within given CATMAID volume
- :func:`~pymaid.prune_by_strahler`: prunes the neuron by strahler index
- :func:`~pymaid.split_axon_dendrite`: split neuron into axon, dendrite and primary neurite 
- :func:`~pymaid.stitch_neurons`: merge multiple neurons/fragments

Interface with R (nat, rcatmaid, etc.):

- :func:`~pymaid.rmaid.init_rcatmaid`: initialize connection with Catmaid server in R
- :func:`~pymaid.rmaid.data2py`: wrapper to convert R data to Python 
- :func:`~pymaid.rmaid.nblast`: wrapper to nblast a set neurons against external database
- :func:`~pymaid.rmaid.nblast_allbyall`: wrapper to nblast a set of neurons against each other
- :func:`~pymaid.rmaid.neuron2py`: converts R neuron and neuronlist objects to Pymaid neurons
- :func:`~pymaid.rmaid.neuron2r`: converts Pymaid neuron and list of neurons to R neuron and neuronlist objects, respectively

