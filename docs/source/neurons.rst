.. _overview_link:

Neuron objects
==============

Neurons and lists of neurons are represented by:

.. autosummary::
    :toctree: generated/

 	~pymaid.CatmaidNeuron
 	~pymaid.CatmaidNeuronList

They can be minimally initialized with just skeleton IDs but usually you would get them returned from :func:`~pymaid.get_neuron`.

Data (e.g. nodes, connectors, name, review status, annotation) are retrieved/calculated on-demand the first time they are **explicitly** requested. CatmaidNeuronList also allows indexing similar to pandas DataFrames (see examples).

Class attributes
----------------

This is a *selection* of :class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList` class attributes:

- ``skeleton_id``: neurons' skeleton ID(s)
- ``neuron_name``: neurons' name(s)
- ``nodes``: treenode table
- ``connectors``: connector table
- ``partners``: connectivity table
- ``tags``: node tags (dict)
- ``annotations``: list of neurons' annotations
- ``cable_length``: cable length(s) in nm
- ``review_status``: review status of neuron(s)
- ``soma``: treenode ID of soma (if applicable)
- ``segments``: list of linear segments
- ``graph``: NetworkX graph representation of the neuron
- ``igraph``: iGraph representation of the neuron (if available)
- ``dps``: Dotproduct representation of the neuron


Class methods
-------------

See :class:`~pymaid.CatmaidNeuron` or ``help(pymaid.CatmaidNeuron)`` for complete list.

CatmaidNeuron/List methods
++++++++++++++++++++++++++
.. autosummary::
	:toctree: generated/

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
	pymaid.CatmaidNeuron.summary
	pymaid.CatmaidNeuron.resample
	pymaid.CatmaidNeuron.downsample
	pymaid.CatmaidNeuron.copy

CatmaidNeuron-specific
++++++++++++++++++++++
.. autosummary::
	:toctree: generated/

	pymaid.CatmaidNeuron.from_swc

CatmaidNeuronList-specific
+++++++++++++++++++++++++++
.. autosummary::
	:toctree: generated/

	pymaid.CatmaidNeuronList.to_json
	pymaid.CatmaidNeuronList.from_json
	pymaid.CatmaidNeuronList.has_annotation
