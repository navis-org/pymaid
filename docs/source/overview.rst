.. _overview_link:

Neuron objects
==============

Neurons and lists of neurons are represented by:

.. autosummary::
    :toctree: generated/

 	CatmaidNeuron
 	CatmaidNeuronList

They can be minimally initialized with just skeleton IDs but usually you would get them returned from :func:`~pymaid.get_neuron`.

Data (e.g. nodes, connectors, name, review status, annotation) are retrieved/calculated on-demand the first time they are **explicitly** requested. CatmaidNeuronList also allows indexing similar to pandas DataFrames (see examples).

Class attributes
----------------

This is a *selection* of class attributes:

- ``skeleton_id``: neurons' skeleton ID(s)
- ``neuron_name``: neurons' name(s)
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


Class methods
-------------

See :class:`~pymaid.CatmaidNeuron` or ``help(pymaid.CatmaidNeuron)`` for complete list.

.. autosummary::
    :toctree: generated/

	CatmaidNeuron.plot3d
	CatmaidNeuron.plot2d
	CatmaidNeuron.plot_dendrogram
	CatmaidNeuron.prune_by_strahler
	CatmaidNeuron.prune_by_volume
	CatmaidNeuron.prune_distal_to
	CatmaidNeuron.prune_proximal_to
	CatmaidNeuron.reroot
	CatmaidNeuron.reload
	CatmaidNeuron.summary
	CatmaidNeuron.resample
	CatmaidNeuron.downsample
	CatmaidNeuron.copy
	CatmaidNeuron.from_swc
	CatmaidNeuronList.to_json
	CatmaidNeuronList.from_json
