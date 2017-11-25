Morphological Analyses
**********************

This section should give you an impression of how to access and compute morphological properties of neurons.

Basic properties
================

Many basic parameters are readily accessible through attributes of :class:`~pymaid.CatmaidNeuron` or :class:`~pymaid.CatmaidNeuronList`

>>> import pymaid
>>> rm = pymaid.CatmaidInstance('server_url','user','pw','token')
>>> nl = pymaid.get_neurons('annotation:glomerulus DA1')
>>> # Access single attribute: e.g. cable lengths [um]
>>> nl.cable_length
>>> # .. or get a full summary as pandas DataFrame
>>> df = nl.summary()
>>> df.n_connectors.tolist()
[437, 394, 326, 307, 356, 483, 316, 960, 438, 408, 553, 335, 380, 316, 620]

Cutting, pruning, pasting
=========================

Pymaid let's you perform (virtual) surgery on neurons. :class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList` have also some thin wrappers for these functions (e.g. :func:`~pymaid.CatmaidNeuron.prune_by_strahler` is a wrapper for :func:`~pymaid.prune_by_strahler`).

Some examples continuing with above neuronlist ``nl``:

>>> import pymaid
>>> # Cut a neuron in two using either a treenode ID or (in this case) a node tag
>>> distal, proximal = pymaid.cut_neuron( nl[0], cut_node='SCHLEGEL_LH' )
>>> # Plot neuron fragments
>>> core.CatmaidNeuronList( [ distal, proximal ] ).plot3d()
>>> # Alternatively, we can also just prune bits off a neuron objects
>>> nl[0].prune_distal_to('SCHLEGEL_LH')
>>> nl[0].plot3d()
>>> # To undo, simply reload the neuron from server
>>> nl[0].reload()
>>> # These operations can also be performed on a collection of neurons
>>> nl.prune_distal_to('SCHLEGEL_LH')
>>> nl.plot3d(clear3d=True)
>>> # Again, let's undo
>>> nl.reload()
>>> # Something more sophisticated: pruning by strahler index
>>> nl.prune_by_strahler( to_prune = [1,2,3] )
>>> nl.plot3d(connectors=True,clear3d=True)

For morphological comparisons using NBLAST, see * :ref:`_rmaid_link`.

Documentation
=============

.. automodule:: pymaid
    :members: arbor_confidence, calc_bending_flow, calc_cable, calc_flow_centrality, calc_overlap, calc_segregation_index, calc_strahler_index, classify_nodes, cut_neuron, distal_to, dist_between, downsample_neuron, resample_neuron, filter_connectivity, in_volume, longest_neurite, prune_by_strahler, reroot_neuron, split_axon_dendrite, stitch_neurons, synapse_root_distances
    :undoc-members:
    :show-inheritance: