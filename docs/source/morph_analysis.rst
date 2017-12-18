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
array([1590, 1180, 1035, 1106, 1215, 1183, 1059, 1870, 1747, 1296,  836,
        836, 1117,  752, 1109,  756, 1128, 1382,  715])
>>> # .. or get a full summary as pandas DataFrame
>>> df = nl.summary()
>>> df.head()
                      neuron_name skeleton_id  n_nodes  n_connectors  \
0      PN glomerulus DA1 27296 BH       27295     9969           463   
1      PN glomerulus DA1 57312 LK       57311     4874           420   
2  PN glomerulus DA1 57324 LK JSL       57323     4584           433   
3      PN glomerulus DA1 57354 GA       57353     4873           317   
4      PN glomerulus DA1 57382 ML       57381     7727           358   
   n_branch_nodes  n_end_nodes  open_ends  cable_length review_status  soma  
0             211          218         58   1590.676253            NA  True  
1             156          163        105   1180.597489            NA  True  
2             120          127         59   1035.076853            NA  True  
3              90           95         53   1106.768757            NA  True  
4             153          162         71   1215.920594            NA  True


Rerooting, resampling, simplifying
==================================
Pymaid lets you perform (virtual) surgery on neurons. Many of the base functions are also accessible directly via :class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList` methods. E.g. :func:`pymaid.CatmaidNeuronList.resample` is simply calling :func:`pymaid.resample_neuron`.

Examples continue from above code.

>>> # Reroot a single neuron to its soma
>>> nl[0].soma
3005291
>>> nl[0].reroot(nl[0].soma)
>>> # You can also perform this operation on the entire CatmaidNeuronList
>>> nl.reroot( nl.soma )
>>> # If you work with large lists it may be a good idea to resample before e.g. plotting
>>> # Get simplified copies
>>> nl_simple = nl.downsample( 10, inplace=False )
>>> # This is equivalent to this
>>> nl_simple = pymaid.downsample_neuron( nl, 10 )
>>> # More elaborate: resample to given resolution (in nanometers)
>>> nl_res = nl.resample( 1000, inplace=False )


Cutting, pruning, pasting
=========================

Examples continue from above code.

>>> # Cut a neuron in two using either a treenode ID or (in this case) a node tag
>>> distal, proximal = pymaid.cut_neuron( nl[0], cut_node='SCHLEGEL_LH' )
>>> # Plot neuron fragments
>>> pymaid.plot3d([ distal, proximal ])
>>> # We can also just prune bits off a neuron objects
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

For morphological comparisons using NBLAST, see :ref:`_rmaid_link`.

Reference
=========

Manipulation
------------
.. autosummary::
    :toctree: generated/

	~pymaid.cut_neuron
	~pymaid.reroot_neuron
	~pymaid.stitch_neurons
	~pymaid.split_axon_dendrite
	~pymaid.longest_neurite
	~pymaid.prune_by_strahler

Resampling
----------
.. autosummary::
    :toctree: generated/

    ~pymaid.resample_neuron
    ~pymaid.downsample_neuron

Analysis
--------
.. autosummary::
    :toctree: generated/

    ~pymaid.arbor_confidence
    ~pymaid.calc_bending_flow
    ~pymaid.calc_cable
    ~pymaid.calc_flow_centrality
    ~pymaid.calc_segregation_index
    ~pymaid.calc_strahler_index
    ~pymaid.classify_nodes

Distances
---------
.. autosummary::
    :toctree: generated/

    ~pymaid.calc_overlap
    ~pymaid.geodesic_matrix
    ~pymaid.distal_to
    ~pymaid.dist_between

Intersection
------------
.. autosummary::
    :toctree: generated/

    ~pymaid.in_volume
