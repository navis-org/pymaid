Connectivity analyses
+++++++++++++++++++++

Getting connectivity information for a set of neurons is pretty straight forward. You can get connectivity tables, edges between neurons, connectors between neurons. Check out functions below and in "Fetching data from the 
server"!

Here, we will focus on more complex examples: restricting connectivity
to specific parts of a neuron (e.g. dendrite vs axon) or to a given volume
(e.g. the lateral horn).

In the first example, we will generate an adjacency matrix for neurons' axons
and dendrites separately:

>>> import pandas as pd
>>> import seaborn as sns
>>> import matplotlib.pyplot as plt
>>> # Get a set of neurons
>>> nl = pymaid.get_neurons('annnotation:type_16_candidates')
>>> # Split into axon dendrite by using a tag
>>> nl.reroot(nl.soma)
>>> nl_axon = nl.prune_proximal_to('axon', inplace=False)
>>> nl_dend = nl.prune_distal_to('axon', inplace=False)
>>> # Get a list of the downstream partners
>>> cn_table = pymaid.get_partners(nl)
>>> ds_partners = cn_table[ cn_table.relation == 'downstream' ]
>>> # Take the top 10 downstream partners
>>> top_ds = ds_partners.iloc[:10].skeleton_id.values
>>> # Generate separate adjacency matrices for axon and dendrites
>>> adj_axon = pymaid.adjacency_matrix(nl_axon, top_ds, use_connectors=True )
>>> adj_dend = pymaid.adjacency_matrix(nl_dend, top_ds, use_connectors=True )
>>> # Rename rows and merge dataframes
>>> adj_axon.index += '_axon'
>>> adj_dend.index += '_dendrite'
>>> adj_merged = pd.concat([adj_axon, adj_dend], axis=0)
>>> # Plot heatmap using seaborn
>>> ax = sns.heatmap(adj_merged)
>>> plt.show()

Following up on above example, we will next subset the connectivity table to connections in a given CATMAID volume:

>>> # Get a CATMAID volume
>>> vol = pymaid.get_volume('LH_R')
>>> cn_table_lh = pymaid.filter_connectivity(cn_table, vol)

Reference
=========

Graphs
------
.. autosummary::
    :toctree: generated/

    ~pymaid.neuron2nx
    ~pymaid.neuron2igraph
    ~pymaid.neuron2KDTree
    ~pymaid.network2nx
    ~pymaid.network2igraph

Predict connectivity
--------------------
.. autosummary::
    :toctree: generated/

	~pymaid.predict_connectivity

Matrices
--------
.. autosummary::
    :toctree: generated/

    ~pymaid.adjacency_matrix
    ~pymaid.group_matrix

Clustering
----------
.. autosummary::
    :toctree: generated/

    ~pymaid.cluster_by_connectivity
    ~pymaid.cluster_by_synapse_placement
    ~pymaid.ClustResults

Plotting
--------
.. autosummary::
    :toctree: generated/

    ~pymaid.plot_network

Filtering
---------
.. autosummary::
    :toctree: generated/

	~pymaid.filter_connectivity
