Fetching data from the server
*****************************
With minor exceptions, pymaid covers every API endpoint of your CATMAID server - i.e. you should be able to get all the data that your webbrowser has access to. Please see :ref:`_overview_link` for a list of available functions.

Neurons
-------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_neuron
    ~pymaid.delete_neuron
    ~pymaid.find_neurons
    ~pymaid.get_arbor
    ~pymaid.get_neurons_in_volume
    ~pymaid.get_neuron_list
    ~pymaid.get_skids_by_annotation
    ~pymaid.get_skids_by_name
    ~pymaid.rename_neurons
    ~pymaid.get_names

Annotations
-----------
.. autosummary::
    :toctree: generated/

    ~pymaid.add_annotations
    ~pymaid.get_annotations
    ~pymaid.get_annotation_details
    ~pymaid.get_user_annotations

Treenodes
----------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_treenode_table
    ~pymaid.get_treenode_info
    ~pymaid.get_skid_from_treenode
    ~pymaid.get_node_user_details

Tags
----
.. autosummary::
    :toctree: generated/

    ~pymaid.get_label_list
    ~pymaid.add_tags
    ~pymaid.delete_tags
    ~pymaid.get_node_tags

Connectivity
------------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_connectors
    ~pymaid.get_connector_details
    ~pymaid.get_connectors_between
    ~pymaid.get_edges
    ~pymaid.get_partners
    ~pymaid.get_partners_in_volume
    ~pymaid.get_paths

User stats
----------
.. autosummary::
    :toctree: generated/

    ~pymaid.get_user_list
    ~pymaid.get_history
    ~pymaid.get_time_invested
    ~pymaid.get_user_contributions
    ~pymaid.get_contributor_statistics
    ~pymaid.get_logs
    ~pymaid.get_transactions

Volumes
-------
.. autosummary::
    :toctree: generated/
    
    ~pymaid.get_volume

Misc 
----
.. autosummary::
    :toctree: generated/

    ~pymaid.url_to_coordinates
    ~pymaid.get_review
    ~pymaid.get_review_details
