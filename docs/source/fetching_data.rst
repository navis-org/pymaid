Fetching data from the server
*****************************
With minor exceptions, pymaid covers every API endpoint of your CATMAID server - i.e. you should be able to get all the data that your webbrowser has access to. Please see :ref:`_overview_link` for a list of available functions.

.. automodule:: pymaid.fetch

Neurons
-------
.. autosummary::
    :toctree: generated/

    get_neuron
    delete_neuron
    find_neurons
    get_arbor
    get_neurons_in_volume
    get_neuron_list
    get_skids_by_annotation
    get_skids_by_name
    rename_neurons
    get_names

Annotations
-----------
.. autosummary::
    :toctree: generated/

    add_annotations
    get_annotations
    get_annotation_details
    get_user_annotations

Treenodes
----------
.. autosummary::
    :toctree: generated/

    get_treenode_table
    get_treenode_info
    get_skid_from_treenode
    get_node_user_details

Tags
----
.. autosummary::
    :toctree: generated/
    
    get_label_list
    add_tags
    delete_tags
    get_node_tags

Connectivity
------------
.. autosummary::
    :toctree: generated/

    get_connectors
    get_connector_details
    get_connectors_between
    get_edges
    get_partners
    get_partners_in_volume
    get_paths

User stats
----------
.. autosummary::
    :toctree: generated/

    get_user_list
    get_history
    get_time_invested
    get_user_contributions
    get_contributor_statistics
    get_logs

Volumes
-------
.. autosummary::
    :toctree: generated/
    
    get_volume

Misc 
----
.. autosummary::
    :toctree: generated/

    url_to_coordinates
    get_review
    get_review_details
