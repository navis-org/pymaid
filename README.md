pymaid
==================

Collection of Python 3 tools to interface with CATMAID servers

## Basic example:

from pymaid import CatmaidInstance, get_3D_skeleton

myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )
skeleton_data = get_3D_skeleton ( ['12345','67890'] , myInstance )

## Available wrappers:
Currently pymaid features a range of wrappers to conveniently fetch data from CATMAID servers.
Use help() to learn more about their function, parameters and usage.

- add_annotations
- get_3D_skeleton
- get_arbor
- get_edges
- get_connectors
- get_review
- get_neuron_annotation
- get_neurons_in_volume
- get_annotations_from_list
- get_contributor_statistics
- retrieve_annotation_id
- retrieve_skids_by_annotation
- retrieve_skeleton_list
- retrieve_history
- retrieve_partners
- retrieve_names
- retrieve_node_lists
- skid_exists

## License:
This code is under GNU GPL V3
