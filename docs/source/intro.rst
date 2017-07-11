.. _example:

Introduction
************
This section will teach you the basics of how to use PyMaid. If you are impatient, check out the *Quickstart Guide* but I recommend having a look at the *Basics* too.

Quickstart Guide
================
At the beginning of each session, you have to initialise a :class:`pymaid.pymaid.CatmaidInstance` that holds your credentials for the Catmaid server. In most examples, this instance is assigned to a variable called ``remote_instance`` or just ``rm``. Here we are requesting a list of two neurons from the server:

>>> from pymaid.pymaid import CatmaidInstance, get_3D_skeleton
>>> #HTTP_USER AND HTTP_PASSWORD are only necessary if your server requires a 
... #http authentification
>>> remote_instance = CatmaidInstance(   'www.your.catmaid-server.org' , 
...                                 	'HTTP_USER' , 
...                                 	'HTTP_PASSWORD', 
...                                 	'TOKEN' )
>>> neuron_list = get_3D_skeleton ( ['12345','67890'] , remote_instance )
>>> neuron_list[0]
type              <class 'pymaid.core.CatmaidNeuron'>
neuron_name                PN glomerulus DA1 27296 BH
skeleton_id                                     27295
n_nodes                                          9924
n_connectors                                      437
n_branch_nodes                                    207
n_end_nodes                                       214
cable_length                                  1479.81
review_status                                      NA
annotations                                     False
igraph                                          False
tags                                             True
dtype: object
>>> #Note how some entries are ``False``: these are still empty. They will be retrieved/computed on-demand upon first *explicit* request

``neuron_list`` is an instance of the :class:`pymaid.core.CatmaidNeuronList` class and holds two neurons, both of which are of the :class:`pymaid.core.CatmaidNeuron` class. Check out their documentation for methods and attributes.

Plotting is easy and straight forward:

>>> neuron_list.plot3d()

This method simply calls :func:`pymaid.plot.plot3d` - check out the docs for which parameters you can pass along.

The Basics
==========
Neuron data is (in most cases) stored as :class:`pymaid.core.CatmaidNeuron`. Multiple :class:`pymaid.core.CatmaidNeuron` are grouped into :class:`pymaid.core.CatmaidNeuronList`. 

You can minimally create an neuron object with just its skeleton ID:

>>> from pymaid import core
>>> neuron = core.CatmaidNeuron( 123456 )

Attributes of this neuron will be retrieved from the server on-demand. For this, you will have to assign a :class:`pymaid.pymaid.CatmaidInstance` to that neuron:

>>> neuron.set_remote_instance( server_url = 'url', http_user = 'user', http_pw = 'pw', auth_token = 'token' ) 
>>> #Retrieve the name of the neuron on-demand
>>> neuron.neuron_name
>>> #You can also just pass an existing instance 
>>> neuron = core.CatmaidNeuron( 123456, remote_instance = rm )

Some functions already return partially completed neurons (e.g. :func:`pymaid.pymaid.get_3D_skeleton`)

>>> rm = pymaid.CatmaidInstance( 'server_url', 'http_user', 'http_pw', 'auth_token' )
>>> neuron = pymaid.pymaid.get_3D_skeleton( 123456, rm )


Advanced Stuff
==============

Connection to the server: CatmaidInstance 
-----------------------------------------
Instances of :class:`pymaid.pymaid.CatmaidInstance` can be either explicitly passed to functions:

>>> from pymaid import pymaid
>>> rm = pymaid.CatmaidInstance( 'server_url', 'http_user', 'http_pw', 'auth_token' )
>>> partners = pymaid.get_partners( [12345,67890], remote_instance = rm )

Alternatively, you can also define it module-wide for the duration of your session:

>>> from pymaid import pymaid
>>> pymaid.remote_instance = rm
>>> partners = pymaid.get_partners( [12345,67890] )

The project ID is part of the CatmaidInstance and defaults to 1. You can change this either when initialising or later on-the-go:

>>> rm = pymaid.CatmaidInstance( 'server_url', 'http_user', 'http_pw', 'auth_token', project_id = 2 )
>>> rm.project_id = 1

CatmaidNeuron and CatmaidNeuronList objects
-------------------------------------------

Accessing data
++++++++++++++

As laid out in the Quickstart, :class:`pymaid.core.CatmaidNeuron` can be initialised with just a skeleton ID and the rest will then be requested/calculated on-demand:

>>> from pymaid.core import CatmaidNeuron
>>> from pymaid.pymaid import CatmaidInstance
>>> # Initialize a new neuron
>>> n = CatmaidNeuron( 123456 ) 
>>> # Initialize Catmaid connections
>>> rm = CatmaidInstance(server_url, http_user, http_pw, token) 
>>> #Add CatmaidInstance to the neuron for convenience    
>>> n.remote_instance = rm 

To access any of the data stored in a CatmaidNeuron simply use:

>>> # Retrieve node data from server on-demand
>>> n.nodes 
CatmaidNeuron - INFO - Retrieving skeleton data...
    treenode_id  parent_id  creator_id  x  y  z radius confidence
0   ...

You might have noticed that nodes are stored as pandas.DataFrame. That allows some fancy indexing and processing.

Other data, such as annotations are stored as simple lists.

>>> n.annotations
[ 'annotation1', 'annotation2' ]

All this data is loaded once upon the first explicit request and then stored in the CatmaidNeuron object. You can force updates by using the ``get`` functions:

>>> n.get_annotations()
>>> n.annotations
[ 'annotation1', 'annotation2', 'new_annotation' ]

Attributes in :class:`pymaid.core.CatmaidNeuron` work much the same way but instead you will get that data for all neurons that are within that neuron list.

>>> nl = CatmaidNeuronList( [ 123456, 456789, 123455 ], remote_instance = rm ) 
>>> nl.skeleton_id
[ 123456, 456789, 123455 ]
>>> nl.review_status
[ 10, 99, 12 ]

Indexing CatmaidNeuronLists
+++++++++++++++++++++++++++

:class:`pymaid.core.CatmaidNeuron` is much like pandas DataFrames in that it allows some fancing indexing

>>> #Initialize with just a Skeleton ID 
>>> nl = CatmaidNeuronList( [ 123456, 45677 ] )
>>> #Add CatmaidInstance to neurons in neuronlist
>>> rm = CatmaidInstance(server_url, http_user, http_pw, token)
>>> nl.set_remote_instance( rm )
>>> Index using node count
>>> subset = nl [ nl.n_nodes > 6000 ]
>>> Index by skeleton ID 
>>> subset = nl [ '123456' ]
>>> #Index by neuron name
>>> subset = nl [ 'name1' ]
>>> #Concatenate lists
>>> nl += pymaid.get_3D_skeleton( [ 912345 ], remote_instance = rm )
>>> #Remove item
>>> subset = nl - [ 45677 ]