#    Copyright (C) 2017 Philipp Schlegel

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along

""" This module contains neuron and neuronlist classes returned and accepted
by many low-level functions within pymaid.
"""

import datetime
import logging
import pandas as pd
import numpy as np
import datetime
import random
import json
import os
from tqdm import tqdm
from copy import copy, deepcopy

from pymaid import igraph_catmaid, morpho, pymaid, plot


class CatmaidNeuron:
    """ 
    Catmaid neuron object holding neuron data: nodes, connectors, name, etc.

    Notes
    -----
    CatmaidNeuron can be minimally constructed from just a skeleton ID
    and a CatmaidInstance. Other parameters (nodes, connectors, neuron name, 
    annotations, etc.) will then be retrieved from the server 'on-demand'.

    The easiest way to construct a CatmaidNeuron is by using
    :func:`pymaid.pymaid.get_neuron`. 

    Manually, a complete CatmaidNeuron can be constructed from a pandas 
    DataFrame (df) containing: df.nodes, df.connectors, df.skeleton_id, 
    df.neuron_name, df.tags

    Parameters
    ----------
    x             
                    Data to construct neuron from:
                    1. skeleton ID or
                    2. pandas DataFrame or Series from pymaid.get_neuron() or
                    3. another CatmaidNeuron (will create a deep copy). This will override other, redundant attributes.                    
    remote_instance :   CatmaidInstance, optional
                        Storing this makes it more convenient to retrieve e.g. 
                        neuron annotations, review status, etc.
    meta_data :         dict, optional
                        any additional data
    make_copy :         boolean, optional
                        If true, DataFrames are copied [.copy()] before being 
                        assigned to the neuron object to prevent 
                        backpropagation of subsequent changes to the data. 
                        Default = True

    Attributes
    ----------
    skeleton_id :       str
                        This neurons skeleton ID
    neuron_name :       str
                        This neurons name
    nodes :             pandas DataFrame
                        Contains complete treenode table
    connectors :        pandas DataFrame
                        Contains complete connector table
    date_retrieved :    ``datetime`` object
                        Timestamp of data retrieval
    tags :              dict
                        Treenode tags
    annotations :       list
                        This neuron's annotations
    igraph :            ``iGraph`` object
                        iGraph representation of this neuron
    review_status :     int
                        This neuron's review status
    n_branch_nodes :    int
                        Number of branch nodes
    n_end_nodes :       int
                        Number of end nodes
    n_open_ends :       int
                        Number of open end nodes. Leaf nodes that are not 
                        tagged with either: 'ends', 'not a branch', 
                        'uncertain end', 'soma' or 'uncertain continuation'
    cable_length :      float
                        Cable length in micrometers [um]    
    slabs :             list of treenode IDs
    soma :              treenode_id of soma
                        Returns None if no soma or 'NA' if data not available
    root :              treenode_id of root

    Examples
    --------    
    >>> from pymaid.core import CatmaidNeuron
    >>> from pymaid.pymaid import CatmaidInstance
    >>> # Initialize a new neuron
    >>> n = CatmaidNeuron( 123456 ) 
    >>> # Initialize Catmaid connections
    >>> rm = CatmaidInstance(server_url, http_user, http_pw, token) 
    >>> # Add CatmaidInstance to the neuron for convenience    
    >>> n.remote_instance = rm 
    >>> # Retrieve node data from server on-demand
    >>> n.nodes 
    ... CatmaidNeuron - INFO - Retrieving skeleton data...
    ...    treenode_id  parent_id  creator_id  x  y  z radius confidence
    ... 0  ...
    ...
    >>> # Initialize with skeleton data
    >>> n = pymaid.get_neuron( 123456, remote_instance = rm )
    >>> # Get annotations from server
    >>> n.annotations
    ... [ 'annotation1', 'annotation2' ]
    >>> # Force update of annotations
    >>> n.get_annotations()
    """    

    def __init__(self, x, remote_instance=None, meta_data=None ):
        self.logger = logging.getLogger('CatmaidNeuron')

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            # Create formatter and add it to the handlers
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)        

        if isinstance(x, pd.DataFrame) or isinstance(x, CatmaidNeuronList):
            if x.shape[0] == 1:
                x = x.ix[0]
            else:
                raise Exception(
                    'Unable to construct CatmaidNeuron from data containing multiple neurons.')

        if not isinstance(x, str) and not isinstance(x, int) and not isinstance(x, pd.Series) and not isinstance(x, CatmaidNeuron):
            raise TypeError('Unable to construct CatmaidNeuron from data type %s' % str(type(x)))        

        if remote_instance is None:
            if 'remote_instance' in globals():
                remote_instance = globals()['remote_instance']

        # These will be overriden if x is a CatmaidNeuron
        self._remote_instance = remote_instance
        self._meta_data = meta_data
        self.date_retrieved = datetime.datetime.now().isoformat() 

        #Parameters for soma detection
        self.soma_detection_radius = 100
        #Soma tag - set to None if no tag needed
        self.soma_detection_tag = 'soma'       

        if isinstance(x, CatmaidNeuron) or isinstance(x, pd.Series):
            self.skeleton_id = copy( x.skeleton_id )

            if 'type' not in x.nodes:
                morpho.classify_nodes(x)

            self.nodes = copy( x.nodes )
            self.connectors = copy( x.connectors )
            
            self.neuron_name = copy(x.neuron_name)
            self.tags = copy(x.tags)

            if 'igraph' in x.__dict__:
                self.igraph = x.igraph.copy()

            if isinstance(x,CatmaidNeuron):
                self._remote_instance = x._remote_instance # Remote instance will not be copied!
                self._meta_data = copy(x._meta_data)
                self.date_retrieved = copy(x.date_retrieved)

                self.soma_detection_radius = copy(x.soma_detection_radius)                
                self.soma_detection_tag = copy(x.soma_detection_tag)
        else:
            try:
                int(x)  # Check if this is a skeleton ID
                self.skeleton_id = str(x)
            except:
                raise Exception(
                    'Unable to construct CatmaidNeuron from data provided: %s' % str(type(x)))

    def __getattr__(self, key):
        if key == 'igraph':
            return self.get_igraph()
        elif key == 'neuron_name':
            return self.get_name()
        elif key == 'annotations':
            return self.get_annotations()
        elif key == 'review_status':
            return self.get_review()
        elif key == 'nodes':
            self.get_skeleton()
            return self.nodes
        elif key == 'connectors':
            self.get_skeleton()
            return self.connectors
        elif key == 'slabs':
            self.get_slabs()
            return self.slabs
        elif key == 'soma':
            return self.get_soma()
        elif key == 'root':
            return self.get_root()
        elif key == 'tags':
            self.get_skeleton()
            return self.tags
        elif key == 'n_open_ends':
            if 'nodes' in self.__dict__:
                closed = self.tags.get('ends', []) + self.tags.get('uncertain end', []) + self.tags.get(
                    'uncertain continuation', []) + self.tags.get('not a branch', []) + self.tags.get('soma', [])
                return len([n for n in self.nodes[self.nodes.type == 'end'].treenode_id.tolist() if n not in closed])
            else:
                return 'NA'
        elif key == 'n_branch_nodes':
            if 'nodes' in self.__dict__:
                return self.nodes[self.nodes.type == 'branch'].shape[0]
            else:
                return 'NA'
        elif key == 'n_end_nodes':
            if 'nodes' in self.__dict__:
                return self.nodes[self.nodes.type == 'end'].shape[0]
            else:
                return 'NA'
        elif key == 'n_nodes':
            if 'nodes' in self.__dict__:
                return self.nodes.shape[0]
            else:
                return 'NA'
        elif key == 'n_connectors':
            if 'connectors' in self.__dict__:
                return self.connectors.shape[0]
            else:
                return 'NA'
        elif key == 'cable_length':
            if 'nodes' in self.__dict__:
                return morpho.calc_cable(self)
            else:
                return 'NA'
        else:
            raise AttributeError('Attribute %s not found' % key)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def copy(self):
        """Create a copy of the neuron"""
        return CatmaidNeuron(self)

    def get_skeleton(self, remote_instance=None, **kwargs):
        """Get skeleton data for neuron using pymaid.get_neuron() 

        Parameters
        ----------       
        **kwargs
                    Will be passed to pymaid.get_neuron()
                    e.g. to get the full treenode history use:
                    n.get_skeleton( with_history = True )
                    or to get abutting connectors:
                    n.get_skeleton( get_abutting = True )

        See Also
        --------
        :func:`pymaid.pymaid.get_neuron` 
                    Function called to get skeleton information
        """
        if not remote_instance and not self._remote_instance:
            raise Exception(
                'Get_skeleton - Unable to connect to server without remote_instance. See help(core.CatmaidNeuron) to learn how to assign.')
        elif not remote_instance:
            remote_instance = self._remote_instance
        self.logger.info('Retrieving skeleton data...')
        skeleton = pymaid.get_neuron(
            self.skeleton_id, remote_instance, return_df=True, kwargs=kwargs).ix[0]

        if 'type' not in skeleton.nodes:
            morpho.classify_nodes(skeleton)
        
        self.nodes = skeleton.nodes
        self.connectors = skeleton.connectors
        self.tags = skeleton.tags
        self.neuron_name = skeleton.neuron_name
        self.date_retrieved = datetime.datetime.now().isoformat()

        # Delete outdated attributes
        self._clear_temp_attr()
        return

    def _clear_temp_attr(self):
        """Clear temporary attributes"""
        try:
            delattr(self, "igraph")
        except:
            pass
        try:
            delattr(self, "slabs")
        except:
            pass

    def get_igraph(self):
        """Calculate igraph representation of neuron """
        level = igraph_catmaid.module_logger.level
        igraph_catmaid.module_logger.setLevel('WARNING')
        self.igraph = igraph_catmaid.neuron2graph(self)
        igraph_catmaid.module_logger.setLevel(level)
        return self.igraph

    def get_slabs(self):
        """Generate slabs from neuron"""
        self.slabs = morpho._generate_slabs(self)
        return self.slabs

    def get_soma(self):
        """Search for soma

        Notes
        -----
        Uses either a treenode tag or treenode radius or a combination of both
        to identify the soma. This is set in the class attributes 
        ``soma_detection_radius`` and ``soma_detection_tag``. The default
        values for these are::

            
                soma_detection_radius = 100 
                soma_detection_tag = 'soma'        
            

        Returns
        -------
        treenode_id
            Returns treenode ID if soma was found, None if no soma.
                
        """
        tn = self.nodes[self.nodes.radius > self.soma_detection_radius].treenode_id.tolist()

        if self.soma_detection_tag: 
            if self.soma_detection_tag not in self.tags:
                return None
            else:
                tn = [n for n in tn if n in self.tags[self.soma_detection_tag]]

        if len(tn) == 1:
            return tn[0]
        elif len(tn) == 0:
            return None

        module_logger.warning('Multiple possible somas found')
        return tn

    def get_root(self):
        """Thin wrapper to get root node"""
        return self.nodes[ self.nodes.parent_id.isnull() ].treenode_id.tolist()[0]

    def get_review(self, remote_instance=None):
        """Get review status for neuron"""
        if not remote_instance and not self._remote_instance:
            self.logger.error(
                'Get_review: Unable to connect to server. Please provide CatmaidInstance as <remote_instance>.')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance
        self.review_status = pymaid.get_review(self.skeleton_id, remote_instance).ix[
            0].percent_reviewed
        return self.review_status

    def get_annotations(self, remote_instance=None):
        """Retrieve annotations for neuron"""
        if not remote_instance and not self._remote_instance:
            self.logger.error(
                'Get_annotations: Need CatmaidInstance to retrieve annotations. Use neuron.get_annotations( remote_instance = CatmaidInstance )')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        self.annotations = pymaid.get_annotations(
            self.skeleton_id, remote_instance)[str(self.skeleton_id)]
        return self.annotations

    def plot2d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot2d()   

        Parameters
        ----------     
        **kwargs         
                Will be passed to plot.plot2d() 
                See help(plot.plot3d) for a list of keywords  

        See Also
        --------
        :func:`pymaid.plot.plot2d` 
                    Function called to generate 2d plot
        """
        if 'nodes' not in self.__dict__:
            self.get_skeleton()
        return plot.plot2d(skdata=self, **kwargs)

    def plot3d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot3d()  

        Parameters
        ----------      
        **kwargs
                Will be passed to plot.plot3d() 
                See help(plot.plot3d) for a list of keywords      

        See Also
        --------
        :func:`pymaid.plot.plot3d` 
                    Function called to generate 3d plot   

        Examples
        --------
        >>> nl = pymaid.get_neuron('annotation:uPN right')
        >>> #Plot with connectors
        >>> nl.plot3d( connectors=True )               
        """

        if 'remote_instance' not in kwargs:
            kwargs.update({'remote_instance':self._remote_instance})

        if 'nodes' not in self.__dict__:
            self.get_skeleton()
        return plot.plot3d(skdata=CatmaidNeuronList(self), **kwargs)

    def get_name(self, remote_instance=None):
        """Retrieve name of neuron"""
        if not remote_instance and not self._remote_instance:
            self.logger.error(
                'Get_name: Need CatmaidInstance to retrieve annotations. Use neuron.get_annotations( remote_instance = CatmaidInstance )')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        self.neuron_name = pymaid.get_names(self.skeleton_id, remote_instance)[
            str(self.skeleton_id)]
        return self.neuron_name

    def downsample(self, factor=5, inplace=True):
        """Downsample the neuron by given factor 

        Parameters
        ----------
        factor :    int, optional
                    Factor by which to downsample the neurons. Default = 5
        """
        if not inplace:
            nl_copy = self.copy()
            nl_copy.downsample(factor=factor)
            return nl_copy

        morpho.downsample_neuron(self, factor, inplace=True)        

        # Delete outdated attributes
        self._clear_temp_attr()

    def reroot(self, new_root):
        """Downsample the neuron by given factor 

        Parameters
        ----------
        new_root :  {int, str}
                    Either treenode ID or node tag
        """
        morpho.reroot_neuron(self, new_root, inplace=True)

        # Clear temporary attributes
        self._clear_temp_attr()

    def prune_distal_to(self, node):
        """Cut off nodes distal to given node. 

        Parameters
        ----------
        node :      {treenode_id, node_tag}
                    Provide either a treenode ID or a (unique) tag
        """
        dist, prox = morpho.cut_neuron(self, node)
        self.__init__(prox, self._remote_instance, self._meta_data)

        # Clear temporary attributes
        self._clear_temp_attr()

    def prune_proximal_to(self, node):
        """Remove nodes proximal to given node. Reroots neuron to cut node.

        Parameters
        ----------
        node :      {treenode_id, node tag}
                    Provide either a treenode ID or a (unique) tag
        """
        dist, prox = morpho.cut_neuron(self, node)
        self.__init__(dist, self._remote_instance, self._meta_data)

        # Clear temporary attributes
        self._clear_temp_attr()

    def prune_by_strahler(self, to_prune=range(1, 2)):
        """ Prune neuron based on strahler order. Will reroot neuron to
        soma if possible.

        Notes
        -----
        Calls :func:`pymaid.morpho.prune_by_strahler`

        Parameters
        ----------
        to_prune :      {int, list, range}, optional
                        Strahler indices to prune. 
                        1. ``to_prune = 1`` removes all leaf branches
                        2. ``to_prune = [1,2]`` removes indices 1 and 2
                        3. ``to_prune = range(1,4)`` removes indices 1, 2 and 3  
                        4. ``to_prune = -1`` removes everything but the highest index 
        """
        morpho.prune_by_strahler(
            self, to_prune=to_prune, inplace=True, reroot_soma=True)

        # Clear temporary attributes
        self._clear_temp_attr()

    def reload(self, remote_instance=None):
        """Reload neuron from server. 

        Notes
        -----
        Currently only updates name, nodes, connectors and tags.
        """
        if not remote_instance and not self._remote_instance:
            self.logger.error(
                'Get_update: Unable to connect to server. Please provide CatmaidInstance as <remote_instance>.')
        elif not remote_instance:
            remote_instance = self._remote_instance

        n = pymaid.get_neuron(
            self.skeleton_id, remote_instance=remote_instance)
        self.__init__(n, self._remote_instance, self._meta_data)

        # Clear temporary attributes
        self._clear_temp_attr()

    def set_remote_instance(self, remote_instance=None, server_url=None, http_user=None, http_pw=None, auth_token=None):
        """Assign remote_instance to neuron

        Parameters
        ----------
        remote_instance :       pymaid.CatmaidInstance, optional
        server_url :            str, optional
        http_user :             str, optional
        http_pw :               str, optional
        auth_token :            str, optional

        Notes
        -----
        Provide either existing CatmaidInstance OR your credentials.

        """
        if remote_instance:
            self._remote_instance = remote_instance
        elif server_url and auth_token:
            self._remote_instance = pymaid.CatmaidInstance(server_url,
                                                           http_user,
                                                           http_pw,
                                                           auth_token
                                                           )
        else:
            raise Exception('Provide either CatmaidInstance or credentials.')

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.summary())

    def __add__(self, to_add):
        if isinstance(to_add, list):
            if False not in [isinstance(n, CatmaidNeuron) for n in to_add]:
                return CatmaidNeuronList(list(set([self] + to_add)))
            else:
                return CatmaidNeuronList(list(set([self] + [ CatmaidNeuron[n] for n in to_add ])))
        elif isinstance(to_add, CatmaidNeuron):
            return CatmaidNeuronList(list(set( [self] + [ to_add ])))
        elif isinstance(to_add, CatmaidNeuronList):
            return CatmaidNeuronList(list(set( [self] + to_add.neurons )))
        else:
            self.logger.error('Unable to add data of type %s.' %
                              str(type(to_add)))

    def summary(self):
        """ Get summary over all neurons in this NeuronList

        Parameters
        ----------
        n :     int, optional
                Get only first N entries

        Returns
        -------
        pandas Series                
        """

        # Look up these values without requesting them
        neuron_name = self.__dict__.get('neuron_name', 'NA')
        igraph = self.__dict__.get('igraph', 'NA')
        tags = self.__dict__.get('tags', 'NA')
        review_status = self.__dict__.get('review_status', 'NA')
        annotations = self.__dict__.get('annotations', 'NA')

        if 'nodes' in self.__dict__:
            soma_temp = self.soma
        else:
            soma_temp = 'NA'

        if tags != 'NA':
            tags = True

        if igraph != 'NA':
            igraph = True

        if annotations != 'NA':
            annotations = True

        return pd.Series([type(self), neuron_name, self.skeleton_id, self.n_nodes, self.n_connectors, self.n_branch_nodes, self.n_end_nodes, self.n_open_ends, self.cable_length, review_status, soma_temp, annotations, igraph, tags, self._remote_instance != None],
                         index=['type', 'neuron_name', 'skeleton_id', 'n_nodes', 'n_connectors', 'n_branch_nodes', 'n_end_nodes',
                                'n_open_ends', 'cable_length', 'review_status', 'soma', 'annotations', 'igraph', 'tags', 'remote_instance']
                         )


class CatmaidNeuronList:
    """ Catmaid neuron list. It is designed to work in many ways much like a 
    pandas DataFrame by supporting e.g. .ix[], .itertuples(), .empty, .copy() 

    Notes
    -----
    CatmaidNeuronList can be minimally constructed from just skeleton IDs. 
    Other parameters (nodes, connectors, neuron name, annotations, etc.) 
    will then be retrieved from the server 'on-demand'. 

    The easiest way to get a CatmaidNeuronList is by using 
    :func:`pymaid.pymaid.get_neuron` (see examples).

    Manually, a CatmaidNeuronList can constructed from a pandas DataFrame (df)
    containing: df.nodes, df.connectors, df.skeleton_id, df.neuron_name, 
    df.tags for a set of neurons.

    Parameters
    ----------
    x                 
                        Data to construct neuron from. Can be either:
                        1. skeleton ID or
                        2. pandas DataFrame or Series from `pymaid.get_neuron()` or
                        3. CatmaidNeuron (will create a deep copy). This will override other, redundant attributes
    remote_instance :   CatmaidInstance, optional
                        Storing this makes it more convenient to retrieve e.g. 
                        neuron annotations, review status, etc.
    meta_data :         dict, optional
                        Any additional data
    make_copy :         boolean, optional
                        If true, DataFrames are copied [.copy()] before being 
                        assigned to the neuron object to prevent 
                        backpropagation of subsequent changes to the data. 
                        Default = True   

    Attributes
    ----------
    skeleton_id :       list of str
                        Neurons' skeleton IDs
    neuron_name :       list of str
                        Neurons' names
    nodes :             list of pandas DataFrame
                        Neurons' complete treenode tables    
    connectors :        list of pandas DataFrame
                        Neurons' complete connector tables
    tags :              list of dict
                        Neurons' treenode tags
    annotations :       list of list
                        Neurons' annotations    
    review_status :     list of int
                        Neurons' review status
    n_branch_nodes :    list of int
                        Number of branch nodes  for each neuron
    n_end_nodes :       list of int
                        Number of end nodes for each neuron
    n_open_ends :       int
                        Number of open end nodes. Leaf nodes that are not 
                        tagged with either: 'ends', 'not a branch', 
                        'uncertain end', 'soma' or 'uncertain continuation'
    cable_length :      list of float
                        Cable length in micrometers [um]
    slabs :             list of treenode IDs
    soma :              list of soma nodes
    root :              list of root nodes

    Examples
    --------
    >>> # Initialize with just a Skeleton ID 
    >>> nl = CatmaidNeuronList( [ 123456, 45677 ] )
    >>> # Add CatmaidInstance to neurons in neuronlist
    >>> rm = CatmaidInstance(server_url, http_user, http_pw, token)
    >>> nl.set_remote_instance( rm )
    >>> # Retrieve review status from server on-demand
    >>> nl.review_status
    ... array([ 90, 10 ])
    >>> # Initialize with skeleton data
    >>> nl = pymaid.get_neuron( [ 123456, 45677 ], remote_instance = rm )
    >>> # Get annotations from server
    >>> nl.annotations
    ... [ ['annotation1','annotation2'],['annotation3','annotation4'] ]
    >>> Index using node count
    >>> subset = nl [ nl.n_nodes > 6000 ]
    >>> # Index by skeleton ID 
    >>> subset = nl [ '123456' ]
    >>> # Index by neuron name
    >>> subset = nl [ 'name1' ]
    >>> # Index using annotation
    >>> subset = nl ['annotation:uPN right']
    >>> # Concatenate lists
    >>> nl += pymaid.get_neuron( [ 912345 ], remote_instance = rm )

    """

    def __init__(self, x, remote_instance=None, make_copy=True):
        self.logger = logging.getLogger('CatmaidNeuronList')
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            # Create formatter and add it to the handlers
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

        if remote_instance is None:
            if 'remote_instance' in globals():
                remote_instance = globals()['remote_instance']

        if not isinstance(x, list) and not isinstance(x, pd.DataFrame) and not isinstance(x, CatmaidNeuronList) and not isinstance(x, np.ndarray):
            self.neurons = list([x])
        elif isinstance(x, pd.DataFrame):
            self.neurons = [ x.ix[i] for i in range(x.shape[0]) ]
        elif isinstance(x, CatmaidNeuronList):
            self.neurons = [ n for n in x.neurons ] # This has to be made a copy otherwise changes in the list will backpropagate
        else:
            # We have to convert from numpy ndarray to list - do NOT remove
            # list() here
            self.neurons = list(x)

            if True in [x.count(n) > 1 for n in x]:
                self.logger.warning(
                    'Multiple occurrences of the same neuron(s) were removed.')
                self.neurons = list(set(self.neurons))

        # Now convert into CatmaidNeurons if necessary
        for i, n in enumerate(self.neurons):
            if not isinstance(n, CatmaidNeuron) or make_copy is True:                
                self.neurons[i] = CatmaidNeuron(
                    n, remote_instance=remote_instance)            

        # Add indexer class
        self.ix = _IXIndexer(self.neurons, self.logger)

    def __str__(self):
        return self.__repr__()

    def summary(self, n=None):
        """ Get summary over all neurons in this NeuronList

        Parameters
        ----------
        n :     int, optional
                Get only first N entries

        Returns
        -------
        pandas DataFrame  

        """
        d = []
        for n in self.neurons[:None]:
            neuron_name = n.__dict__.get('neuron_name', 'NA')
            igraph = n.__dict__.get('igraph', 'NA')
            tags = n.__dict__.get('tags', 'NA')
            review_status = n.__dict__.get('review_status', 'NA')
            annotations = n.__dict__.get('annotations', 'NA')

            if tags != 'NA':
                tags = True

            if igraph != 'NA':
                igraph = True

            if annotations != 'NA':
                annotations = True

            if 'nodes' in n.__dict__:
                soma_temp = n.soma != None
            else:
                soma_temp = 'NA'

            d.append([neuron_name, n.skeleton_id, n.n_nodes, n.n_connectors, n.n_branch_nodes, n.n_end_nodes, n.n_open_ends,
                      n.cable_length, review_status, soma_temp, annotations, igraph, tags, n._remote_instance != None])

        return pd.DataFrame(data=d,
                            columns=['neuron_name', 'skeleton_id', 'n_nodes', 'n_connectors', 'n_branch_nodes', 'n_end_nodes', 'open_ends',
                                     'cable_length', 'review_status', 'soma', 'annotations', 'igraph', 'node_tags', 'remote_instance']
                            )

    def __repr__(self):
        return str(self.summary())

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter >= len(self.neurons):
            raise StopIteration
        to_return = self.neurons[self.iter]
        self.iter += 1
        return to_return

    def __len__(self):
        return len(self.neurons)

    def __getattr__(self, key):
        if key == 'shape':
            return (self.__len__(),)
        elif key == 'n_nodes':
            self.get_skeletons(skip_existing=True)
            # don't save this as attribute so that it keep getting updated
            return np.array([n.n_nodes for n in self.neurons])
        elif key == 'n_connectors':
            self.get_skeletons(skip_existing=True)
            return np.array([n.n_connectors for n in self.neurons])
        elif key == 'n_open_ends':
            self.get_skeletons(skip_existing=True)
            return np.array([n.n_open_ends for n in self.neurons])
        elif key == 'cable_length':
            self.get_skeletons(skip_existing=True)
            return np.array([n.cable_length for n in self.neurons])
        elif key == 'neuron_name':
            self.get_names(skip_existing=True)
            return np.array([n.neuron_name for n in self.neurons])
        elif key == 'skeleton_id':
            return np.array([n.skeleton_id for n in self.neurons])
        elif key == 'slabs':
            return np.array([n.slabs for n in self.neurons])
        elif key == 'soma':
            return np.array([n.soma for n in self.neurons])
        elif key == 'root':
            return np.array([n.root for n in self.neurons])
        elif key == 'nodes':
            self.get_skeletons(skip_existing=True)
            return pd.DataFrame([n.nodes for n in self.neurons])
        elif key == 'connectors':
            self.get_skeletons(skip_existing=True)
            return pd.DataFrame([n.connectors for n in self.neurons])
        elif key == 'tags':
            self.get_skeletons(skip_existing=True)
            return np.array([n.tags for n in self.neurons])        

        elif key == '_remote_instance':
            all_instances = [
                n._remote_instance for n in self.neurons if n._remote_instance != None]
            if len(set(all_instances)) > 1:
                self.logger.warning(
                    'Neurons are using multiple remote_instances! Returning first entry.')
            elif len(set(all_instances)) == 0:
                raise Exception(
                    'No remote_instance found. Use .set_remote_instance() to assign one to all neurons.')
            else:
                return all_instances[0]

        elif key == 'igraph':
            self.get_skeletons(skip_existing=True)
            return np.array([n.igraph for n in self.neurons])
        elif key == 'review_status':
            self.get_review(skip_existing=True)
            return np.array([n.review_status for n in self.neurons])
        elif key == 'annotations':
            to_retrieve = [
                n.skeleton_id for n in self.neurons if 'annotations' not in n.__dict__]
            if to_retrieve:
                re = pymaid.get_annotations(
                    to_retrieve, remote_instance=self._remote_instance)
                for n in [n for n in self.neurons if 'annotations' not in n.__dict__]:
                    n.annotations = re[str(n.skeleton_id)]
            return np.array([n.annotations for n in self.neurons])
        elif key == 'empty':
            return len(self.neurons) == 0
        else:
            raise AttributeError('Attribute %s not found' % key)

    def __contains__(self, x):
        return x in self.neurons or str(x) in self.skeleton_id or x in self.neuron_name

    def __getitem__(self, key):
        if isinstance(key, str):
            if key.startswith('annotation:'):
                skids = pymaid.eval_skids(
                    key, remote_instance=self._remote_instance)
                subset = self[skids]
            else:
                subset = [
                    n for n in self.neurons if key in n.neuron_name or key in n.skeleton_id]
        elif isinstance(key, list):
            if True in [isinstance(k, str) for k in key]:
                subset = [n for i, n in enumerate(self.neurons) if True in [
                    k in n.neuron_name for k in key] or True in [k in n.skeleton_id for k in key]]
            else:
                subset = [self.neurons[i] for i in key]
        elif isinstance(key, np.ndarray) and key.dtype == 'bool':
            subset = [n for i, n in enumerate(self.neurons) if key[i]]
        else:
            subset = self.neurons[key]

        if isinstance(subset, CatmaidNeuron):
            return subset
        
        return CatmaidNeuronList(subset)

    def sum(self):
        """Returns sum numeric and boolean values over all neurons"""
        return self.summary().sum(numeric_only=True)

    def mean(self):
        """Returns mean numeric and boolean values over all neurons"""
        return self.summary().mean(numeric_only=True)

    def sample(self, N=1):
        """Use to return random subset of neurons"""
        indices = list(range(len(self.neurons)))
        random.shuffle(indices)
        return CatmaidNeuronList( [n for i, n in enumerate(self.neurons) if i in indices[:N]] )

    def downsample(self, factor=5, inplace=True):
        """Downsamples all neurons by factor X

        Parameters
        ----------
        factor :    int, optional
                    Factor by which to downsample the neurons. Default = 5

        """
        if not inplace:
            nl_copy = self.copy()
            nl_copy.downsample(factor=factor)
            return nl_copy

        _set_loggers('ERROR')
        for n in tqdm(self.neurons, desc='Downsampling'):
            n.downsample(factor=factor)
        _set_loggers('INFO')

    def prune_distal_to(self, tag):
        """Cut off nodes distal to given node. 

        Parameters
        ----------
        node :      node tag
                    A (unique) tag at which to cut the neurons

        """
        _set_loggers('ERROR')
        for n in tqdm(self.neurons, desc='Pruning'):
            try:
                n.prune_distal_to(tag)
            except:
                pass
        _set_loggers('INFO')

    def prune_proximal_to(self, tag):
        """Remove nodes proximal to given node. Reroots neurons to cut node.

        Parameters
        ----------
        node :      node tag
                    A (unique) tag at which to cut the neurons

        """

        _set_loggers('ERROR')
        for n in tqdm(self.neurons, desc='Pruning'):
            try:
                n.prune_proximal_to(tag)
            except:
                pass
        _set_loggers('INFO')

    def prune_by_strahler(self, to_prune=range(1, 2)):
        """ Prune neurons based on strahler order. Will reroot neurons to
        soma if possible.

        Parameters
        ----------
        to_prune :      {int, list, range}, optional
                        Strahler indices to prune. 
                        1. ``to_prune = 1`` removes all leaf branches
                        2. ``to_prune = [1,2]`` removes indices 1 and 2
                        3. ``to_prune = range(1,4)`` removes indices 1, 2 and 3  
                        4. ``to_prune = -1`` removes everything but the highest index 

        See also
        --------
        :func:`pymaid.morpho.prune_by_strahler`
                        Function called to prune.

        """

        _set_loggers('ERROR')
        for n in tqdm(self.neurons, desc='Pruning'):
            n.prune_by_strahler(to_prune=to_prune)
        _set_loggers('INFO')

    def get_review(self, skip_existing=False):
        """ Use to get/update review status"""
        to_retrieve = [
            n.skeleton_id for n in self.neurons if 'review_status' not in n.__dict__]
        if to_retrieve:
            re = pymaid.get_review(
                to_retrieve, remote_instance=self._remote_instance).set_index('skeleton_id')
            for n in [n for n in self.neurons if 'review_status' not in n.__dict__]:
                n.review_status = re.ix[str(n.skeleton_id)].percent_reviewed

    def get_names(self, skip_existing=False):
        """ Use to get/update neuron names"""
        if skip_existing:
            to_update = [
                n.skeleton_id for n in self.neurons if 'neuron_name' not in n.__dict__]
        else:
            to_update = self.skeleton_id.tolist()

        if to_update:
            names = pymaid.get_names(
                self.skeleton_id, remote_instance=self._remote_instance)
            for n in self.neurons:
                try:
                    n.neuron_name = names[str(n.skeleton_id)]
                except:
                    pass

    def reload(self):
        """ Update neuron skeletons."""
        self.get_skeletons(skip_existing=False)

    def get_skeletons(self, skip_existing=False):
        """Helper function to fill in/update skeleton data of neurons. 

        Notes
        -----
        Will change/update nodes, connectors, df, tags, date_retrieved and 
        neuron_name. Will also generate new igraph representation to match 
        nodes/connectors.
        """
        if skip_existing:
            to_update = [n for n in self.neurons if 'nodes' not in n.__dict__]
        else:
            to_update = self.neurons

        if to_update:
            skdata = pymaid.get_neuron(
                [n.skeleton_id for n in to_update], remote_instance=self._remote_instance, return_df=True).set_index('skeleton_id')
            for n in tqdm(to_update, desc='Extracting data'):
                
                if 'type' not in skdata.ix[str(n.skeleton_id)].nodes:
                    morpho.classify_nodes( skdata.ix[ str(n.skeleton_id) ] )

                n.nodes = skdata.ix[str(n.skeleton_id)].nodes
                n.connectors = skdata.ix[str(n.skeleton_id)].connectors
                n.tags = skdata.ix[str(n.skeleton_id)].tags
                n.neuron_name = skdata.ix[str(n.skeleton_id)].neuron_name
                n.date_retrieved = datetime.datetime.now().isoformat()

                # Delete outdated attributes
                n._clear_temp_attr()

    def set_remote_instance(self, remote_instance=None, server_url=None, http_user=None, http_pw=None, auth_token=None):
        """Assign remote_instance to all neurons

        Parameters
        ----------
        remote_instance :       pymaid.CatmaidInstance, optional
        server_url :            str, optional
        http_user :             str, optional
        http_pw :               str, optional
        auth_token :            str, optional

        Notes
        -----
        Provide either existing CatmaidInstance OR your credentials.

        """


        if not remote_instance and server_url and auth_token:
            remote_instance = pymaid.CatmaidInstance(server_url,
                                                     http_user,
                                                     http_pw,
                                                     auth_token
                                                     )
        elif not remote_instance:
            raise Exception('Provide either CatmaidInstance or credentials.')

        for n in self.neurons:
            n._remote_instance = remote_instance

    def plot3d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot3d()        

        Parameters
        ---------
        **kwargs
                will be passed to plot.plot3d() 
                see help(plot.plot3d) for a list of keywords      

        See Also
        --------
        :func:`pymaid.plot.plot3d` 
                    Function called to generate 3d plot                  
        """

        if 'remote_instance' not in kwargs:
            kwargs.update({'remote_instance':self._remote_instance})

        self.get_skeletons(skip_existing=True)
        return plot.plot3d(skdata=self, **kwargs)

    def plot2d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot2d()        

        Parameters
        ---------
        **kwargs        
                will be passed to plot.plot2d() 
                see help(plot.plot3d) for a list of keywords  

        See Also
        --------
        :func:`pymaid.plot.plot2d` 
                    Function called to generate 2d plot                      
        """
        self.get_skeletons(skip_existing=True)
        return plot.plot2d(skdata=self, **kwargs)

    def to_json(self, fname = 'selection.json'):
        """ Saves neuron selection as json file which can be loaded
        in CATMAID selection table.

        Parameters
        ----------
        fname :     str, optional
                    Filename to save selection to
        """

        data = [  dict( skeleton_id = int(n.skeleton_id),
                        color = "#%02x%02x%02x" % (255, 255, 0),
                        opacity = 1
                        ) for n in self.neurons ]

        with open(fname, 'w') as outfile:
            json.dump(data, outfile)

        self.logger.error('Selection saved as %s in %s' % (fname, os.getcwd() ))

    def __missing__(self, key):
        self.logger.error('No neuron matching the search critera.')
        raise AttributeError('No neuron matching the search critera.')

    def __add__(self, to_add):
        if isinstance(to_add, list):
            if False not in [isinstance(n, CatmaidNeuron) for n in to_add]:
                return CatmaidNeuronList(list(set(self.neurons + to_add)))
            else:
                return CatmaidNeuronList(list(set(self.neurons + [CatmaidNeuron[n] for n in to_add])))
        elif isinstance(to_add, CatmaidNeuron):
            return CatmaidNeuronList(list(set(self.neurons + [to_add])))
        elif isinstance(to_add, CatmaidNeuronList):
            return CatmaidNeuronList(list(set(self.neurons + to_add.neurons)))
        else:
            self.logger.error('Unable to add data of type %s.' %
                              str(type(to_add)))

    def __sub__(self, to_sub):
        if isinstance(to_sub, str) or isinstance(to_sub, int):
            return CatmaidNeuronList([n for n in self.neurons if n.skeleton_id != to_sub and n.neuron_name != to_sub])
        elif isinstance(to_sub, list):
            return CatmaidNeuronList([n for n in self.neurons if n.skeleton_id not in to_sub and n.neuron_name not in to_sub])
        elif isinstance(to_sub, CatmaidNeuron):
            return CatmaidNeuronList([n for n in self.neurons if n != to_sub])
        elif isinstance(to_sub, CatmaidNeuronList):
            return CatmaidNeuronList([n for n in self.neurons if n not in to_sub])

    def itertuples(self):
        """Helper class to mimic pandas DataFrame itertuples()"""
        return self.neurons

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def copy(self):
        """Return copy of this CatmaidNeuronList """
        return CatmaidNeuronList(self, make_copy=True)

    def head(self, n=5):
        """Return summary for top N neurons"""
        return str(self.summary(n=n))


def _set_loggers(level='ERROR'):
    """Helper function to set levels for all associated module loggers """
    morpho.module_logger.setLevel(level)
    igraph_catmaid.module_logger.setLevel(level)
    plot.module_logger.setLevel(level)


class _IXIndexer():
    """ Location based indexer added to CatmaidNeuronList objects to allow
    indexing similar to pandas DataFrames using df.ix[0]. This is really 
    just a helper to allow code to operate on CatmaidNeuron the same way
    it would on DataFrames.
    """

    def __init__(self, obj, logger=None):
        self.obj = obj
        self.logger = logger

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.obj[key]
        else:
            raise Exception('Unable to index non-integers.')
