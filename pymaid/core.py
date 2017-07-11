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

""" This module contains definitions for neuron classes
"""

import datetime
import logging
import pandas as pd
import numpy as np
import datetime
import random

from pymaid import igraph_catmaid, morpho, pymaid, plot   

class CatmaidNeuron:
    """ 
    Catmaid neuron object holding neuron data: nodes, connectors, name, etc.

    Notes
    -----
    CatmaidNeuron can be minimally constructed from just a skeleton ID. 
    Other parameters (nodes, connectors, neuron name, annotations, etc.) 
    will then be retrieved from the server 'on-demand'. 

    Ideally, a CatmaidNeuron is constructed from a pandas DataFrame (df)
    containing: df.nodes, df.connectors, df.skeleton_id, df.neuron_name, 
    df.tags

    Parameters
    ----------
    x :             data to construct neuron from
                    1. skeleton ID or
                    2. pandas DataFrame or Series from pymaid.get_3D_skeleton() or
                    3. CatmaidNeuron (will create a deep copy)

                    This will override other, redundant attributes
    remote_instance :   CatmaidInstance, optional
                        Storing this makes it more convenient to retrieve e.g. 
                        neuron annotations, review status, etc.
    project_id :        integer, optional 
                        Default = 1
    meta_data :         dict, optional
                        any additional data
    copy :              boolean, optional
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
    cable_length :      float
                        Cable length in micrometers [um]    

    Examples
    --------    
    >>> from pymaid.core import CatmaidNeuron
    >>> from pymaid.pymaid import CatmaidInstance
    >>> # Initialize a new neuron
    >>> n = CatmaidNeuron( 123456 ) 
    >>> # Initialize Catmaid connections
    >>> rm = CatmaidInstance(server_url, http_user, http_pw, token) 
    >>> #Add CatmaidInstance to the neuron for convenience    
    >>> n.remote_instance = rm 
    >>> # Retrieve node data from server on-demand
    >>> n.nodes 
    CatmaidNeuron - INFO - Retrieving skeleton data...
        treenode_id  parent_id  creator_id  x  y  z radius confidence
    0   ...
    ...
    >>> #Initialize with skeleton data
    >>> n = pymaid.get_3D_skeleton( 123456, remote_instance = rm )
    >>> # Get annotations from server
    >>> n.annotations
    [ 'annotation1', 'annotation2' ]
    >>> Force update of annotations
    >>> n.get_annotations()
    """

    def __init__(self, x, remote_instance=None, project_id=1, meta_data=None, copy=True):        
        self.logger = logging.getLogger('CatmaidNeuron')

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            #Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)        

        if isinstance( x, pd.DataFrame ) or isinstance( x, CatmaidNeuronList ):
            if x.shape[0] == 1:
                x = x.ix[0]
            else:                
                raise Exception('Unable to construct CatmaidNeuron from data containing multiple neurons.')

        #These will be overriden if x is a CatmaidNeuron
        self._remote_instance = remote_instance
        self._project_id = project_id
        self._meta_data = meta_data
        self.date_retrieved = datetime.datetime.now().isoformat()
        self._is_copy = copy

        if isinstance( x, pd.Series ) or isinstance( x, CatmaidNeuron ):            
            if 'type' not in x.nodes:
                x  = morpho.classify_nodes( x )

            if copy:               
                if isinstance(x, CatmaidNeuron):                                        
                    self.df = x.df.copy()                    
                else:                                  
                    self.df = x.copy()

                self.df.nodes = self.df.nodes.copy()
                self.df.connectors = self.df.connectors.copy()

                if self.df.igraph != None:
                    self.df.igraph = self.df.igraph.copy()

            self.skeleton_id = self.df.skeleton_id
            self.neuron_name = self.df.neuron_name

            self.nodes = self.df.nodes                  

            self.connectors = self.df.connectors            

            self.tags = self.df.tags

            if isinstance(x, CatmaidNeuron):
                self._remote_instance = x._remote_instance
                self._project_id = x._project_id
                self._meta_data = x._meta_data
                self.date_retrieved = x.date_retrieved

                if 'igraph' in x.__dict__:                    
                    self.igraph = x.igraph
                    if copy:
                        self.igraph = self.igraph.copy()                
        else:
            try: 
                int( x ) #Check if this is a skeleton ID
                self.skeleton_id = str(x) 
            except:                
                raise Exception('Unable to construct CatmaidNeuron from data provided: %s' % str(type(x)))

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
        elif key == 'df':
            self.get_skeleton()
            return self.df
        elif key == 'tags':
            self.get_skeleton()
            return self.tags
        elif key == 'n_branch_nodes':
            if 'nodes' in self.__dict__:                
                return self.nodes[ self.nodes.type == 'branch' ].shape[0]
            else:
                return 'NA'
        elif key == 'n_end_nodes':
            if 'nodes' in self.__dict__:                
                return self.nodes[ self.nodes.type == 'end' ].shape[0]
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
            return morpho.calc_cable(self)
        else:            
            raise AttributeError('Attribute %s not found' % key)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def copy(self):
        """Create a copy of the neuron"""
        return CatmaidNeuron( self, copy = True )

    def get_skeleton(self, remote_instance = None, **kwargs ):
        """Get skeleton data for neuron using pymaid.get_3D_skeleton() 

        Parameters
        ----------       
        **kwargs
                    Will be passed to pymaid.get_3D_skeleton()
                    e.g. to get the full treenode history use:
                    n.get_skeleton( with_history = True )
                    or to get abutting connectors:
                    n.get_skeleton( get_abutting = True )
        """
        if not remote_instance and not self._remote_instance:
            raise Exception('Get_skeleton - Unable to connect to server without remote_instance. See help(core.CatmaidNeuron) to learn how to assign.')            
        elif not remote_instance:
            remote_instance = self._remote_instance
        self.logger.info('Retrieving skeleton data...')
        skeleton = pymaid.get_3D_skeleton( self.skeleton_id, remote_instance, self._project_id, return_neuron = False, kwargs = kwargs ).ix[0]   

        if 'type' not in skeleton.nodes:
            skeleton  = morpho.classify_nodes( skeleton )

        self.df = skeleton     
        self.nodes = skeleton.nodes
        self.connectors = skeleton.connectors
        self.tags = skeleton.tags
        self.neuron_name = skeleton.neuron_name        
        self.get_igraph()

        self.date_retrieved = datetime.datetime.now().isoformat()
        return

    def get_igraph( self ):
        """Calculate igraph representation of neuron """
        level = igraph_catmaid.module_logger.level
        igraph_catmaid.module_logger.setLevel('WARNING')
        self.igraph = igraph_catmaid.igraph_from_skeleton( self.df )
        igraph_catmaid.module_logger.setLevel(level)
        return self.igraph

    def get_review(self, remote_instance = None ):
        """Get review status for neuron"""
        if not remote_instance and not self._remote_instance:
            self.logger.error('Get_review: Unable to connect to server. Please provide CatmaidInstance as <remote_instance>.')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance
        self.review_status = pymaid.get_review( self.skeleton_id, remote_instance, self._project_id ).ix[0].percent_reviewed
        return self.review_status    

    def get_annotations(self, remote_instance = None ):
        """Retrieve annotations for neuron"""
        if not remote_instance and not self._remote_instance:
            self.logger.error('Get_annotations: Need CatmaidInstance to retrieve annotations. Use neuron.get_annotations( remote_instance = CatmaidInstance )')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        self.annotations = pymaid.get_annotations_from_list( self.skeleton_id, remote_instance , self._project_id )[ str(self.skeleton_id) ]
        return self.annotations

    def plot2d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot2d()   

        Parameters
        ----------     
        **kwargs         
                Will be passed to plot.plot2d() 
                See help(plot.plot3d) for a list of keywords                        
        """         
        return plot.plot2d( skdata = self, **kwargs )

    def plot3d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot3d()  

        Parameters
        ----------      
        **kwargs
                Will be passed to plot.plot3d() 
                See help(plot.plot3d) for a list of keywords                        
        """         
        return plot.plot3d( skdata = CatmaidNeuronList(self), **kwargs )

    def get_name(self, remote_instance = None):
        """Retrieve name of neuron"""
        if not remote_instance and not self._remote_instance:
            self.logger.error('Get_name: Need CatmaidInstance to retrieve annotations. Use neuron.get_annotations( remote_instance = CatmaidInstance )')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        self.neuron_name = pymaid.get_names( self.skeleton_id, remote_instance , self._project_id )[ str(self.skeleton_id) ]
        return self.neuron_name

    def downsample(self, factor = 5):        
        """Downsample the neuron by given factor 

        Parameters
        ----------
        factor :    int, optional
                    Factor by which to downsample the neurons. Default = 5
        """
        morpho.downsample_neuron ( self, factor, inplace=True)
        self.get_igraph()

    def reroot(self, new_root):        
        """Downsample the neuron by given factor 

        Parameters
        ----------
        new_root :  {int, str}
                    Either treenode ID or node tag
        """
        morpho.reroot_neuron( self, new_root, inplace=True)
        self.get_igraph()

    def update(self, remote_instance = None ):
        """Reload neuron from server. 

        Notes
        -----
        Currently only updates name, nodes, connectors and tags.
        """
        if not remote_instance and not self._remote_instance:
            self.logger.error('Get_update: Unable to connect to server. Please provide CatmaidInstance as <remote_instance>.')
        elif not remote_instance:
            remote_instance = self._remote_instance

        n = pymaid.get_3D_skeleton( self.skeleton_id, remote_instance = remote_instance )
        self.__init__(n, self._remote_instance, self._project_id, self._meta_data)

    def __str__(self):        
        return self.__repr__()

    def __repr__(self):
        #Look up these values without requesting them
        neuron_name = self.__dict__.get('neuron_name', 'NA')
        igraph = self.__dict__.get('igraph', None)
        tags = self.__dict__.get('tags', None)
        review_status = self.__dict__.get('review_status', 'NA')
        annotations = self.__dict__.get('annotations', None)
        if self.n_nodes == 'NA':
            cable = 'NA'
        else:
            cable = self.cable_length

        self._repr = pd.Series( [ type(self), neuron_name, self.skeleton_id, self.n_nodes, self.n_connectors, self.n_branch_nodes, self.n_end_nodes, cable, review_status, annotations != None, igraph != None, tags != None, self._remote_instance != None ],
                                 index = [ 'type','neuron_name', 'skeleton_id', 'n_nodes', 'n_connectors', 'n_branch_nodes', 'n_end_nodes', 'cable_length', 'review_status', 'annotations', 'igraph', 'tags', 'remote_instance' ]
                                )
        return str( self._repr )    


class CatmaidNeuronList:
    """ Catmaid neuron list. It is designed to work in many ways much like a 
    pandas DataFrame by supporting e.g. .ix[], .itertuples(), .empty, .copy() 

    Notes
    -----
    CatmaidNeuronList can be minimally constructed from just skeleton IDs. 
    Other parameters (nodes, connectors, neuron name, annotations, etc.) 
    will then be retrieved from the server 'on-demand'. 

    Ideally, a CatmaidNeuron is constructed from a pandas DataFrame (df)
    containing: df.nodes, df.connectors, df.skeleton_id, df.neuron_name, 
    df.tags for a set of neurons.

    Parameters
    ----------
    x :                 data to construct neuron from
                        1. skeleton ID or
                        2. pandas DataFrame or Series from `pymaid.get_3D_skeleton()` or
                        3. CatmaidNeuron (will create a deep copy)

                        This will override other, redundant attributes
    remote_instance :   CatmaidInstance, optional
                        Storing this makes it more convenient to retrieve e.g. 
                        neuron annotations, review status, etc.
    project_id :        integer, optional 
                        Default = 1
    meta_data :         dict, optional
                        Any additional data
    copy :              boolean, optional
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
    cable_length :      list of float
                        Cable length in micrometers [um]

    Examples
    --------
    >>> #Initialize with just a Skeleton ID 
    >>> nl = CatmaidNeuronList( [ 123456, 45677 ] )
    >>> #Add CatmaidInstance to neurons in neuronlist
    >>> rm = CatmaidInstance(server_url, http_user, http_pw, token)
    >>> nl.assign_remote_instance( rm )
    >>> #Retrieve review status from server on-demand
    >>> nl.review_status
    array([ 90, 10 ])
    >>> #Initialize with skeleton data
    >>> nl = pymaid.get_3D_skeleton( [ 123456, 45677 ], remote_instance = rm )
    >>> #Get annotations from server
    >>> nl.annotations
    [ ['annotation1','annotation2'],['annotation3','annotation4'] ]
    >>>Index using node count
    >>> subset = nl [ nl.n_nodes > 6000 ]
    >>> Index by skeleton ID 
    >>> subset = nl [ '123456' ]
    >>> #Index by neuron name
    >>> subset = nl [ 'name1' ]
    >>> #Concatenate lists
    >>> nl += pymaid.get_3D_skeleton( [ 912345 ], remote_instance = rm )

    """

    def __init__(self, x, remote_instance=None, project_id=1, copy=True ):
        self.logger = logging.getLogger('CatmaidNeuronList')
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            #Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

        if not isinstance(x, list) and not isinstance( x, pd.DataFrame ) and not isinstance(x, CatmaidNeuronList) and not isinstance(x, np.ndarray):            
            self.neurons = list([ x ])
        elif isinstance(x, pd.DataFrame):            
            self.neurons = [ x.ix[i] for i in range( x.shape[0] ) ]
        elif isinstance(x, CatmaidNeuronList):
            self.neurons = x.neurons
        else:            
            self.neurons = list(x) #We have to convert from numpy ndarray to list - do NOT remove list() here

        #Now convert into CatmaidNeurons if necessary
        for i,n in enumerate(self.neurons):            
            if not isinstance(n, CatmaidNeuron) or copy is True:
                self.neurons[i] = CatmaidNeuron( n, remote_instance = remote_instance, project_id = project_id, copy = copy )

        self.ix = _IXIndexer(self.neurons, self.logger)                   

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        #Prepare data without rquesting it from server
        d = []
        for n in self.neurons:
            neuron_name = n.__dict__.get('neuron_name', 'NA')
            igraph = n.__dict__.get('igraph', None)
            tags = n.__dict__.get('tags', None)
            review_status = n.__dict__.get('review_status', 'NA')
            annotations = n.__dict__.get('annotations', None)
            
            if n.n_nodes == 'NA':
                cable = 'NA'
            else:
                cable = n.cable_length

            d.append( [ neuron_name, n.skeleton_id, n.n_nodes, n.n_connectors, n.n_branch_nodes, n.n_end_nodes, cable, review_status, annotations != None, igraph != None,tags != None, n._remote_instance != None] )

        self._repr = pd.DataFrame( data = d,
                                   columns = ['neuron_name','skeleton_id','n_nodes','n_connectors','n_branch_nodes','n_end_nodes','cable_length','review_status','annotations','igraph','tags', 'remote_instance' ]
                                )        
        return str( self._repr )

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
            return ( self.__len__(),  )
        elif key == 'n_nodes':
            return np.array ( [ n.n_nodes for n in self.neurons ] ) #don't save this as attribute so that it keep getting updated
        elif key == 'n_connectors':
            return np.array ( [ n.n_connectors for n in self.neurons ] )
        elif key == 'cable_length':
            return np.array ( [ n.cable_length for n in self.neurons ] )
        elif key == 'neuron_name':
            return np.array ( [ n.neuron_name for n in self.neurons ] )
        elif key == 'skeleton_id':
            return np.array ( [ n.skeleton_id for n in self.neurons ] )

        elif key == 'nodes':  
            self.get_skeletons(skip_existing = True)
            return pd.DataFrame ( [ n.nodes for n in self.neurons ] )
        elif key == 'connectors':
            self.get_skeletons(skip_existing = True)
            return pd.DataFrame ( [ n.connectors for n in self.neurons ] )
        elif key == 'tags':
            self.get_skeletons(skip_existing = True)
            return np.array ( [ n.tags for n in self.neurons ] )
        elif key == 'df':
            self.get_skeletons(skip_existing = True)
            return pd.DataFrame ( [ n.df for n in self.neurons ] )

        elif key == '_remote_instance':
            all_instances = [ n._remote_instance for n in self.neurons if n._remote_instance != None ]
            if len(set(all_instances)) > 1:
                self.logger.warning('Neurons are using multiple remote_instances! Returning first entry.')
            elif len(set(all_instances)) == 0:
                raise Exception('No remote_instance found. Use .assign_remote_instance(rm) to assign one to all neurons.')
            else:
                return all_instances[0]

        elif key == 'igraph':
            return np.array ( [ n.igraph for n in self.neurons ] )
        elif key == 'review_status':
            to_retrieve = [ n.skeleton_id for n in self.neurons if 'review_status' not in n.__dict__ ]
            if to_retrieve:
                re = pymaid.get_review( to_retrieve, remote_instance = self._remote_instance ).set_index('skeleton_id')
                for n in [ n for n in self.neurons if 'review_status' not in n.__dict__ ]:
                    n.review_status = re.ix[ str( n.skeleton_id ) ].percent_reviewed
            return np.array ( [ n.review_status for n in self.neurons ] )
        elif key == 'annotations':
            to_retrieve = [ n.skeleton_id for n in self.neurons if 'annotations' not in n.__dict__ ]
            if to_retrieve:
                re = pymaid.get_annotations_from_list( to_retrieve, remote_instance = self._remote_instance )
                for n in [ n for n in self.neurons if 'annotations' not in n.__dict__ ]:
                    n.annotations = re[ str( n.skeleton_id ) ]
            return np.array ( [ n.annotations for n in self.neurons ] )        
        elif key == 'empty':
            return len(self.neurons) == 0
        else:            
            raise AttributeError('Attribute %s not found' % key)

    def __contains__(self, x):  
        return x in self.neurons or str(x) in self.skeleton_id or x in self.neuron_name

    def __getitem__(self, key):              
        if isinstance(key, str):
            subset =  [ n for n in self.neurons if key in n.neuron_name or key in n.skeleton_id ]
        elif isinstance(key, list):
            if True in [ isinstance(k, str) for k in key ]:
                subset =  [ n for i, n in enumerate(self.neurons) if True in [ k in n.neuron_name for k in key ] or True in [ k in n.skeleton_id for k in key ] ]
            else:                
                subset =  [ self.neurons[i] for i in key ]
        elif isinstance(key, np.ndarray) and key.dtype == 'bool':                                    
            subset = [ n for i,n in enumerate(self.neurons) if key[i] == True ]
        else:
            subset = self.neurons[key]

        if isinstance(subset, CatmaidNeuron):
            return subset
        else:
            return CatmaidNeuronList( subset )    

    def sample(self, N=1):
        """Use to return random subset of neurons"""
        indices = list( range( len(self.neurons) ) )
        random.shuffle(indices)
        return [ n for i,n in enumerate( self.neurons ) if i in indices[:N] ]

    def downsample(self, factor = 5):
        """Downsamples all neurons by factor X
        
        Parameters
        ----------
        factor :    int, optional
                    Factor by which to downsample the neurons. Default = 5
        """
        for n in self.neurons:
            n.downsample( downsampling = factor )

    def get_skeletons(self, skip_existing = False):
        """Helper function to fill in/update skeleton data of neurons. 

        Notes
        -----
        Will change/update nodes, connectors, df, tags, date_retrieved and 
        neuron_name. Will also generate new igraph representation to match 
        nodes/connectors.
        """
        if skip_existing:
            to_update = [ n for n in self.neurons if 'nodes' not in n.__dict__ ]
        else:
            to_update = self.neurons

        if to_update:
            skdata = pymaid.get_3D_skeleton( [n.skeleton_id for n in to_update], remote_instance=self._remote_instance, return_neuron=False ).set_index('skeleton_id')        
            for n in to_update:
                n.df = skdata.ix[ str( n.skeleton_id ) ]

                if 'type' not in n.df:
                    n.df  = morpho.classify_nodes( n.df )

                n.nodes = skdata.ix[ str( n.skeleton_id ) ].nodes
                n.connectors = skdata.ix[ str( n.skeleton_id ) ].connectors
                n.tags = skdata.ix[ str( n.skeleton_id ) ].tags
                n.neuron_name = skdata.ix[ str( n.skeleton_id ) ].neuron_name
                n.neuron_name = skdata.ix[ str( n.skeleton_id ) ].neuron_name
                n.date_retrieved = datetime.datetime.now().isoformat()
                n.get_igraph()

    def assign_remote_instance(self, remote_instance):
        """Assign remote_instance to all neurons"""
        for n in self.neurons:
            n._remote_instance = remote_instance

    def plot3d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot3d()        

        Parameters
        ---------
        **kwargs
                will be passed to plot.plot3d() 
                see help(plot.plot3d) for a list of keywords                        
        """         
        return plot.plot3d( skdata = self, **kwargs )

    def plot2d(self, **kwargs):
        """Plot neuron using pymaid.plot.plot2d()        

        Parameters
        ---------
        **kwargs        
                will be passed to plot.plot2d() 
                see help(plot.plot3d) for a list of keywords                        
        """         
        return plot.plot2d( skdata = self, **kwargs )

    def __missing__(self, key):
        self.logger.error('No neuron matching the search critera.')
        raise AttributeError('No neuron matching the search critera.')

    def __add__(self, to_add):
        if isinstance(to_add, list ):
            if False not in [ isinstance(n, CatmaidNeuron) for n in to_add ]:
                return CatmaidNeuronList( list(set(self.neurons + to_add )) )
            else:
                return CatmaidNeuronList( list(set(self.neurons + [ CatmaidNeuron[n] for n in to_add ])) )
        elif isinstance(to_add, CatmaidNeuron):
            return CatmaidNeuronList( list(set(self.neurons + [ to_add ])))
        elif isinstance(to_add, CatmaidNeuronList):             
            return CatmaidNeuronList( list(set(self.neurons + to_add )))
        else:
            self.logger.error('Unable to add data of type %s.' % str(type(to_add)))

    def __sub__(self, to_sub):
        if isinstance(to_sub, str) or isinstance(to_sub, int):
            return CatmaidNeuronList( [n for n in self.neurons if n.skeleton_id != to_sub and n.neuron_name != to_sub ] )
        elif isinstance(to_sub, list ):
            return CatmaidNeuronList( [n for n in self.neurons if n.skeleton_id not in to_sub and n.neuron_name not in to_sub ] )
        elif isinstance(to_sub, CatmaidNeuron):
            return CatmaidNeuronList( [n for n in self.neurons if n != to_sub ] )            
        elif isinstance(to_sub, CatmaidNeuronList):             
            return CatmaidNeuronList( [n for n in self.neurons if n not in to_sub ] )
    
    def itertuples(self):
        """Helper class to mimic pandas DataFrame itertuples()"""
        return self.neurons

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def copy(self):
        """Return copy of this CatmaidNeuronList """
        return CatmaidNeuronList( self, copy = True )


class _IXIndexer():
    """ Location based indexer added to CatmaidNeuronList objects to allow
    indexing similar to pandas DataFrames using df.ix[0]. This is really 
    just a helper to allow code to operate on CatmaidNeuron the same way
    it would on DataFrames.
    """

    def __init__(self, obj, logger = None):
        self.obj = obj
        self.logger = logger

    def __getitem__(self, key):              
        if isinstance(key, int) or isinstance(key, slice):
            return self.obj[key]
        else:                        
            raise Exception('Unable to index non-integers.')

