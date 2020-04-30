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
by many functions within pymaid. CatmaidNeuron and CatmaidNeuronList objects
also provided quick access to many other PyMaid functions.

Examples
--------
>>> # Get a bunch of neurons from CATMAID server as CatmaidNeuronList
>>> nl = pymaid.get_neuron('annotation:uPN right')
>>> # CatmaidNeuronLists work in, many ways, like pandas DataFrames
>>> nl.head()
                            neuron_name skeleton_id  n_nodes  n_connectors  \
0              PN glomerulus VA6 017 DB          16    12721          1878
1          PN glomerulus VL2a 22000 JMR       21999    10740          1687
2            PN glomerulus VC1 22133 BH       22132     8446          1664
3  PN putative glomerulus VC3m 22278 AA       22277     6228           674
4          PN glomerulus DL2v 22423 JMR       22422     4610           384

   n_branch_nodes  n_end_nodes  open_ends  cable_length review_status  soma
0             773          822        280   2863.743284            NA  True
1             505          537        194   2412.045343            NA  True
2             508          548        110   1977.235899            NA  True
3             232          243        100   1221.985849            NA  True
4             191          206         93   1172.948499            NA  True
>>> # Plot neurons
>>> nl.plot3d()
>>> # Neurons in a list can be accessed by index, ...
>>> nl[0]
>>> # ... by skeleton ID, ...
>>> nl.skid[16]
>>> # ... or by attributes
>>> nl[nl.cable_length > 2000]
>>> # Each neuron has a bunch of useful attributes
>>> print(nl[0].skeleton_id, nl[0].soma, nl[0].n_open_ends)
>>> # Attributes can also be accessed for the entire neuronslist
>>> nl.skeleton_id

:class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList`
also allow quick access to other PyMaid functions:

>>> # This ...
>>> pymaid.reroot_neuron(nl[0], nl[0].soma, inplace=True)
>>> # ... is essentially equivalent to this
>>> nl[0].reroot(nl[0].soma)
>>> # Similarly, CatmaidNeurons do on-demand data fetching for you:
>>> # So instead of this ...
>>> an = pymaid.get_annotations(nl[0])
>>> # ..., you can do just this:
>>> an = nl[0].annotations

"""

from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import datetime
import io
import json
import math
import multiprocessing as mp
import numbers
import os
import random
import sys
import six

import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.cluster.hierarchy

from . import (graph, morpho, fetch, graph_utils, resample, intersect,
               utils, config)

try:
    import trimesh
except ImportError:
    trimesh = None

__all__ = ['CatmaidNeuron', 'CatmaidNeuronList', 'Dotprops', 'Volume']

# Set up logging
logger = config.logger


class CatmaidNeuron:
    """ Catmaid neuron object holding neuron data (nodes, connectors, name,
    etc) and providing quick access to various PyMaid functions.

    CatmaidNeuron can be minimally constructed from just a skeleton ID
    and a CatmaidInstance. Other parameters (nodes, connectors, neuron name,
    annotations, etc.) will then be retrieved from the server 'on-demand'.

    The easiest way to construct a CatmaidNeuron is by using
    :func:`~pymaid.get_neuron`.

    Manually, a complete CatmaidNeuron can be constructed from a pandas
    DataFrame (df) containing: df.nodes, df.connectors, df.skeleton_id,
    df.neuron_name, df.tags

    Using a CatmaidNeuron to initialise a CatmaidNeuron will automatically
    make a copy.

    Attributes
    ----------
    skeleton_id :       str
                        This neuron's skeleton ID.
    neuron_name :       str
                        This neuron's name.
    nodes :             ``pandas.DataFrame``
                        Contains complete treenode table.
    connectors :        ``pandas.DataFrame``
                        Contains complete connector table.
    presynapses :       ``pandas.DataFrame``
                        All presynaptic connectors.
    postsynapses :      ``pandas.DataFrame``
                        All postsynaptic connectors.
    gap_junctions :     ``pandas.DataFrame``
                        All gap junction connectors.
    date_retrieved :    ``datetime`` object
                        Timestamp of data retrieval.
    tags :              dict
                        Treenode tags.
    annotations :       list
                        This neuron's annotations.
    graph :             ``network.DiGraph``
                        Graph representation of this neuron.
    igraph :            ``igraph.Graph``
                        iGraph representation of this neuron. Returns ``None``
                        if igraph library not installed.
    dps :               ``pandas.DataFrame``
                        Dotproduct representation of this neuron.
    review_status :     int
                        This neuron's review status.
    n_connectors :      int
                        Total number of synapses.
    n_presynapses :     int
                        Total number of presynaptic sites.
    n_postsynapses :    int
                        Total number of presynaptic sites.
    n_branch_nodes :    int
                        Number of branch nodes.
    n_end_nodes :       int
                        Number of end nodes.
    n_open_ends :       int
                        Number of open end nodes = leaf nodes that are not
                        tagged with either: ``ends``, ``not a branch``,
                        ``uncertain end``, ``soma`` or
                        ``uncertain continuation``.
    cable_length :      float
                        Cable length in micrometers [um].
    segments :          list of lists
                        Treenode IDs making up linear segments. Maximizes
                        segment lengths (similar to CATMAID's review widget).
    small_segments :    list of lists
                        Treenode IDs making up linear segments between
                        end/branch points.
    soma :              treenode ID of soma
                        Returns ``None`` if no soma or 'NA' if data not
                        available.
    root :              numpy.array
                        Treenode ID(s) of root.
    color :             tuple
                        Color of neuron. Used for e.g. export to json.
    partners :          pd.DataFrame
                        Connectivity table of this neuron.

    Examples
    --------
    >>> # Initialize a new neuron
    >>> n = pymaid.CatmaidNeuron(123456)
    >>> # Retrieve node data from server on-demand
    >>> n.nodes
    CatmaidNeuron - INFO - Retrieving skeleton data...
      treenode_id  parent_id  creator_id  x  y  z radius confidence
    0  ...
    >>> # Initialize with skeleton data
    >>> n = pymaid.get_neuron(123456)
    >>> # Get annotations from server
    >>> n.annotations
    ['annotation1', 'annotation2']
    >>> # Force update of annotations
    >>> n.get_annotations()

    """

    def __init__(self, x, remote_instance=None, meta_data=None):
        """ Initialize CatmaidNeuron.

        Parameters
        ----------
        x                   skeletonID | CatmaidNeuron
                            Data to construct neuron from.
        remote_instance :   CatmaidInstance, optional
                            Storing this makes it more convenient to retrieve
                            e.g. neuron annotations, review status, etc. If
                            not provided, will try using global CatmaidInstance.
        meta_data :         dict, optional
                            Any additional data to attach to neuron.
        """
        if isinstance(x, (pd.DataFrame, CatmaidNeuronList)):
            if x.shape[0] != 1:
                raise Exception('Unable to construct CatmaidNeuron from data '
                                'containing multiple neurons. Try '
                                'CatmaidNeuronList instead.')
            if isinstance(x, pd.DataFrame):
                x = x.iloc[0]
            else:
                x = x[0]

        if not isinstance(x, (str, int, np.int64, pd.Series, CatmaidNeuron)):
            raise TypeError('Unable to construct CatmaidNeuron from data '
                            'type %s' % str(type(x)))

        if remote_instance is None:
            if hasattr(x, 'remote_instance'):
                remote_instance = x.remote_instance
            else:
                remote_instance = utils._eval_remote_instance(None,
                                                              raise_error=False)

        # These will be overriden if x is a CatmaidNeuron
        self._remote_instance = remote_instance
        self.meta_data = meta_data
        self.date_retrieved = datetime.datetime.now().isoformat()

        # Parameters for soma detection
        self.soma_detection_radius = 500
        # Soma tag - set to None if no tag needed
        self.soma_detection_tag = 'soma'

        # Default color is yellow
        self.color = (255, 255, 0)

        if not isinstance(x, (CatmaidNeuron, pd.Series)):
            try:
                int(x)  # Check if this is a skeleton ID
                self.skeleton_id = str(x)  # Make sure skid is a string
            except BaseException:
                raise Exception('Unable to construct CatmaidNeuron from data '
                                'provided: %s' % str(type(x)))
        else:
            # If we are dealing with a DataFrame or a CatmaidNeuron get the essentials
            essential_attributes = ['skeleton_id']

            # If we have been passed a DataFrame/Series, just get the essentials
            if isinstance(x, pd.Series):
                essential_attributes += ['neuron_name', 'nodes', 'connectors',
                                         'tags']
            else:
                # Move all attributes if CatmaidNeuron
                for at in x.__dict__:
                    if at not in essential_attributes:
                        setattr(self, at, getattr(x, at))

            # Get essential attributes
            for at in essential_attributes:
                setattr(self, at, getattr(x, at))

        # Classify nodes if applicable
        if self.node_data and 'type' not in self.nodes:
            graph_utils.classify_nodes(self)

        # If a CatmaidNeuron is used to initialize, we need to make this
        # new object independent by copying important attributes
        if isinstance(x, CatmaidNeuron):
            for at in self.__dict__:
                try:
                    # Simple attributes don't have a .copy()
                    setattr(self, at, getattr(self, at).copy())
                except BaseException:
                    pass

    def __dir__(self):
        """ Custom __dir__ to add some parameters that we want to make
        searchable.
        """
        add_attributes = ['n_open_ends', 'n_branch_nodes', 'n_end_nodes',
                          'cable_length', 'root', 'neuron_name',
                          'nodes', 'annotations', 'partners', 'review_status',
                          'connectors', 'presynapses', 'postsynapses',
                          'gap_junctions', 'soma', 'root', 'tags',
                          'n_presynapses', 'n_postsynapses', 'n_connectors',
                          'bbox']

        return list(set(super().__dir__() + add_attributes))

    def __getattr__(self, key):
        # This is to catch empty neurons (e.g. after pruning)
        if key in ['n_open_ends', 'n_branch_nodes', 'n_end_nodes',
                   'cable_length'] and self.node_data and self.nodes.empty:
            return 0

        if key == 'igraph':
            return self.get_igraph()
        elif key == 'graph':
            return self.get_graph_nx()
        elif key == 'simple':
            self.simple = self.downsample(float('inf'),
                                          preserve_cn_treenodes=False,
                                          inplace=False)
            return self.simple
        elif key == 'dps':
            return self.get_dps()
        elif key == 'nodes_geodesic_distance_matrix':
            self.nodes_geodesic_distance_matrix = graph_utils.geodesic_matrix(self)
            return self.nodes_geodesic_distance_matrix
        elif key == 'neuron_name':
            return self.get_name()
        elif key == 'annotations':
            return self.get_annotations()
        elif key == 'partners':
            return self.get_partners()
        elif key == 'review_status':
            return self.get_review()
        elif key == 'nodes':
            self.get_skeleton()
            return self.nodes
        elif key == 'connectors':
            self.get_skeleton()
            return self.connectors
        elif key == 'presynapses':
            return self.connectors[self.connectors.relation == 0].reset_index()
        elif key == 'postsynapses':
            return self.connectors[self.connectors.relation == 1].reset_index()
        elif key == 'gap_junctions':
            return self.connectors[self.connectors.relation == 2].reset_index()
        elif key == 'segments':
            self._get_segments(how='length')
            return self.segments
        elif key == 'small_segments':
            self._get_segments(how='break')
            return self.small_segments
        elif key == 'soma':
            return self._get_soma()
        elif key == 'root':
            return self._get_root()
        elif key == 'tags':
            self.get_skeleton()
            return self.tags
        elif key == 'sampling_resolution':
            return self.n_nodes / self.cable_length
        elif key == 'n_open_ends':
            if self.node_data:
                closed = set(self.tags.get('ends', []) +
                             self.tags.get('uncertain end', []) +
                             self.tags.get('uncertain continuation', []) +
                             self.tags.get('not a branch', []) +
                             self.tags.get('soma', []))
                return len([n for n in self.nodes[self.nodes.type == 'end'].treenode_id.values if n not in closed])
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'n_branch_nodes':
            if self.node_data:
                return self.nodes[self.nodes.type == 'branch'].shape[0]
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'n_end_nodes':
            if self.node_data:
                return self.nodes[self.nodes.type == 'end'].shape[0]
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'n_nodes':
            if self.node_data:
                return self.nodes.shape[0]
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'n_connectors':
            if self.cn_data:
                return self.connectors.shape[0]
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'n_presynapses':
            if self.cn_data:
                return self.connectors[self.connectors.relation == 0].shape[0]
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'n_postsynapses':
            if self.cn_data:
                return self.connectors[self.connectors.relation == 1].shape[0]
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'cable_length':
            if self.node_data:
                # Simply sum up edge weight of all graph edges
                if self.igraph and config.use_igraph:
                    w = self.igraph.es.get_attribute_values('weight')
                else:
                    w = nx.get_edge_attributes(self.graph, 'weight').values()
                return sum(w) / 1000
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'bbox':
            if self.node_data:
                return self.nodes.describe().loc[['min', 'max'],
                                                 ['x', 'y', 'z']].values.T
            else:
                logger.info('No skeleton data available. Use .get_skeleton() '
                            'to fetch.')
                return 'NA'
        elif key == 'node_data':
            return 'nodes' in self.__dict__
        elif key == 'cn_data':
            return 'connectors' in self.__dict__
        elif key == 'n_skeletons':
            #len(list(nx.connected_components(self.graph.to_undirected())))
            return self.nodes[self.nodes.parent_id.isnull()].shape[0]
        else:
            raise AttributeError('Attribute "%s" not found' % key)

    def __copy__(self):
        return self.copy(deepcopy=False)

    def __deepcopy__(self):
        return self.copy(deepcopy=True)

    def copy(self, deepcopy=False):
        """Returns a copy of the neuron.

        Parameters
        ----------
        deepcopy :  bool, optional
                    If False, `.graph` (NetworkX DiGraph) will be returned as
                    view - changes to nodes/edges can progagate back!
                    ``.igraph`` (iGraph) - if available - will always be
                    deepcopied.

        """
        x = CatmaidNeuron(self.skeleton_id)
        x.__dict__ = {k: copy.copy(v) for k, v in self.__dict__.items()}

        # Remote instance is excluded from copy -> otherwise we are *silently*
        # creating a new CatmaidInstance that will be identical to the original
        # but will have it's own cache!
        x._remote_instance = self._remote_instance

        if 'graph' in self.__dict__:
            x.graph = self.graph.copy(as_view=deepcopy is not True)
        if 'igraph' in self.__dict__:
            if self.igraph is not None:
                # This is pretty cheap, so we will always make a deep copy
                x.igraph = self.igraph.copy()

        return x

    def get_skeleton(self, remote_instance=None, **fetch_kwargs):
        """Get/Update skeleton data for neuron.

        Parameters
        ----------
        **fetch_kwargs
                    Will be passed to :func:`pymaid.get_neuron` e.g. to get
                    the full treenode history use::

                        n.get_skeleton(with_history = True)

                    or to get abutting connectors::

                        n.get_skeleton(get_abutting = True)

        See Also
        --------
        :func:`~pymaid.get_neuron`
                    Function called to get skeleton information
        """
        if not remote_instance and not self._remote_instance:
            raise Exception('Get_skeleton - Unable to connect to server '
                            'without remote_instance. See '
                            'help(core.CatmaidNeuron) to learn how to '
                            'assign.')
        elif not remote_instance:
            remote_instance = self._remote_instance
        logger.info('Retrieving skeleton data...')
        skeleton = fetch.get_neuron(self.skeleton_id,
                                    remote_instance=remote_instance,
                                    return_df=True,
                                    fetch_kwargs=fetch_kwargs).iloc[0]

        self.nodes = skeleton.nodes
        self.connectors = skeleton.connectors
        self.tags = skeleton.tags
        self.neuron_name = skeleton.neuron_name
        self.date_retrieved = datetime.datetime.now().isoformat()

        # Delete outdated attributes
        self._clear_temp_attr()

        if 'type' not in self.nodes:
            graph_utils.classify_nodes(self)

        return

    def _clear_temp_attr(self, exclude=[]):
        """Clear temporary attributes."""
        temp_att = ['igraph', 'graph', 'segments', 'small_segments',
                    'nodes_geodesic_distance_matrix', 'dps', 'simple',
                    'centrality_method']
        for a in [at for at in temp_att if at not in exclude]:
            try:
                delattr(self, a)
                logger.debug('Neuron {}: {} cleared'.format(self.skeleton_id, a))
            except BaseException:
                logger.debug('Neuron {}: Unable to clear temporary attribute "{}"'.format(self.skeleton_id, a))
                pass

        temp_node_cols = ['flow_centrality', 'strahler_index']

        # Remove type only if we do not classify -> this speeds up things
        # b/c we don't have to recreate the column, just change the values
        # if 'classify_nodes' in exclude:
        #    temp_node_cols.append('type')

        # Remove temporary node values
        self.nodes = self.nodes[[
            c for c in self.nodes.columns if c not in temp_node_cols]]

        if 'classify_nodes' not in exclude:
            # Reclassify nodes
            graph_utils.classify_nodes(self, inplace=True)

    def get_graph_nx(self):
        """Calculates networkX representation of neuron.

        Once calculated stored as ``.graph``. Call function again to update
        graph.

        See Also
        --------
        :func:`pymaid.neuron2nx`
        """
        self.graph = graph.neuron2nx(self)
        return self.graph

    def get_igraph(self):
        """Calculates iGraph representation of neuron.

        Once calculated stored as ``.igraph``. Call function again to update
        iGraph.

        Important
        ---------
        Returns ``None`` if igraph is not installed!

        See Also
        --------
        :func:`pymaid.neuron2igraph`
        """
        self.igraph = graph.neuron2igraph(self)
        return self.igraph

    def get_dps(self):
        """Calculates/updates dotprops representation of the neuron.

        Once calculated stored as ``.dps``.

        See Also
        --------
        :func:`pymaid.to_dotprops`
        """

        self.dps = morpho.to_dotprops(self)
        return self.dps

    def _get_segments(self, how='length'):
        """Generate segments for neuron."""
        if how == 'length':
            self.segments = graph_utils._generate_segments(self)
        elif how == 'break':
            self.small_segments = graph_utils._break_segments(self)
        return self.segments

    def _get_soma(self):
        """Search for soma and return treenode ID of soma.

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
        tn = self.nodes[self.nodes.radius >
                        self.soma_detection_radius].treenode_id.values

        if self.soma_detection_tag:
            if self.soma_detection_tag not in self.tags:
                return None
            else:
                tn = [n for n in tn if n in self.tags[self.soma_detection_tag]]

        if len(tn) == 1:
            return tn[0]
        elif len(tn) == 0:
            return None

        logger.warning('Multiple somas found for neuron #{}'.format(self.skeleton_id))
        return tn

    def _get_root(self):
        """Thin wrapper to get root node(s)."""
        roots = self.nodes[self.nodes.parent_id.isnull()].treenode_id.values
        return roots

    def get_partners(self, remote_instance=None):
        """Get connectivity table for this neuron."""
        if not remote_instance and not self._remote_instance:
            logger.error('Get_partners: Unable to connect to server. Please '
                         'provide CatmaidInstance as <remote_instance>.')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        # Get partners
        self.partners = fetch.get_partners(self.skeleton_id,
                                           remote_instance=remote_instance)

        return self.partners

    def get_review(self, remote_instance=None):
        """Get review status for neuron."""
        if not remote_instance and not self._remote_instance:
            logger.error('Get_review: Unable to connect to server. Please '
                         'provide CatmaidInstance as <remote_instance>.')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance
        self.review_status = fetch.get_review(self.skeleton_id,
                                              remote_instance=remote_instance).loc[0, 'percent_reviewed']
        return self.review_status

    def get_annotations(self, remote_instance=None):
        """Retrieve annotations for neuron."""
        if not remote_instance and not self._remote_instance:
            logger.error('Get_annotations: Need CatmaidInstance to retrieve '
                         'annotations. Use neuron.get_annotations( '
                         'remote_instance = CatmaidInstance )')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        self.annotations = fetch.get_annotations(self.skeleton_id,
                                                 remote_instance=remote_instance).get(str(self.skeleton_id), [])
        return self.annotations

    def plot2d(self, **kwargs):
        """Plot neuron using :func:`pymaid.plot2d`.

        Parameters
        ----------
        **kwargs
                Will be passed to :func:`pymaid.plot2d`.
                See ``help(pymaid.plot2d)`` for a list of keywords.

        See Also
        --------
        :func:`pymaid.plot2d`
                    Function called to generate 2d plot.

        """

        from pymaid import plotting

        if 'nodes' not in self.__dict__:
            self.get_skeleton()
        return plotting.plot2d(self, **kwargs)

    def plot3d(self, **kwargs):
        """Plot neuron using :func:`pymaid.plot3d`.

        Parameters
        ----------
        **kwargs
                Keyword arguments. Will be passed to :func:`pymaid.plot3d`.
                See ``help(pymaid.plot3d)`` for a list of keywords.

        See Also
        --------
        :func:`pymaid.plot3d`
                    Function called to generate 3d plot.

        Examples
        --------
        >>> nl = pymaid.get_neuron('annotation:uPN right')
        >>> #Plot with connectors
        >>> nl.plot3d( connectors=True )

        """

        from pymaid import plotting

        if 'remote_instance' not in kwargs:
            kwargs.update({'remote_instance': self._remote_instance})

        if 'nodes' not in self.__dict__:
            self.get_skeleton()
        return plotting.plot3d(CatmaidNeuronList(self, make_copy=False),
                               **kwargs)

    def plot_dendrogram(self, linkage_kwargs={}, dend_kwargs={}):
        """ Plot neuron as dendrogram.

        Parameters
        ----------
        linkage_kwargs :    dict
                            Passed to ``scipy.cluster.hierarchy.linkage``.
        dend_kwargs :       dict
                            Passed to ``scipy.cluster.hierarchy.dendrogram``.

        Returns
        -------
        scipy.cluster.hierarchy.dendrogram
        """

        # First get the all by all distances
        ends = self.nodes[self.nodes.type == 'end'].treenode_id.values
        dist_mat = graph_utils.geodesic_matrix(self,
                                               tn_ids=ends,
                                               directed=False)

        # Prune matrix to ends
        dist_mat = dist_mat.loc[ends, ends]

        # Turn into observation vector
        obs_vec = scipy.spatial.distance.squareform(dist_mat.values,
                                                    checks=False)

        # Cluster
        linkage = scipy.cluster.hierarchy.linkage(obs_vec, **linkage_kwargs)

        # Plot
        return scipy.cluster.hierarchy.dendrogram(linkage, **dend_kwargs)

    def get_name(self, remote_instance=None):
        """Retrieve/update name of neuron."""
        if not remote_instance and not self._remote_instance:
            logger.error('Get_name: Need CatmaidInstance to retrieve '
                         'annotations. Use neuron.get_annotations( '
                         'remote_instance = CatmaidInstance )')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        self.neuron_name = fetch.get_names(self.skeleton_id,
                                           remote_instance=remote_instance)[str(self.skeleton_id)]
        return self.neuron_name

    def resample(self, resample_to, inplace=True):
        """Resample the neuron to given resolution [nm].

        Parameters
        ----------
        resample_to :           int
                                Resolution in nanometer to which to resample
                                the neuron.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.

        See Also
        --------
        :func:`~pymaid.resample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        resample.resample_neuron(x, resample_to, inplace=True)

        # No need to call this as base function does this for us
        # x._clear_temp_attr()

        if not inplace:
            return x

    def downsample(self, factor=5, inplace=True, **kwargs):
        """Downsample the neuron by given factor.

        Parameters
        ----------
        factor :                int, optional
                                Factor by which to downsample the neurons.
                                Default = 5.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.
        **kwargs
                                Additional arguments passed to
                                :func:`~pymaid.downsample_neuron`.

        See Also
        --------
        :func:`~pymaid.downsample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        resample.downsample_neuron(x, factor, inplace=True, **kwargs)

        # Delete outdated attributes
        x._clear_temp_attr()

        if not inplace:
            return x

    def reroot(self, new_root, inplace=True):
        """ Reroot neuron to given treenode ID or node tag.

        Parameters
        ----------
        new_root :  int | str
                    Either treenode ID or node tag.
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~pymaid.reroot_neuron`
            Base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        graph_utils.reroot_neuron(x, new_root, inplace=True)

        # Clear temporary attributes is done by morpho.reroot_neuron()
        # x._clear_temp_attr()

        if not inplace:
            return x

    def prune_distal_to(self, node, inplace=True):
        """Cut off nodes distal to given nodes.

        Parameters
        ----------
        node :      treenode_id | node_tag
                    Provide either treenode ID(s) or a unique tag(s)
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~pymaid.cut_neuron`
            Base function. See for details and examples.
        """

        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils._make_iterable(node, force_type=None)

        for n in node:
            prox = graph_utils.cut_neuron(x, n, ret='proximal')
            # Reinitialise with proximal data
            x.__init__(prox, x._remote_instance, x.meta_data)
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr()

        if not inplace:
            return x

    def prune_proximal_to(self, node, inplace=True):
        """Remove nodes proximal to given node. Reroots neuron to cut node.

        Parameters
        ----------
        node :      treenode_id | node tag
                    Provide either a treenode ID or a (unique) tag
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~pymaid.cut_neuron`
            Base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils._make_iterable(node, force_type=None)

        for n in node:
            dist = graph_utils.cut_neuron(x, n, ret='distal')
            # Reinitialise with distal data
            x.__init__(dist, x._remote_instance, x.meta_data)
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr()

        # Clear temporary attributes is done by cut_neuron
        # x._clear_temp_attr()

        if not inplace:
            return x

    def prune_by_strahler(self, to_prune, inplace=True):
        """ Prune neuron based on `Strahler order
        <https://en.wikipedia.org/wiki/Strahler_number>`_.

        Will reroot neuron to soma if possible.

        Parameters
        ----------
        to_prune :  int | list | range | slice
                    Strahler indices to prune. For example:

                    1. ``to_prune=1`` removes all leaf branches
                    2. ``to_prune=[1, 2]`` removes SI 1 and 2
                    3. ``to_prune=range(1, 4)`` removes SI 1, 2 and 3
                    4. ``to_prune=slice(1, -1)`` removes everything but the
                       highest SI
                    5. ``to_prune=slice(-1, None)`` removes only the highest
                       SI

        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~pymaid.prune_by_strahler`
            This is the base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy()

        morpho.prune_by_strahler(
            x, to_prune=to_prune, reroot_soma=True, inplace=True)

        # No need to call this as morpho.prune_by_strahler does this already
        # self._clear_temp_attr()

        if not inplace:
            return x

    def prune_by_longest_neurite(self, n=1, reroot_to_soma=False,
                                 inplace=True):
        """ Prune neuron down to the longest neurite.

        Parameters
        ----------
        n :                 int, optional
                            Number of longest neurites to preserve.
        reroot_to_soma :    bool, optional
                            If True, will reroot to soma before pruning.
        inplace :           bool, optional
                            If True, operation will be performed on itself.
                            If False, operation is performed on copy which is
                            then returned.

        See Also
        --------
        :func:`~pymaid.longest_neurite`
            This is the base function. See for details and examples.

        """

        if inplace:
            x = self
        else:
            x = self.copy()

        graph_utils.longest_neurite(
            x, n, inplace=True, reroot_to_soma=reroot_to_soma)

        # Clear temporary attributes
        x._clear_temp_attr()

        if not inplace:
            return x

    def prune_by_volume(self, v, mode='IN', prevent_fragments=False,
                        inplace=True):
        """ Prune neuron by intersection with given volume(s).

        Parameters
        ----------
        v :                 str | pymaid.Volume | list of either
                            Volume(s) to check for intersection
        mode :              'IN' | 'OUT', optional
                            If 'IN', parts of the neuron inside the volume are
                            kept.
        prevent_fragments : bool, optional
                            If True, will add nodes to ``subset`` required to
                            keep neuron from fragmenting.
        inplace :           bool, optional
                            If True, operation will be performed on itself. If
                            False, operation is performed on copy which is then
                            returned.

        See Also
        --------
        :func:`~pymaid.in_volume`
            Base function. See for details and examples.
        """

        if not isinstance(v, Volume):
            v = fetch.get_volume(v, combine_vols=True,
                                 remote_instance=self._remote_instance)

        if inplace:
            x = self
        else:
            x = self.copy()

        intersect.in_volume(x, v, inplace=True,
                            prevent_fragments=prevent_fragments,
                            remote_instance=self._remote_instance, mode=mode)

        # Clear temporary attributes
        # x._clear_temp_attr()

        if not inplace:
            return x

    def reload(self, remote_instance=None):
        """Reload neuron from server.

        Currently only updates name, nodes, connectors and tags, not e.g.
        annotations.

        """

        if not remote_instance and not self._remote_instance:
            logger.error('Get_update: Unable to connect to server. Please '
                         'provide CatmaidInstance as <remote_instance>.')
        elif not remote_instance:
            remote_instance = self._remote_instance

        n = fetch.get_neuron(
            self.skeleton_id, remote_instance=remote_instance)
        self.__init__(n, self._remote_instance, self.meta_data)

        # Clear temporary attributes
        self._clear_temp_attr()

    def set_remote_instance(self, remote_instance=None, server_url=None,
                            http_user=None, http_pw=None, auth_token=None):
        """Assign remote_instance to neuron.

        Provide either existing CatmaidInstance OR your credentials.

        Parameters
        ----------
        remote_instance :       pymaid.CatmaidInstance, optional
        server_url :            str, optional
        http_user :             str, optional
        http_pw :               str, optional
        auth_token :            str, optional

        See Also
        --------
        :class:`~pymaid.CatmaidInstance`

        """
        if remote_instance:
            self._remote_instance = remote_instance
        elif server_url and auth_token:
            self._remote_instance = fetch.CatmaidInstance(server_url,
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

    def _repr_html_(self):
        frame = self.summary().to_frame()
        frame.columns = ['']
        # return self._gen_svg_thumbnail() + frame._repr_html_()
        return frame._repr_html_()

    def _gen_svg_thumbnail(self):
        import matplotlib.pyplot as plt
        # Store some previous states
        prev_level = logger.getEffectiveLevel()
        prev_pbar = config.pbar_hide
        prev_int = plt.isinteractive()

        plt.ioff()  # turn off interactive mode
        logger.setLevel('WARNING')
        config.pbar_hide = True
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        fig, ax = self.plot2d(connectors=False, ax=ax)
        output = io.StringIO()
        fig.savefig(output, format='svg')

        if prev_int:
            plt.ion()  # turn on interactive mode
        logger.setLevel(prev_level)
        config.pbar_hide = prev_pbar
        _ = plt.clf()
        return output.getvalue()

    def __eq__(self, other):
        """Implements neuron comparison."""
        if isinstance(other, CatmaidNeuron):
            # We will do this sequentially and stop as soon as we find a
            # discrepancy -> this saves tons of time
            to_comp = ['skeleton_id', 'neuron_name']

            # Make some morphological comparisons if we have node data
            if self.node_data and other.node_data:
                # Make sure to go from simple to computationally expensive
                to_comp += ['n_nodes', 'n_connectors', 'soma', 'root',
                            'n_branch_nodes', 'n_end_nodes', 'n_open_ends',
                            'cable_length']

            for at in to_comp:
                comp = getattr(self, at) == getattr(other, at)
                if isinstance(comp, np.ndarray) and not all(comp):
                    return False
                elif comp is False:
                    return False
            # If all comparisons have passed, return True
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Generate a hashable value."""
        # We will simply use the neuron's memory address
        return id(self)

    def __add__(self, other):
        """ Implements addition. """
        if isinstance(other, CatmaidNeuron):
            return CatmaidNeuronList([self, other])
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Implements division for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] /= other
            n.connectors.loc[:, ['x', 'y', 'z']] /= other
            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implements multiplication for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] *= other
            n.connectors.loc[:, ['x', 'y', 'z']] *= other
            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        else:
            return NotImplemented

    def summary(self):
        """Get a summary of this neuron."""

        # Set logger to warning only - otherwise you might get tons of
        # "skeleton data not available" messages
        lvl = logger.level
        logger.setLevel('WARNING')

        # Look up these values without requesting them
        neuron_name = self.__dict__.get('neuron_name', 'NA')
        review_status = self.__dict__.get('review_status', 'NA')
        soma = self.soma if self.node_data else 'NA'

        s = pd.Series([type(self), neuron_name, self.skeleton_id,
                       self.n_nodes, self.n_connectors,
                       self.n_branch_nodes, self.n_end_nodes,
                       self.n_open_ends, self.cable_length,
                       review_status, soma],
                      index=['type', 'neuron_name', 'skeleton_id',
                             'n_nodes', 'n_connectors', 'n_branch_nodes',
                             'n_end_nodes', 'n_open_ends', 'cable_length',
                             'review_status', 'soma'])

        logger.setLevel(lvl)
        return s

    def to_dataframe(self):
        """ Turn this CatmaidNeuron into a pandas DataFrame containing
        only the original CATMAID data.

        Returns
        -------
        pandas DataFrame
                neuron_name  skeleton_id  nodes  connectors  tags
            0
        """

        return pd.DataFrame([[self.neuron_name, self.skeleton_id,
                              self.nodes, self.connectors, self.tags]],
                            columns=['neuron_name', 'skeleton_id', 'nodes',
                                     'connectors', 'tags'])

    def to_swc(self, filename=None, **kwargs):
        """ Generate SWC file from this neuron.

        This converts CATMAID nanometer coordinates into microns.

        Parameters
        ----------
        filename :      str | None, optional
                        If ``None``, will use "neuron_{skeletonID}.swc".
        kwargs
                        Additional arguments passed to :func:`~pymaid.to_swc`.

        Returns
        -------
        Nothing

        See Also
        --------
        :func:`~pymaid.to_swc`
                See this function for further details.

        """

        return utils.to_swc(self, filename, **kwargs)

    @classmethod
    def from_graph(self, g, **kwargs):
        """ Generate neuron object from NetworkX Graph.

        This function will try to generate a neuron-like tree structure from
        the Graph. Therefore the graph may not contain loops!

        Treenode coordinates (``x``, ``y``, ``z``) need to be properties of
        the graph's nodes.

        Parameters
        ----------
        g :         networkx.Graph | networkx.DiGraph
        **kwargs
                    Additional neuron parameters as keyword arguments.
                    For example, ``skeleton_id``, ``neuron_name``, etc.

        Returns
        -------
        core.CatmaidNeuron

        See Also
        --------
        pymaid.graph.nx2neuron
                    Base function with more parameters.
        """

        return graph.nx2neuron(g, **kwargs)

    @classmethod
    def from_swc(self, filename, neuron_name=None, neuron_id=None):
        """ Generate neuron object from SWC file.

        This import is following format specified `here
        <http://research.mssm.edu/cnic/swc.html>`_.

        Parameters
        ----------
        filename :      str
                        Name of SWC file.
        neuronname :    str, optional
                        Name to use for the neuron. If not provided, will use
                        filename
        neuron_id :     int, optional
                        Unique identifier (essentially skeleton ID). If not
                        provided, will generate one from scratch.

        Returns
        -------
        CatmaidNeuron

        See Also
        --------
        :func:`~pymaid.from_swc`
                See this function for further details.

        """
        return utils.from_swc(filename, neuron_name, neuron_id)


class CatmaidNeuronList:
    """ Compilation of :class:`~pymaid.CatmaidNeuron` that allow quick
    access to neurons' attributes/functions. They are designed to work in many
    ways much like a pandas.DataFrames by, for example, supporting ``.ix[ ]``,
    ``.itertuples()``, ``.empty`` or ``.copy()``.

    CatmaidNeuronList can be minimally constructed from just skeleton IDs.
    Other parameters (nodes, connectors, neuron name, annotations, etc.)
    will then be retrieved from the server 'on-demand'.

    The easiest way to get a CatmaidNeuronList is by using
    :func:`~pymaid.get_neuron` (see examples).

    Manually, a CatmaidNeuronList can constructed from a pandas DataFrame (df)
    consisting of: df.nodes, df.connectors, df.skeleton_id, df.neuron_name,
    df.tags for a set of neurons.

    Attributes
    ----------
    skeleton_id :       np.array of str
    neuron_name :       np.array of str
    nodes :             ``pandas.DataFrame``
                        Merged treenode table.
    connectors :        ``pandas.DataFrame``
                        Merged connector table. This also works for
                        `presynapses`, `postsynapses` and `gap_junctions`.
    tags :              np.array of dict
                        Treenode tags.
    annotations :       np.array of list
    partners :          pd.DataFrame
                        Connectivity table for these neurons.
    graph :             np.array of ``networkx`` graph objects
    igraph :            np.array of ``igraph`` graph objects
    review_status :     np.array of int
    n_connectors :      np.array of int
    n_presynapses :     np.array of int
    n_postsynapses :    np.array of int
    n_branch_nodes :    np.array of int
    n_end_nodes :       np.array of int
    n_open_ends :       np.array of int
    cable_length :      np.array of float
                        Cable lengths in micrometers [um].
    soma :              np.array of treenode_ids
    root :              np.array of treenode_ids
    n_cores :           int
                        Number of cores to use. Default ``os.cpu_count()-1``.
    _use_threading :    bool (default=True)
                        If True, will use parallel threads. Should be slightly
                        up to a lot faster depending on the numbers of cores.
                        Switch off if you experience performance issues.

    Examples
    --------
    >>> # Initialize with just a Skeleton ID
    >>> nl = pymaid.CatmaidNeuronList([123456, 45677])
    >>> # Retrieve review status from server on-demand
    >>> nl.review_status
    array([90, 23])
    >>> # Initialize with skeleton data
    >>> nl = pymaid.get_neuron([123456, 45677])
    >>> # Get annotations from server
    >>> nl.annotations
    [['annotation1', 'annotation2'], ['annotation3', 'annotation4']]
    >>> Index using node count
    >>> subset = nl[nl.n_nodes > 6000]
    >>> # Get neuron by its skeleton ID
    >>> n = nl.skid[123456]
    >>> # Index by multiple skeleton ID
    >>> subset = nl[['123456', '45677']]
    >>> # Index by neuron name
    >>> subset = nl['name1']
    >>> # Index using annotation
    >>> subset = nl['annotation:uPN right']
    >>> # Concatenate lists
    >>> nl += pymaid.get_neuron([912345])

    """

    def __init__(self, x, remote_instance=None, make_copy=False,
                 _use_parallel=False):
        """ Initialize CatmaidNeuronList.

        Parameters
        ----------
        x :                 int | list | array | CatmaidNeuron/List
                            Data to construct neuronlist from. Can be either:

                            1. skeleton ID(s)
                            2. CatmaidNeuron(s)
                            3. CatmaidNeuronList(s)
        remote_instance :   CatmaidInstance, optional
                            If not provided, will try extracting from input
                            or from global CatmaidInstance.
        make_copy :         boolean, optional
                            If True, CatmaidNeurons are deepcopied before
                            being assigned to the neuronlist.
        """

        # Set number of cores
        self.n_cores = max(1, os.cpu_count())

        # If below parameter is True, most calculations will be parallelized
        # which speeds them up quite a bit. Unfortunately, this uses A TON of
        # memory - for large lists this might make your system run out of
        # memory. In these cases, leave this property at False
        self._use_parallel = _use_parallel
        self._use_threading = True

        # Determines if subsetting this NeuronList will return copies
        self.copy_on_subset = False

        if remote_instance is None:
            try:
                remote_instance = x.remote_instance
            except BaseException:
                remote_instance = utils._eval_remote_instance(None,
                                                              raise_error=False)

        if not isinstance(x, (list, pd.DataFrame, CatmaidNeuronList,
                              np.ndarray)):
            self.neurons = list([x])
        elif isinstance(x, pd.DataFrame):
            self.neurons = [x.loc[i] for i in range(x.shape[0])]
        elif isinstance(x, CatmaidNeuronList):
            # This has to be made a copy otherwise changes in the list will
            # backpropagate
            self.neurons = [n for n in x.neurons]
        elif utils._is_iterable(x):
            # If x is a list of mixed objects we need to unpack/flatten that
            # E.g. x = [CatmaidNeuronList, CatmaidNeuronList, CatmaidNeuron, skeletonID]

            to_unpack = [e for e in x if isinstance(e, CatmaidNeuronList)]
            x = [e for e in x if not isinstance(e, CatmaidNeuronList)]
            x += [n for ob in to_unpack for n in ob.neurons]

            # We have to convert from numpy ndarray to list - do NOT remove
            # list() here
            self.neurons = list(x)
        else:
            raise TypeError(
                'Unable to generate CatmaidNeuronList from %s' % str(type(x)))

        # Now convert into CatmaidNeurons if necessary
        to_convert = []
        for i, n in enumerate(self.neurons):
            if not isinstance(n, CatmaidNeuron) or make_copy is True:
                to_convert.append((n, remote_instance, i))

        if to_convert:
            if self._use_threading:
                with ThreadPoolExecutor(max_workers=self.n_cores) as e:
                    futures = e.map(_convert_helper,
                                    to_convert)

                    converted = [n for n in config.tqdm(futures,
                                                        total=len(to_convert),
                                                        desc='Make nrn',
                                                        disable=config.pbar_hide,
                                                        leave=config.pbar_leave)]

                    for i, c in enumerate(to_convert):
                        self.neurons[c[2]] = converted[i]

            else:
                for n in config.tqdm(to_convert, desc='Make nrn',
                                     disable=config.pbar_hide,
                                     leave=config.pbar_leave):
                    self.neurons[n[2]] = CatmaidNeuron(n[0], remote_instance=remote_instance)

        # Add indexer class
        self.iloc = _IXIndexer(self.neurons)

        # Add skeleton ID indexer class
        self.skid = _SkidIndexer(self.neurons)

    def summary(self, n=None, add_cols=[]):
        """ Get summary over all neurons in this NeuronList.

        Parameters
        ----------
        n :         int | slice, optional
                    If int, get only first N entries.
        add_cols :  list, optional
                    Additional columns for the summary. If attribute not
                    available will return 'NA'.

        Returns
        -------
        pandas DataFrame

        """
        d = []

        # Set level to warning to avoid spam of "skeleton data not available"
        lvl = logger.level
        logger.setLevel('WARNING')

        if not isinstance(n, slice):
            n = slice(n)

        for n in self.neurons[n]:
            neuron_name = n.__dict__.get('neuron_name', 'NA')
            review_status = n.__dict__.get('review_status', 'NA')

            if 'nodes' in n.__dict__:
                soma_temp = n.soma is not None
            else:
                soma_temp = 'NA'

            this_n = [neuron_name, n.skeleton_id, n.n_nodes, n.n_connectors,
                      n.n_branch_nodes, n.n_end_nodes, n.n_open_ends,
                      n.cable_length, review_status, soma_temp]
            this_n += [getattr(n, a, 'NA') for a in add_cols]

            d.append(this_n)

        logger.setLevel(lvl)

        return pd.DataFrame(data=d,
                            columns=['neuron_name', 'skeleton_id', 'n_nodes',
                                     'n_connectors', 'n_branch_nodes',
                                     'n_end_nodes', 'open_ends',
                                     'cable_length', 'review_status',
                                     'soma'] + add_cols
                            )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '{0} of {1} neurons \n {2}'.format(type(self),
                                                  len(self.neurons),
                                                  str(self.summary()))

    def _repr_html_(self):
        return self.summary()._repr_html_()

    def __iter__(self):
        """ Iterator instanciates a new class everytime it is called.
        This allows the use of nested loops on the same neuronlist object.
        """
        class prange_iter:
            def __init__(self, neurons, start):
                self.iter = start
                self.neurons = neurons

            def __next__(self):
                if self.iter >= len(self.neurons):
                    raise StopIteration
                to_return = self.neurons[self.iter]
                self.iter += 1
                return to_return

        return prange_iter(self.neurons, 0)

    def __len__(self):
        """Use skeleton ID here, otherwise this is terribly slow."""
        return len(self.skeleton_id)

    def __dir__(self):
        """ Custom __dir__ to add some parameters that we want to make
        searchable.
        """
        add_attributes = ['n_open_ends', 'n_branch_nodes', 'n_end_nodes',
                          'cable_length', 'root', 'neuron_name',
                          'nodes', 'annotations', 'partners', 'review_status',
                          'connectors', 'presynapses', 'postsynapses',
                          'gap_junctions', 'soma', 'root', 'tags',
                          'n_presynapses', 'n_postsynapses', 'n_connectors',
                          'skeleton_id', 'empty', 'shape', 'bbox']

        return list(set(super().__dir__() + add_attributes))

    def __getattr__(self, key):
        if key == 'shape':
            return (self.__len__(),)
        elif key in ['n_nodes', 'n_connectors', 'n_presynapses',
                     'n_postsynapses', 'n_open_ends', 'n_end_nodes',
                     'cable_length', 'tags', 'igraph', 'soma', 'root',
                     'segments', 'graph', 'n_branch_nodes', 'dps',
                     'sampling_resolution']:
            self.get_skeletons(skip_existing=True)
            return np.array([getattr(n, key) for n in self.neurons])
        elif key == 'neuron_name':
            self.get_names(skip_existing=True)
            return np.array([n.neuron_name for n in self.neurons])
        elif key == 'partners':
            return self.get_partners()
        elif key == 'skeleton_id':
            return np.array([n.skeleton_id for n in self.neurons])
        elif key in ['nodes', 'connectors', 'presynapses', 'postsynapses',
                     'gap_junctions']:
            self.get_skeletons(skip_existing=True)
            data = []
            for n in self.neurons:
                this_n = getattr(n, key)
                # Do NOT remove this: downstream functions may depend on having
                # this reference
                this_n['skeleton_id'] = n.skeleton_id
                data.append(this_n)
            return pd.concat(data, axis=0, ignore_index=True, sort=True,
                             join='inner')
        elif key == 'bbox':
            return self.nodes.describe().loc[['min', 'max'], ['x', 'y', 'z']].values.T
        elif key == '_remote_instance':
            all_instances = [n._remote_instance for n in self.neurons if n._remote_instance]

            if len(set(all_instances)) > 1:
                # Note that multiprocessing causes remote_instances to be pickled
                # and thus not be the same anymore
                logger.debug('Neurons are using multiple remote_instances! '
                             'Returning first entry.')
            elif len(set(all_instances)) == 0:
                logger.warning('No remote_instance found. Use '
                               '.set_remote_instance() to assign one to all '
                               'neurons.')
                return None
            return all_instances[0]
        elif key == 'review_status':
            self.get_review(skip_existing=True)
            return np.array([n.review_status for n in self.neurons])
        elif key == 'annotations':
            to_retrieve = [
                n.skeleton_id for n in self.neurons if 'annotations' not in n.__dict__]
            if to_retrieve:
                re = fetch.get_annotations(
                    to_retrieve, remote_instance=self._remote_instance)
                for n in [n for n in self.neurons if 'annotations' not in n.__dict__]:
                    n.annotations = re.get(str(n.skeleton_id), [])
            return np.array([n.annotations for n in self.neurons])
        elif key == 'empty':
            return len(self.neurons) == 0
        elif key == 'skeletons_missing':
            return any([not n.node_data for n in self.neurons])
        else:
            if False not in [key in n.__dict__ for n in self.neurons]:
                return np.array([getattr(n, key) for n in self.neurons])
            else:
                raise AttributeError('Attribute "%s" not found' % key)

    def __contains__(self, x):
        return x in self.neurons or str(x) in self.skeleton_id or x in self.neuron_name

    def __getitem__(self, key):
        if isinstance(key, six.string_types):
            if key.startswith('annotation:'):
                skids = utils.eval_skids(
                    key, remote_instance=self._remote_instance)
                subset = self[skids]
            else:
                subset = [n for n in self.neurons if key in n.neuron_name or key in n.skeleton_id]
        elif isinstance(key, np.ndarray) and key.dtype == 'bool':
            if len(self.neurons) != len(key):
                raise IndexError('Length of key ({}) must match number of '
                                 'neurons ({})'.format(len(key),
                                                       len(self.neurons)))
            subset = [n for i, n in enumerate(self.neurons) if key[i]]
        elif utils._is_iterable(key):
            if True in [isinstance(k, str) for k in key]:
                subset = [n for i, n in enumerate(self.neurons) if True in [
                    k == n.neuron_name for k in key] or True in [k == n.skeleton_id for k in key]]
            elif False not in [isinstance(k, bool) for k in key]:
                subset = [n for i, n in enumerate(self.neurons) if key[i]]
            else:
                subset = [self.neurons[i] for i in key]
        else:
            subset = self.neurons[key]

        if isinstance(subset, CatmaidNeuron):
            return subset

        return CatmaidNeuronList(subset, make_copy=self.copy_on_subset)

    def __missing__(self, key):
        logger.error('No neuron matching the search critera.')
        raise AttributeError('No neuron matching the search critera.')

    def __add__(self, to_add):
        """Implements addition. """
        if isinstance(to_add, CatmaidNeuron):
            return CatmaidNeuronList(self.neurons + [to_add],
                                     make_copy=self.copy_on_subset)
        elif isinstance(to_add, CatmaidNeuronList):
            return CatmaidNeuronList(self.neurons + to_add.neurons,
                                     make_copy=self.copy_on_subset)
        elif utils._is_iterable(to_add):
            if False not in [isinstance(n, CatmaidNeuron) for n in to_add]:
                return CatmaidNeuronList(self.neurons + list(to_add),
                                         make_copy=self.copy_on_subset)
            else:
                return CatmaidNeuronList(self.neurons
                                         + [CatmaidNeuron[n] for n in to_add],
                                         make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def __eq__(self, other):
        """Implements equality. """
        if isinstance(other, CatmaidNeuronList):
            if len(self) != len(other):
                return False
            else:
                return all([n1 == n2 for n1, n2 in zip(self, other)])
        else:
            return NotImplemented

    def __sub__(self, to_sub):
        """Implements substraction. """
        if isinstance(to_sub, (str, int)):
            return CatmaidNeuronList([n for n in self.neurons if n.skeleton_id != str(to_sub) and n.neuron_name != to_sub],
                                     make_copy=self.copy_on_subset)
        elif isinstance(to_sub, CatmaidNeuron):
            if to_sub.skeleton_id in self and to_sub not in self.neurons:
                logger.warning('Skeleton ID in neuronlist but neuron not '
                               'identical! Substraction cancelled! Try using '
                               '.skeleton_id instead.')
                return
            return CatmaidNeuronList([n for n in self.neurons if n != to_sub],
                                     make_copy=self.copy_on_subset)
        elif isinstance(to_sub, CatmaidNeuronList):
            if len(set(self.neurons) & set(to_sub.neurons)) != len(set(self.skeleton_id) & set(to_sub.skeleton_id)):
                logger.warning('Skeleton ID(s) in neuronlist but neuron(s) '
                               'not identical! Substraction cancelled! Try '
                               'using .skeleton_id instead.')
                return
            return CatmaidNeuronList([n for n in self.neurons if n not in to_sub],
                                     make_copy=self.copy_on_subset)
        elif utils._is_iterable(to_sub):
            # Make sure everything is a string
            to_sub = [str(s) for s in to_sub]
            return CatmaidNeuronList([n for n in self.neurons if n.skeleton_id not in to_sub and n.neuron_name not in to_sub],
                                     make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Implements division for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            nl = self.copy()
            for n in nl:
                n.nodes.loc[:, ['x', 'y', 'z', 'radius']] /= other
                n.connectors.loc[:, ['x', 'y', 'z']] /= other
                n._clear_temp_attr(exclude=['classify_nodes'])
            return nl
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implements multiplication for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            nl = self.copy()
            for n in nl:
                n.nodes.loc[:, ['x', 'y', 'z', 'radius']] *= other
                n.connectors.loc[:, ['x', 'y', 'z']] *= other
                n._clear_temp_attr(exclude=['classify_nodes'])
            return nl
        else:
            return NotImplemented

    def __and__(self, other):
        """Implements bitwise AND using the & operator. """
        if isinstance(other, (str, int)):
            return CatmaidNeuronList([n for n in self.neurons if n.skeleton_id == str(other) or n.neuron_name == other],
                                     make_copy=self.copy_on_subset)
        elif isinstance(other, CatmaidNeuron):
            if other.skeleton_id in self and other not in self.neurons:
                logger.warning('Skeleton IDs overlap but neuron not '
                               'identical! Bitwise cancelled! Try using '
                               '.skeleton_id instead.')
                return
            return CatmaidNeuronList([n for n in self.neurons if n == other],
                                     make_copy=self.copy_on_subset)
        elif isinstance(other, CatmaidNeuronList):
            if len(set(self.neurons) & set(other.neurons)) != len(set(self.skeleton_id) & set(other.skeleton_id)):
                logger.warning('Skeleton IDs overlap but neuron(s) not '
                               'identical! Bitwise cancelled! Try using '
                               '.skeleton_id instead.')
                return
            return CatmaidNeuronList([n for n in self.neurons if n in other],
                                     make_copy=self.copy_on_subset)
        elif utils._is_iterable(other):
            # Make sure everything is a string
            other = [str(s) for s in other]
            return CatmaidNeuronList([n for n in self.neurons if n.skeleton_id in other or n.neuron_name in other],
                                     make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def sum(self):
        """Returns sum numeric and boolean values over all neurons. """
        return self.summary().sum(numeric_only=True)

    def mean(self):
        """Returns mean numeric and boolean values over all neurons. """
        return self.summary().mean(numeric_only=True)

    def sample(self, N=1):
        """Returns random subset of neurons.

        Parameters
        ----------
        N :     int | float
                If int >= 1, will return N neurons. If float < 1, will return
                fraction of total neurons.
        """
        if N < 1:
            N = int(len(self.neurons) * N)

        indices = list(range(len(self.neurons)))
        random.shuffle(indices)
        return CatmaidNeuronList([n for i, n in enumerate(self.neurons) if i in indices[:N]],
                                 make_copy=self.copy_on_subset)

    def resample(self, resample_to, preserve_cn_treenodes=False, inplace=True):
        """Resamples all neurons to given resolution [nm].

        Parameters
        ----------
        resample_to :           int
                                Resolution in nm to resample neurons to.
        inplace :               bool, optional
                                If False, a downsampled COPY of this
                                CatmaidNeuronList is returned.

        See Also
        --------
        :func:`~pymaid.resample_neuron`
                Base function - see for details.
        """

        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(n, resample_to) for i, n in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(x._resample_helper, combinations,
                                            chunksize=10),
                                  total=len(combinations),
                                  desc='Downsampling',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for n in config.tqdm(x.neurons, desc='Resampling',
                          disable=config.pbar_hide, leave=config.pbar_leave):
                n.resample(resample_to=resample_to, inplace=True)

        if not inplace:
            return x

    def _resample_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].resample(resample_to=x[1], inplace=True)
        return x[0]

    def downsample(self, factor=5, inplace=True, **kwargs):
        """Downsamples (simplifies) all neurons by given factor.

        Parameters
        ----------
        factor :                int, optional
                                Factor by which to downsample the neurons.
                                Default = 5
        inplace :               bool, optional
                                If False, a downsampled COPY of this
                                CatmaidNeuronList is returned.
        **kwargs
                                Additional arguments passed to
                                :func:`~pymaid.downsample_neuron`.


        See Also
        --------
        :func:`~pymaid.downsample_neuron`
                Base function - see for details.
        """

        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(n, factor, kwargs)
                            for i, n in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(x._downsample_helper, combinations,
                                            chunksize=10),
                                  desc='Downsampling',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for n in config.tqdm(x.neurons, desc='Downsampling',
                          disable=config.pbar_hide, leave=config.pbar_leave):
                n.downsample(factor=factor, inplace=True, **kwargs)

        if not inplace:
            return x

    def _downsample_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].downsample(factor=x[1], inplace=True, **x[2])
        return x[0]

    def reroot(self, new_root, inplace=True):
        """ Reroot neuron to treenode ID or node tag.

        Parameters
        ----------
        new_root :  int | str | list of either
                    Either treenode IDs or node tag(s). If not a list, the
                    same tag is used to reroot all neurons.
        inplace :   bool, optional
                    If False, a rerooted COPY of this CatmaidNeuronList is
                    returned.

        See Also
        --------
        :func:`~pymaid.reroot_neuron`
                    Base function. See for details and more examples.

        Examples
        --------
        >>> # Reroot all neurons to soma
        >>> nl = pymaid.get_neuron('annotation:glomerulus DA1')
        >>> nl.reroot(nl.soma)
        """

        if not utils._is_iterable(new_root):
            new_root = [new_root] * len(self.neurons)

        if len(new_root) != len(self.neurons):
            raise ValueError('Must provided a new root for every neuron.')

        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        # Silence loggers (except Errors)
        level = logger.getEffectiveLevel()

        logger.setLevel('ERROR')

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(n, new_root[i]) for i, n in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(x._reroot_helper, combinations,
                                            chunksize=10),
                                  total=len(combinations),
                                  desc='Rerooting',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for i, n in enumerate(config.tqdm(x.neurons, desc='Rerooting',
                                       disable=config.pbar_hide,
                                       leave=config.pbar_leave)):
                n.reroot(new_root[i], inplace=True)

        # Reset logger level to previous state
        logger.setLevel(level)

        if not inplace:
            return x

    def _reroot_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].reroot(x[1], inplace=True)
        return x[0]

    def prune_distal_to(self, tag, inplace=True):
        """Cut off nodes distal to given node.

        Parameters
        ----------
        node :      node tag
                    A (unique) tag at which to cut the neurons
        inplace :   bool, optional
                    If False, a pruned COPY of this CatmaidNeuronList is
                    returned.

        """
        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(n, tag) for i, n in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(x._prune_distal_helper,
                                            combinations, chunksize=10),
                                  total=len(combinations),
                                  desc='Pruning',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for n in config.tqdm(x.neurons, desc='Pruning', disable=config.pbar_hide,
                          leave=config.pbar_leave):
                n.prune_distal_to(tag, inplace=True)

        if not inplace:
            return x

    def _prune_distal_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].prune_distal_to(x[1], inplace=True)
        return x[0]

    def prune_proximal_to(self, tag, inplace=True):
        """Remove nodes proximal to given node.

        Reroots neurons to cut node.

        Parameters
        ----------
        node :      node tag
                    A (unique) tag at which to cut the neurons
        inplace :   bool, optional
                    If False, a pruned COPY of this CatmaidNeuronList is
                    returned.

        """
        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(n, tag) for i, n in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(x._prune_proximal_helper,
                                            combinations,
                                            chunksize=10),
                                  total=len(combinations),
                                  desc='Pruning',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for n in config.tqdm(x.neurons, desc='Pruning', disable=config.pbar_hide,
                          leave=config.pbar_leave):
                n.prune_proximal_to(tag, inplace=True)

        if not inplace:
            return x

    def _prune_proximal_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].prune_proximal_to(x[1], inplace=True)
        return x[0]

    def prune_by_strahler(self, to_prune, inplace=True):
        """ Prune neurons based on `Strahler order
        <https://en.wikipedia.org/wiki/Strahler_number>`_.

        Will reroot neurons to soma if possible.

        Parameters
        ----------
        to_prune :  int | list | range | slice
                    Strahler indices to prune. For example:

                    1. ``to_prune=1`` removes all leaf branches
                    2. ``to_prune=[1, 2]`` removes SI 1 and 2
                    3. ``to_prune=range(1, 4)`` removes SI 1, 2 and 3
                    4. ``to_prune=slice(1, -1)`` removes everything but the
                       highest SI
                    5. ``to_prune=slice(-1, None)`` removes only the highest
                       SI

        inplace :   bool, optional
                    If False, pruning is done on a copy of this
                    CatmaidNeuronList which is then returned.

        See Also
        --------
        :func:`pymaid.prune_by_strahler`
                    Basefunction - see for details and examples.

        """
        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(n, to_prune) for i, n in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(x._prune_strahler_helper,
                                            combinations, chunksize=10),
                                  total=len(combinations),
                                  desc='Pruning',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for n in config.tqdm(x.neurons, desc='Pruning', disable=config.pbar_hide,
                          leave=config.pbar_leave):
                n.prune_by_strahler(to_prune=to_prune, inplace=True)

        if not inplace:
            return x

    def _prune_strahler_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].prune_by_strahler(to_prune=x[1], inplace=True)
        return x[0]

    def prune_by_longest_neurite(self, n=1, reroot_to_soma=False,
                                 inplace=True):
        """ Prune neurons down to their longest neurites.

        Parameters
        ----------
        n :                 int, optional
                            Number of longest neurites to preserve.
        reroot_to_soma :    bool, optional
                            If True, neurons will be rerooted to their somas
                            before pruning.
        inplace :           bool, optional
                            If False, a pruned COPY of this CatmaidNeuronList
                            is returned.

        See Also
        --------
        :func:`pymaid.prune_by_strahler`
                        Basefunction - see for details and examples.
        """

        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(neuron, n, reroot_to_soma)
                            for i, neuron in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(self._prune_neurite_helper,
                                            combinations, chunksize=10),
                                  total=len(combinations),
                                  desc='Pruning',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for neuron in config.tqdm(x.neurons, desc='Pruning',
                               disable=config.pbar_hide,
                               leave=config.pbar_leave):
                neuron.prune_by_longest_neurite(
                    n, reroot_to_soma=reroot_to_soma, inplace=True)

        if not inplace:
            return x

    def _prune_neurite_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].prune_by_longest_neurite(x[1], x[2], inplace=True)
        return x[0]

    def prune_by_volume(self, v, mode='IN', prevent_fragments=False,
                        inplace=True):
        """ Prune neurons by intersection with given volume(s).

        Parameters
        ----------
        v :                 str | pymaid.Volume | list of either
                            Volume(s) to check for intersection
        mode :              'IN' | 'OUT', optional
                            If 'IN', parts of the neuron inside the volume are
                            kept.
        prevent_fragments : bool, optional
                            If True, will add nodes to ``subset`` required to
                            keep neuron from fragmenting.
        inplace :           bool, optional
                            If True, operation will be performed on itself. If
                            False, operation is performed on copy which is then
                            returned.


        See Also
        --------
        :func:`~pymaid.in_volume`
                Basefunction - see for details and examples.
        """

        x = self
        if not inplace:
            x = x.copy(deepcopy=False)

        if not isinstance(v, Volume):
            v = fetch.get_volume(v, combine_vols=True,
                                 remote_instance=self._remote_instance)

        if x._use_parallel:
            pool = mp.Pool(x.n_cores)
            combinations = [(n, v, mode, prevent_fragments)
                                             for i, n in enumerate(x.neurons)]
            x.neurons = list(config.tqdm(pool.imap(x._prune_by_volume_helper,
                                            combinations, chunksize=10),
                                  total=len(combinations), desc='Pruning',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave))

            pool.close()
            pool.join()
        else:
            for n in config.tqdm(x.neurons, desc='Pruning', disable=config.pbar_hide,
                          leave=config.pbar_leave):
                n.prune_by_volume(v, mode=mode, inplace=True,
                                  prevent_fragments=prevent_fragments)

        if not inplace:
            return x

    def _prune_by_volume_helper(self, x):
        """ Helper function to parallelise basic operations."""
        x[0].prune_by_volume(x[1], mode=x[2], inplace=True,
                             prevent_fragments=x[3])
        return x[0]

    def get_partners(self, remote_instance=None):
        """ Get connectivity table for neurons."""
        if not remote_instance and not self._remote_instance:
            logger.error(
                'Get_partners: Unable to connect to server. Please provide CatmaidInstance as <remote_instance>.')
            return None
        elif not remote_instance:
            remote_instance = self._remote_instance

        # Get all partners
        self.partners = fetch.get_partners(self.skeleton_id,
                                           remote_instance=remote_instance)

        # Propagate connectivity table to individual neurons
        for n in self.neurons:
            n.partners = self.partners[self.partners[n.skeleton_id] > 0][['neuron_name', 'skeleton_id', 'num_nodes',
                                                                          'relation', n.skeleton_id]].reset_index(drop=True).sort_values(['relation', n.skeleton_id], ascending=False)
            n.partners['total'] = n.partners[n.skeleton_id]

        return self.partners

    def get_review(self, skip_existing=False):
        """ Use to get/update review status."""
        if skip_existing:
            to_update = [
                n.skeleton_id for n in self.neurons if 'review_status' not in n.__dict__]
        else:
            to_update = self.skeleton_id.tolist()

        if to_update:
            re = fetch.get_review(to_update,
                                  remote_instance=self._remote_instance).set_index('skeleton_id')
            for n in self.neurons:
                if str(n.skeleton_id) in re:
                    n.review_status = re.loc[str(
                        n.skeleton_id), 'percent_reviewed']

    def get_annotations(self, skip_existing=False):
        """Get/update annotations for neurons."""
        if skip_existing:
            to_update = [
                n.skeleton_id for n in self.neurons if 'annotations' not in n.__dict__]
        else:
            to_update = self.skeleton_id.tolist()

        if to_update:
            annotations = fetch.get_annotations(
                to_update, remote_instance=self._remote_instance)
            for n in self.neurons:
                n.annotations = annotations.get(str(n.skeleton_id), [])

    def get_names(self, skip_existing=False):
        """ Use to get/update neuron names."""
        if skip_existing:
            to_update = [
                n.skeleton_id for n in self.neurons if 'neuron_name' not in n.__dict__]
        else:
            to_update = self.skeleton_id.tolist()

        if to_update:
            names = fetch.get_names(
                self.skeleton_id, remote_instance=self._remote_instance)
            for n in self.neurons:
                if str(n.skeleton_id) in names:
                    n.neuron_name = names[str(n.skeleton_id)]

    def _generate_segments(self):
        """ Helper function to use multiprocessing to generate segments for all
        neurons. This will NOT force update of existing segments! This is about
        1.5X faster than calling them individually on a 4 core system.
        """

        if self._use_parallel:
            to_retrieve = [
                n for n in self.neurons if 'segments' not in n.__dict__]
            to_retrieve_ix = [i for i, n in enumerate(
                self.neurons) if 'segments' not in n.__dict__]

            pool = mp.Pool(self.n_cores)
            update = list(config.tqdm(pool.imap(self._generate_segments_helper,
                                         to_retrieve, chunksize=10),
                               total=len(to_retrieve), desc='Gen. segments',
                               disable=config.pbar_hide,
                               leave=config.pbar_leave))
            pool.close()
            pool.join()

            for ix, n in zip(to_retrieve_ix, update):
                self.neurons[ix] = n
        else:
            for n in config.tqdm(self.neurons, desc='Gen. segments',
                          disable=config.pbar_hide, leave=config.pbar_leave):
                if 'segments' not in n.__dict__:
                    _ = n.segments

    def _generate_segments_helper(self, x):
        """ Helper function to parallelise basic operations."""
        if 'segments' not in x.__dict__:
            _ = x.segments
        return x

    def reload(self):
        """ Update neuron skeletons from server."""
        self.get_skeletons(skip_existing=False)

    def get_skeletons(self, skip_existing=False):
        """Helper function to fill in/update skeleton data of neurons.
        Updates ``.nodes``, ``.connectors``, ``.tags``, ``.date_retrieved`` and
        ``.neuron_name``. Will also generate new graph representation to match
        nodes/connectors.
        """

        if skip_existing:
            to_update = [n for n in self.neurons if 'nodes' not in n.__dict__]
        else:
            to_update = self.neurons

        if to_update:
            skdata = fetch.get_neuron([n.skeleton_id for n in to_update],
                                      remote_instance=self._remote_instance,
                                      return_df=True).set_index('skeleton_id')
            for n in config.tqdm(to_update, desc='Processing neurons',
                          disable=config.pbar_hide, leave=config.pbar_leave):

                n.nodes = skdata.loc[str(n.skeleton_id), 'nodes']
                n.connectors = skdata.loc[str(n.skeleton_id), 'connectors']
                n.tags = skdata.loc[str(n.skeleton_id), 'tags']
                n.neuron_name = skdata.loc[str(n.skeleton_id), 'neuron_name']
                n.date_retrieved = datetime.datetime.now().isoformat()

                # Delete and update attributes
                n._clear_temp_attr()

    def set_remote_instance(self, remote_instance=None, server_url=None,
                            http_user=None, http_pw=None, auth_token=None):
        """Assign remote_instance to all neurons.

        Provide either existing CatmaidInstance OR your credentials.

        Parameters
        ----------
        remote_instance :       pymaid.CatmaidInstance, optional
        server_url :            str, optional
        http_user :             str, optional
        http_pw :               str, optional
        auth_token :            str, optional

        """

        if not remote_instance and server_url and auth_token:
            remote_instance = fetch.CatmaidInstance(server_url,
                                                    http_user,
                                                    http_pw,
                                                    auth_token
                                                    )
        elif not remote_instance:
            raise Exception('Provide either CatmaidInstance or credentials.')

        for n in self.neurons:
            n._remote_instance = remote_instance

    def plot3d(self, **kwargs):
        """Plot neuron in 3D.

        Parameters
        ----------
        **kwargs
                Keyword arguments will be passed to :func:`pymaid.plot3d`.
                See ``help(pymaid.plot3d)`` for a list of keywords.

        See Also
        --------
        :func:`~pymaid.plot3d`
                Base function called to generate 3d plot.
        """

        from pymaid import plotting

        if 'remote_instance' not in kwargs:
            kwargs.update({'remote_instance': self._remote_instance})

        self.get_skeletons(skip_existing=True)
        return plotting.plot3d(self, **kwargs)

    def plot2d(self, **kwargs):
        """Plot neuron in 2D.

        Parameters
        ----------
        **kwargs
                Keyword arguments will be passed to :func:`pymaid.plot2d`.
                See ``help(pymaid.plot2d)`` for a list of accepted keywords.

        See Also
        --------
        :func:`~pymaid.plot2d`
                Base function called to generate 2d plot.
        """

        from pymaid import plotting

        self.get_skeletons(skip_existing=True)
        return plotting.plot2d(self, **kwargs)

    def has_annotation(self, x, intersect=False, partial=False,
                       raise_not_found=True):
        """Filter neurons by their annotations.

        Parameters
        ----------
        x :               str | list of str
                          Annotation(s) to filter for. Use tilde (~) as prefix
                          to look for neurons WITHOUT given annotation(s).
        intersect :       bool, optional
                          If True, neuron must have ALL positive annotations to
                          be included and ALL negative (~) annotations to be
                          excluded. If False, must have at least one positive
                          to be included and one of the negative annotations
                          to be excluded.
        partial :         bool, optional
                          If True, allow partial match of annotation.
        raise_not_found : bool, optional
                          If True, will raise exception if no match is found.
                          If False, will simply return empty list.

        Returns
        -------
        :class:`pymaid.CatmaidNeuronList`
                          Neurons that have given annotation(s).

        Examples
        --------
        >>> # Get neurons that have "test1" annotation
        >>> nl.has_annotation('test1')
        >>> # Get neurons that have either "test1" or "test2"
        >>> nl.has_annotation(['test1', 'test2'])
        >>> # Get neurons that have BOTH "test1" and "test2"
        >>> nl.has_annotation(['test1', 'test2'], intersect=True)
        >>> # Get neurons that have "test1" but NOT "test2"
        >>> nl.has_annotation(['test1', '~test2'])

        """

        # This makes sure nobody accidentally forgets brackets around
        # multiple annotations
        for v in [intersect, partial, raise_not_found]:
            if not isinstance(v, bool):
                raise TypeError('Expected boolean, got {}'.format(type(v)))

        inc, exc = utils._eval_conditions(x)

        if not inc and not exc:
            raise ValueError('Must provide at least a single annotation')

        # Make sure we have annotations to begin with
        self.get_annotations(skip_existing=True)

        selection = []
        for n in self.neurons:
            if inc:
                if not partial:
                    pos = [a in n.annotations for a in inc]
                else:
                    pos = [any(a in b for b in n.annotations) for a in inc]

                # Skip if any positive annotation is missing
                if intersect and not all(pos):
                    continue
                # Skip if none of the positive annotations are there
                elif not intersect and not any(pos):
                    continue

            if exc:
                if not partial:
                    neg = [a in n.annotations for a in exc]
                else:
                    neg = [any(a in b for b in n.annotations) for a in exc]

                # Skip if all negative annotations are present
                if intersect and all(neg):
                    continue
                # Skip if any of the negative annotations are present
                elif not intersect and any(neg):
                    continue

            selection.append(n)

        if not selection and raise_not_found:
            raise ValueError('No neurons with matching annotation(s) found')
        else:
            return CatmaidNeuronList(selection, make_copy=self.copy_on_subset)

    @classmethod
    def from_selection(self, fname):
        """ Generates NeuronList from json files generated by CATMAID's
        selection table.
        """

        # Read data from file
        with open(fname, 'r') as f:
            data = json.load(f)

        # Generate NeuronLost
        nl = CatmaidNeuronList([e['skeleton_id'] for e in data])

        # Add colors
        for k, e in enumerate(data):
            nl[k].color = tuple(
                int(e['color'].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

        return nl

    def to_selection(self, save_to='selection.json'):
        """Generate neuron selection as json file which can be loaded
        in CATMAID selection tables. Uses neuron's ``.color`` attribute.

        Parameters
        ----------
        save_to :   str | None, optional
                    Filename to save selection to. If ``None``, will
                    return the json data instead.
        """

        data = [dict(skeleton_id=int(n.skeleton_id),
                     color="#{:02x}{:02x}{:02x}".format(
                         n.color[0], n.color[1], n.color[2]),
                     opacity=1
                     ) for n in self.neurons]

        if save_to:
            with open(save_to, 'w') as outfile:
                json.dump(data, outfile)

            logger.info('Selection saved as {}.'.format(save_to))
        else:
            return data

    def to_dataframe(self):
        """ Turn this CatmaidneuronList into a pandas DataFrame containing
        only original Catmaid data.

        Returns
        -------
        pandas DataFrame
                neuron_name  skeleton_id  nodes  connectors  tags
            0
            1
        """

        return pd.DataFrame([[n.neuron_name, n.skeleton_id, n.nodes,
                              n.connectors, n.tags] for n in self.neurons],
                              columns=['neuron_name', 'skeleton_id', 'nodes',
                                       'connectors', 'tags'])

    def to_swc(self, filenames=None):
        """ Generate SWC file from this neuron.

        This converts CATMAID nanometer coordinates into microns.

        Parameters
        ----------
        filenames :  None | str | list, optional
                     If ``None``, will use "neuron_{skeletonID}.swc". Pass
                     filenames as list when processing multiple neurons.

        Returns
        -------
        Nothing

        See Also
        --------
        :func:`~pymaid.to_swc`
                See this for further details.

        """

        return utils.to_swc(self, filenames)

    def itertuples(self):
        """Helper class to mimic ``pandas.DataFrame`` ``itertuples()``."""
        return self.neurons

    def sort_values(self, key, ascending=False):
        """Sort neurons by given key.

        Needs to be an attribute of all neurons: for example ``n_presynapse``.
        Also works with custom attributes."""
        self.neurons = sorted(self.neurons,
                              key=lambda x: getattr(x, key),
                              reverse=ascending is False)

    def __copy__(self):
        return self.copy(deepcopy=False)

    def __deepcopy__(self):
        return self.copy(deepcopy=True)

    def copy(self, deepcopy=False):
        """Return copy of this CatmaidNeuronList.

        Parameters
        ----------
        deepcopy :  bool, optional
                    If False, ``.graph`` (NetworkX DiGraphs) will be returned
                    as views - changes to nodes/edges can progagate back!
                    ``.igraph`` (iGraph) - if available - will always be
                    deepcopied.

        """
        return CatmaidNeuronList([n.copy(deepcopy=deepcopy) for n in self.neurons],
                                 make_copy=False,
                                 _use_parallel=self._use_parallel)

    def head(self, n=5):
        """Return summary for top N neurons."""
        return self.summary(n=n)

    def tail(self, n=5):
        """Return summary for bottom N neurons."""
        return self.summary(n=slice(-n, len(self)))

    def remove_duplicates(self, key='skeleton_id', inplace=False):
        """Removes duplicate neurons from list.

        Parameters
        ----------
        key :       str | list, optional
                    Attribute(s) by which to identify duplicates. In case of
                    multiple, all attributes must match to flag a neuron as
                    duplicate.
        inplace :   bool, optional
                    If False will return a copy of the original with
                    duplicates removed.
        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        key = utils._make_iterable(key)

        # Generate pandas DataFrame
        df = pd.DataFrame([[getattr(n, at) for at in key] for n in x],
                          columns=key)

        # Find out which neurons to keep
        keep = ~df.duplicated(keep='first').values

        # Assign neurons
        x.neurons = x[keep].neurons

        # We have to reassign the Indexer classes here
        # For some reason the neuron list does not propagate
        x.iloc = _IXIndexer(x.neurons)
        x.skid = _SkidIndexer(x.neurons)

        if not inplace:
            return x


class _IXIndexer():
    """ Location based indexer added to CatmaidNeuronList objects to allow
    indexing similar to pandas DataFrames using df.iloc[0]. This is really
    just a helper to allow code to operate on CatmaidNeuron the same way
    it would on DataFrames.
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.obj[key]
        else:
            raise Exception('Unable to index non-integers.')


class _SkidIndexer():
    """ Skeleton ID based indexer added to CatmaidNeuronList objects to allow
    indexing. This allows you to get a neuron by its skeleton ID.
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, skid):
        # Turn into list and force strings
        skid = utils._make_iterable(skid, force_type=str)

        # Get objects that match skid
        sel = [n for n in self.obj if str(n.skeleton_id) in skid]

        # Reorder to keep in the order requested
        sel = sorted(sel, key=lambda x: np.where(skid == str(x.skeleton_id))[0][0])

        if len(sel) == 0:
            raise ValueError('No neuron(s) with given skeleton ID(s):'
                             ' {0}'.format(skid))
        elif len(sel) == 1:
            return sel[0]
        else:
            return CatmaidNeuronList(sel)


class Dotprops(pd.DataFrame):
    """ Class to hold dotprops. This is essentially a pandas DataFrame - we
    just use it to tell dotprops from other objects.

    See Also
    --------
    :func:`pymaid.rmaid.dotprops2py`
        Converts R dotprops to :class:`~pymaid.Dotprops`.

    Notes
    -----
    This class is still in the making but the idea is to write methods for it
    like .plot3d(), .to_X().

    """


class Volume:
    """ Class representing CATMAID meshes.

    Parameters
    ----------
    vertices :  list | array
                Vertices coordinates. Must be shape ``(N, 3)``.
    faces :     list | array
                Indexed faceset.
    name :      str, optional
                Name of volume.
    color :     tuple, optional
                RGB color.
    volume_id : int, optional
                CATMAID volume ID.

    See Also
    --------
    :func:`~pymaid.get_volume`
        Retrieves volumes from CATMAID and returns :class:`pymaid.Volume`.

    """

    def __init__(self, vertices, faces, name=None, color=(1, 1, 1, .1),
                 volume_id=None, **kwargs):
        self.name = name
        self.vertices = vertices
        self.faces = faces
        self.color = color
        self.volume_id = volume_id

    @classmethod
    def from_csv(self, vertices, faces, name=None, color=(1, 1, 1, .1),
                 volume_id=None, **kwargs):
        """ Load volume from csv files containing vertices and faces.

        Parameters
        ----------
        vertices :      filepath | file-like
                        CSV file containing vertices.
        faces :         filepath | file-like
                        CSV file containing faces.
        **kwargs
                        Keyword arguments passed to ``csv.reader``.

        Returns
        -------
        pymaid.Volume

        """

        with open(vertices, 'r') as f:
            reader = csv.reader(f, **kwargs)
            vertices = np.array([r for r in reader]).astype(float)

        with open(faces, 'r') as f:
            reader = csv.reader(f, **kwargs)
            faces = np.array([r for r in reader]).astype(int)

        return Volume(faces=faces, vertices=vertices, name=name, color=color,
                      volume_id=volume_id)

    def to_csv(self, filename, **kwargs):
        """ Save volume as two separated csv files containing vertices and
        faces.

        Parameters
        ----------
        filename :      str
                        Filename to use. Will get a ``_vertices.csv`` and
                        ``_faces.csv`` suffix.
        **kwargs
                        Keyword arguments passed to ``csv.reader``.
        """

        for data, suffix in zip([self.faces, self.vertices],
                                ['_faces.csv', '_vertices.csv']):
            with open(filename + suffix, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)

    @classmethod
    def combine(self, x, name='comb_vol', color=(1, 1, 1, .1)):
        """ Merges multiple volumes into a single object.

        Parameters
        ----------
        x :     list or dict of Volumes
        name :  str, optional
                Name of the combined volume.
        color : tuple, optional
                Color of the combined volume.

        Returns
        -------
        :class:`~pymaid.Volume`
        """

        if isinstance(x, Volume):
            return x

        if isinstance(x, dict):
            x = list(x.values())

        if not utils._is_iterable(x):
            x = [x]

        if False in [isinstance(v, Volume) for v in x]:
            raise TypeError('Input must be list of volumes')

        vertices = np.empty((0, 3))
        faces = []

        # Reindex faces
        for vol in x:
            offs = len(vertices)
            vertices = np.append(vertices, vol.vertices, axis=0)
            faces += [[f[0] + offs, f[1] + offs, f[2] + offs]
                      for f in vol.faces]

        return Volume(vertices=vertices, faces=faces, name=name, color=color)

    @property
    def bbox(self):
        """ Bounding box of this volume. """
        return np.array([self.vertices.min(axis=0),
                         self.vertices.max(axis=0)]).T

    @property
    def vertices(self):
        """Array (N, 3) of vertex coordinates. """
        return self.__vertices

    @vertices.setter
    def vertices(self, v):
        if not isinstance(v, np.ndarray):
            v = np.array(v)

        if not v.shape[1] == 3:
            raise ValueError('Vertices must be of shape N,3.')

        self.__vertices = v

    @property
    def verts(self):
        """Legacy access to ``.vertices``."""
        return self.vertices

    @verts.setter
    def verts(self, v):
        self.vertices = v

    @property
    def faces(self):
        """Array of vertex indices forming faces."""
        return self.__faces

    @faces.setter
    def faces(self, v):
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        self.__faces = v

    @property
    def center(self):
        """ Center of volume as average over all vertices."""
        return np.mean(self.vertices, axis=0)

    def __deepcopy__(self):
        return self.copy()

    def __copy__(self):
        return self.copy()

    def copy(self):
        """Return copy of this volume. Does not maintain generic values."""
        return Volume(self.vertices.copy(), self.faces.copy(),
                      copy.copy(self.name), copy.copy(self.color),
                      copy.copy(self.volume_id))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '{0} "{1}" at {2}: {3} vertices, {4} faces'.format(type(self),
                                                                  self.name,
                                                                  hex(id(self)),
                                                                  self.vertices.shape[0],
                                                                  self.faces.shape[0])

    def __truediv__(self, other):
        """Implements division for vertex coordinates."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            return self.__mul__(1/other)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implements multiplication for vertex coordinates."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            v = self.copy()
            v.vertices *= other
            return v
        else:
            return NotImplemented

    def resize(self, x, method='center', inplace=True):
        """ Resize volume.

        Parameters
        ----------
        x :         int | float
                    Resizing factor. For methods "center", "centroid" and
                    "origin" this is the fraction of original size (e.g.
                    ``.5`` for half size). For method "normals", this is
                    is the absolute displacement (e.g. ``-1000`` to shrink
                    volume by 1um)!
        method :    "center" | "centroid" | "normals" | "origin"
                    Point in space to use for resizing.

                    .. list-table::
                        :widths: 15 75
                        :header-rows: 1

                        * - method
                          - explanation
                        * - center
                          - average of all vertices
                        * - centroid
                          - average of the triangle centroids weighted by the
                            area of each triangle. Requires ``trimesh``.
                        * - origin
                          - resizes relative to origin of coordinate system
                            (0, 0, 0)
                        * - normals
                          - resize using face normals. Note that this method
                            uses absolute displacement for parameter ``x``.
                            Requires ``trimesh``.

        inplace :   bool, optional
                    If False, will return resized copy.

        Returns
        -------
        :class:`pymaid.Volume`
                    Resized copy of original volume. Only if ``inplace=False``.
        None
                    If ``inplace=True``.
        """

        perm_methods = ['center', 'origin', 'normals', 'centroid']
        if method not in perm_methods:
            raise ValueError('Unknown method "{}". Allowed '
                             'methods: {}'.format(method,
                                                  ', '.join(perm_methods)))

        if method in ['normals', 'centroid'] and isinstance(trimesh, type(None)):
            raise ImportError('Must have Trimesh installed to use methods '
                              '"normals" or "centroid"!')

        if not inplace:
            v = self.copy()
        else:
            v = self

        if method == 'normals':
            tm = v.to_trimesh()
            v.vertices = tm.vertices + (tm.vertex_normals * x)
            v.faces = tm.faces
        else:
            # Get the center
            if method == 'center':
                cn = np.mean(v.vertices, axis=0)
            elif method == 'centroid':
                cn = v.to_trimesh().centroid
            elif method == 'origin':
                cn = np.array([0, 0, 0])

            # Get vector from center to each vertex
            vec = v.vertices - cn

            # Multiply vector by resize factor
            vec *= x

            # Recalculate vertex positions
            v.vertices = vec + cn

        # Make sure to reset any pyoctree data on this volume
        try:
            delattr(v, 'pyoctree')
        except BaseException:
            pass

        if not inplace:
            return v

    def plot3d(self, **kwargs):
        """Plot neuron using :func:`pymaid.plot3d`.

        Parameters
        ----------
        **kwargs
                Keyword arguments. Will be passed to :func:`pymaid.plot3d`.
                See ``help(pymaid.plot3d)`` for a list of keywords.

        See Also
        --------
        :func:`pymaid.plot3d`
                    Function called to generate 3d plot.

        Examples
        --------
        >>> vol = pymaid.get_volume('v14.LH_R')
        >>> vol.plot3d(color = (255, 0, 0))
        """

        from pymaid import plotting

        if 'color' in kwargs:
            self.color = kwargs['color']

        return plotting.plot3d(self, **kwargs)

    def to_trimesh(self):
        """ Returns trimesh representation of this volume.

        See Also
        --------
        https://github.com/mikedh/trimesh
                trimesh GitHub page.
        """

        if isinstance(trimesh, type(None)):
            raise ImportError('Unable to import trimesh. Please make sure it '
                              'is installed properly')

        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def _outlines_3d(self, view='xy', **kwargs):
        """ Generate 3d outlines along a given view (see ``.to_2d()``).

        Parameters
        ----------
        **kwargs
                    Keyword arguments passed to :func:`~pymaid.Volume.to_2d`.

        Returns
        -------
        list
                    Coordinates of 2d circumference.
                    e.g. ``[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), ...]``
                    Third dimension is averaged
        """

        co2d = np.array(self.to_2d(view=view, **kwargs))

        if view in ['xy', 'yx']:
            third = np.repeat(self.center[2], co2d.shape[0])
        elif view in ['xz', 'zx']:
            third = np.repeat(self.center[1], co2d.shape[0])
        elif view in ['yz', 'zy']:
            third = np.repeat(self.center[0], co2d.shape[0])

        return np.append(co2d, third.reshape(co2d.shape[0], 1), axis=1)

    def to_2d(self, alpha=0.00017, view='xy', invert_y=False):
        """ Computes the 2d alpha shape (concave hull).

        Uses Scipy Delaunay and shapely.

        Parameters
        ----------
        alpha:      float, optional
                    Alpha value to influence the gooeyness of the border.
                    Smaller numbers don't fall inward as much as larger
                    numbers. Too large, and you lose everything!
        view :      'xy' | 'xz' | 'yz', optional
                    Determines if frontal, lateral or top view.

        Returns
        -------
        list
                    Coordinates of 2d circumference
                    e.g. ``[(x1, y1), (x2, y2), (x3, y3), ...]``

        """

        def add_edge(edges, edge_points, coords, i, j):
            """ Add a line between the i-th and j-th points,
            if not in the list already.
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            edges.add((i, j))
            edge_points.append(coords[[i, j]])

        accepted_views = ['xy', 'xz', 'yz']

        try:
            from shapely.ops import cascaded_union, polygonize
            import shapely.geometry as geometry
        except ImportError:
            raise ImportError('This function needs the <shapely> package.')

        if view in['xy', 'yx']:
            coords = self.vertices[:, [0, 1]]
            if invert_y:
                coords[:, 1] = coords[:, 1] * -1

        elif view in ['xz', 'zx']:
            coords = self.vertices[:, [0, 2]]
        elif view in ['yz', 'zy']:
            coords = self.vertices[:, [1, 2]]
            if invert_y:
                coords[:, 0] = coords[:, 0] * -1
        else:
            raise ValueError(
                'View {0} unknown. Please use either: {1}'.format(view,
                                                                  accepted_views))

        tri = scipy.spatial.Delaunay(coords)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the
        # triangle
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]
            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
            b = math.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
            c = math.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
            # Semiperimeter of triangle
            s = (a + b + c) / 2.0
            # Area of triangle by Heron's formula
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            # Here's the radius filter.
            if circum_r < 1.0 / alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)

        m = geometry.MultiLineString(edge_points)
        triangles = list(polygonize(m))
        concave_hull = cascaded_union(triangles)

        return list(concave_hull.exterior.coords)

def _convert_helper(x):
    """ Helper function to convert x to CatmaidNeuron."""
    return CatmaidNeuron(x[0], remote_instance=x[1])
