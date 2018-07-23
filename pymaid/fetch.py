# A collection of tools to remotely access a CATMAID server via its API
#
#    Copyright (C) 2017 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along

""" This module contains functions to request data from Catmaid server.

Examples
--------
>>> import pymaid
>>> # HTTP_USER AND HTTP_PASSWORD are only necessary if your server requires a
... # http authentification
>>> myInstance = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' ,
...                                      'HTTP_USER' ,
...                                      'HTTP_PASSWORD',
...                                      'TOKEN' )
>>> # Get skeletal data for two neurons
>>> neuron_list = pymaid.get_neuron ( ['12345','67890'] , myInstance )
>>> neuron_list[0]
type              <class 'pymaid.CatmaidNeuron'>
neuron_name                       Example neuron name
skeleton_id                                     12345
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

"""

import urllib
import json
import time
import base64

import requests
from requests_futures.sessions import FuturesSession

import datetime
import re
import pandas as pd
import numpy as np
import sys
import networkx as nx

from pymaid import core, graph, utils, config
from pymaid.intersect import in_volume

__all__ = sorted(['CatmaidInstance', 'add_annotations', 'add_tags',
                  'get_3D_skeleton', 'get_3D_skeletons',
                  'get_annotation_details', 'get_annotation_id',
                  'get_annotation_list', 'get_annotations', 'get_arbor',
                  'get_connector_details', 'get_connectors',
                  'get_contributor_statistics', 'get_edges', 'get_history',
                  'get_logs', 'get_names', 'get_neuron', 'get_neuron_list',
                  'get_neurons', 'get_neurons_in_bbox',
                  'get_neurons_in_volume', 'get_node_tags', 'get_node_details',
                  'get_nodes_in_volume', 'get_partners',
                  'get_partners_in_volume', 'get_paths', 'get_review',
                  'get_review_details', 'get_skids_by_annotation',
                  'get_skids_by_name', 'get_treenode_info',
                  'get_treenode_table', 'get_user_annotations',
                  'get_user_list', 'get_volume', 'has_soma', 'neuron_exists',
                  'delete_tags', 'get_segments', 'delete_neuron',
                  'get_connectors_between', 'url_to_coordinates',
                  'rename_neurons', 'get_label_list', 'find_neurons',
                  'get_skid_from_treenode', 'get_transactions',
                  'remove_annotations', 'get_connector_links'])

# Set up logging
logger = config.logger


class CatmaidInstance:
    """ Class giving access to a CATMAID instance. Holds base url,
    credentials and fetches data. You can either pass it to
    functions individually or define globally (default).

    Attributes
    ----------
    server :        str
                    The url for a CATMAID server.
    authname :      str
                    The http user.
    authpassword :  str
                    The http password.
    authtoken :     str
                    User token - see CATMAID documentation on how to get it.
    project_id :    int, optional
                    ID of your project. Default = 1.
    max_threads :   int | None
                    Maximum parallel threads to be used. Note that some
                    functions (e.g. :func:`pymaid.get_skid_from_treenode`)
                    override this parameter.
    set_global :    bool, optional
                    If True, this instance will be set as global (default)
                    CatmaidInstance. This overrides pre-existing global
                    instances.

    Examples
    --------
    Initialise a CatmaidInstance. Note that ``HTTP_USER`` and ``HTTP_PASSWORD``
    are only necessary if your server requires HTTP authentification.

    >>> rm = pymaid.CatmaidInstance('https://www.your.catmaid-server.org',
    ...                             'HTTP_USER',
    ...                             'HTTP_PASSWORD',
    ...                             'TOKEN')
    INFO  : Global CATMAID instance set. (pymaid.fetch)

    As you instanciate CatmaidInstance, it is made the default, “global”
    remote instance and you don’t need to worry about it anymore.

    By default, a CatmaidInstance will refer to the first project on your
    server. To illustrate, let's assume you have two projects and you want to
    fetch data from both:

    >>> p1 = pymaid.CatmaidInstance('https://www.your.catmaid-server.org',
    ...                             'HTTP_USER',
    ...                             'HTTP_PASSWORD',
    ...                             'TOKEN')
    >>> # Make copy of CatmaidInstance and change project ID
    >>> p2 = p1.copy()
    >>> p2.project_id = 2
    >>> # Fetch a neuron from project 1 and another from project 2
    >>> n1 = pymaid.get_neuron(16, remote_instance=p1)
    >>> n2 = pymaid.get_neuron(233007, remote_instance=p2)

    Manually make one CatmaidInstance the global one

    >>> p2.make_global()

    Ordinarily, you would use one of the wrapper functions to fetch data
    from the server (e.g. :func:`pymaid.get_neuron`). If however you want
    to get the **raw data**, here is how:

    >>> # 1. Fetch raw skeleton data for a single neuron
    >>> rm = pymaid.CatmaidInstance('https://www.your.catmaid-server.org',
    ...                             'HTTP_USER',
    ...                             'HTTP_PASSWORD',
    ...                             'TOKEN')
    >>> skeleton_id = 12345
    >>> url = rm._get_compact_skeleton_url(skeleton_id)
    >>> raw_data = rm.fetch(url)
    >>> # 2. Query for neurons matching given criteria using GET request
    >>> GET = {'nodecount_gt': 1000, # min node size
                    'created_by': 16}    # user ID
    >>> url = rm._get_list_skeletons_url(**GET)
    >>> raw_data = rm.fetch(url)
    >>> # 3. Fetch contributions using POST request
    >>> url = rm._get_contributions_url()
    >>> POST = {'skids[0]': 16, 'skids[1]': 2333007}
    >>> raw_data = rm.fetch(url, POST)
    """

    def __init__(self, server, authname, authpassword, authtoken,
                 project_id=1, max_threads=100, make_global=True):
        # Catch too many backslashes
        if server.endswith('/'):
            server = server[:-1]

        self.server = server
        self.project_id = project_id
        self.authname = authname
        self.authpassword = authpassword
        self.authtoken = authtoken
        self.__max_threads = max_threads

        self._session = requests.Session()
        self._future_session = FuturesSession(session=self._session,
                                              max_workers=self.max_threads)

        if authname is not None and authpassword is not None:
            self._session.auth = (authname, authpassword)

        if authtoken is not None:
            self._session.headers['X-Authorization'] = 'Token ' + authtoken

        if make_global:
            self.make_global()
        else:
            logger.info(
                'CATMAID instance created. See help(pymaid.CatmaidInstance) to learn how to define globally.')

    @property
    def max_threads(self):
        return self.__max_threads

    @max_threads.setter
    def max_threads(self, v):
        if not isinstance(v, int):
            raise TypeError('max_threads has to be integer')
        if v < 1:
            raise ValueError('max_threads must be > 0')

        self.__max_threads = v
        self._future_session = FuturesSession(session=self._session,
                                              max_workers=self.__max_threads)

    def make_global(self):
        """Sets this variable as global by attaching it as sys.module"""
        sys.modules['remote_instance'] = self
        logger.info('Global CATMAID instance set.')

    def fetch(self, url, post=None, method=None, desc='Fetching', raw=False,
              callback=None, disable_pbar=False, leave_pbar=True):
        """ Requires the url to connect to and the variables for POST,
        if any, in a dictionary.
        """
        if utils._is_iterable(url):
            # Prepare futures
            if post or method == 'POST':
                if not utils._is_iterable(post):
                    raise ValueError('POST needs to be provided for each url.')
                futures = [self._future_session.post(u,
                                                     data=p,
                                                     background_callback=callback) for u, p in zip(url, post)]
            else:
                futures = [self._future_session.get(u,
                                                    params=None,
                                                    background_callback=callback) for u in url]

            # Get the responses
            response = [f.result() for f in config.tqdm(futures,
                                                        desc=desc,
                                                        disable=disable_pbar | config.pbar_hide,
                                                        leave=leave_pbar & config.pbar_leave)]

            # Make sure all responses returned data
            for r in response:
                r.raise_for_status()

            if not raw:
                return [r.json() for r in response]
            else:
                return response

        if post or method == 'POST':
            response = self._session.post(url, data=post)
        else:
            response = self._session.get(url, params=None)

        if not raw:
            return response.json()
        else:
            return response

    def make_url(self, *args, **GET):
        """ Generates URL from *args. **GET are assumed to be arguments
        for GET requests and will be added.
        """
        # Generate the URL
        url = self.server
        for arg in args:
            arg_str = str(arg)
            joiner = '' if url.endswith('/') else '/'
            relative = arg_str[1:] if arg_str.startswith('/') else arg_str
            url = requests.compat.urljoin(url + joiner, relative)
        if GET:
            url += '?{}'.format(urllib.parse.urlencode(GET))
        return url

    def _get_catmaid_version(self, **GET):
        """ Use to parse url for retrieving CATMAID server version"""
        return self.make_url('version', **GET)

    def _get_stack_info_url(self, stack_id, **GET):
        """ Use to parse url for retrieving stack infos. """
        return self.make_url(self.project_id, 'stack', stack_id, 'info', **GET)

    def _get_projects_url(self, **GET):
        """ Use to get list of available projects on server. Does not need postdata."""
        return self.make_url('projects', **GET)

    def _get_stacks_url(self, **GET):
        """ Use to get list of available image stacks for the project. Does not need postdata."""
        return self.make_url(self.project_id, 'stacks', **GET)

    def _get_treenode_info_url(self, tn_id, **GET):
        """ Use to parse url for retrieving skeleton info from treenodes."""
        return self.make_url(self.project_id, 'treenodes', tn_id, 'info', **GET)

    def _get_node_labels_url(self, **GET):
        """ Use to parse url for retrieving treenode infos. Needs postdata!"""
        return self.make_url(self.project_id, 'labels-for-nodes', **GET)

    def _get_skeleton_nodes_url(self, skid, **GET):
        """ Use to parse url for retrieving skeleton nodes (no info on parents or synapses, does need post data). """
        return self.make_url(self.project_id, 'treenode', 'table', skid, 'content', **GET)

    def _get_skeleton_for_3d_viewer_url(self, skid, **GET):
        """ ATTENTION: this url doesn't work properly anymore as of 07/07/14
        use compact-skeleton instead
        Used to parse url for retrieving all info the 3D viewer gets (does NOT need post data)
        Format: name, nodes, tags, connectors, reviews
        """
        return self.make_url(self.project_id, 'skeleton', skid, 'compact-json', **GET)

    def _get_add_annotations_url(self, **GET):
        """ Use to parse url to add annotations to skeleton IDs. """
        return self.make_url(self.project_id, 'annotations', 'add', **GET)

    def _get_remove_annotations_url(self, **GET):
        """ Use to parse url to add annotations to skeleton IDs. """
        return self.make_url(self.project_id, 'annotations', 'remove', **GET)

    def _get_connectivity_url(self, **GET):
        """ Use to parse url for retrieving connectivity (does need post data). """
        return self.make_url(self.project_id, 'skeletons', 'connectivity', **GET)

    def _get_connector_links_url(self, **GET):
        """ Use to retrieve list of connectors either pre- or postsynaptic a set of neurons - GET request
        Format: { 'links': [ skeleton_id, connector_id, x,y,z, S(?),
                confidence, creator, treenode_id, creation_date ], 'tags':[] }
        """
        return self.make_url(self.project_id, 'connectors', 'links/', **GET)

    def _get_connectors_url(self, **GET):
        """ Use to retrieve list of connectors - POST request
        """
        return self.make_url(self.project_id, 'connectors/', **GET)

    def _get_connector_types_url(self, **GET):
        """ Use to retrieve list of connectors - POST request
        """
        return self.make_url(self.project_id, 'connectors/types/', **GET)

    def _get_connector_types_url(self, **GET):
        """ Use to retrieve dictionary of connector types in the project
        """
        return self.make_url(self.project_id, 'connectors', 'types', **GET)

    def _get_connectors_between_url(self, **GET):
        """ Use to retrieve list of connectors linking sets of neurons
        """
        return self.make_url(self.project_id, 'connector', 'list', 'many_to_many', **GET)

    def _get_connector_details_url(self, **GET):
        """ Use to parse url for retrieving info connectors (does need post data). """
        return self.make_url(self.project_id, 'connector', 'skeletons', **GET)

    def _get_neuronnames(self, **GET):
        """ Use to parse url for names for a list of skeleton ids
        (does need post data: self.project_id, skid). """
        return self.make_url(self.project_id, 'skeleton', 'neuronnames', **GET)

    def _get_list_skeletons_url(self, **GET):
        """ Use to parse url for names for a list of skeleton ids. GET request. """
        return self.make_url(self.project_id, 'skeletons/', **GET)

    def _get_graph_dps_url(self, **GET):
        """ Use to parse url for getting connections between source and targets. """
        return self.make_url(self.project_id, 'graph', 'dps', **GET)

    def _get_completed_connector_links(self, **GET):
        """ Use to parse url for retrieval of completed connector links by
        given user GET request:
        Returns list: [ connector_id, [x,z,y], node1_id, skeleton1_id,
        link1_confidence, creator_id, [x,y,z], node2_id, skeleton2_id,
        link2_confidence, creator_id ]
        """
        return self.make_url(self.project_id, 'connector', 'list', **GET)

    def _get_user_list_url(self, **GET):
        """ Get user list for project. """
        return self.make_url('user-list', **GET)

    def _get_single_neuronname_url(self, skid, **GET):
        """ Use to parse url for a SINGLE neuron (will also give you neuronID). """
        return self.make_url(self.project_id, 'skeleton', skid, 'neuronname', **GET)

    def _get_review_status_url(self, **GET):
        """ Use to get skeletons review status. """
        return self.make_url(self.project_id, 'skeletons', 'review-status', **GET)

    def _get_review_details_url(self, skid, **GET):
        """ Use to retrieve review status for every single node of a skeleton.
        For some reason this needs to be fetched as POST (even though actual POST data is not necessary)
        Returns list of arbors, the nodes contained and who has been reviewing them at what time
        """
        return self.make_url(self.project_id, 'skeletons', skid, 'review', **GET)

    def _get_reviewed_neurons_url(self, **GET):
        """ Use to retrieve review status for every single node of a skeleton.
        For some reason this needs to fetched as POST (even though actual POST data is not necessary)
        Returns list of arbors, the nodes the contain and who has been reviewing them at what time
        """
        return self.make_url(self.project_id, 'skeletons', skid, 'review', **GET)

    def _get_annotation_table_url(self, **GET):
        """ Use to get annotations for given neuron. DOES need skid as postdata. """
        return self.make_url(self.project_id, 'annotations', 'table-list', **GET)

    def _get_intersects(self, vol_id, x, y, z, **GET):
        """ Use to test if point intersects with volume."""
        GET.update({'x': x, 'y': y, 'z': z})
        return self.make_url(self.project_id, 'volumes', vol_id, 'intersect', **GET)

    def _get_volumes(self, **GET):
        """ Get list of all volumes in project. """
        return self.make_url(self.project_id, 'volumes/', **GET)

    def _get_volume_details(self, volume_id, **GET):
        """ Get details on a given volume (mesh). """
        return self.make_url(self.project_id, 'volumes', volume_id, **GET)

    def _get_annotations_for_skid_list(self, **GET):
        """ ATTENTION: This does not seem to work anymore as of 20/10/2015
        -> although it still exists in CATMAID code
        use get_annotations_for_skid_list2
        Use to get annotations for given neuron. DOES need skid as postdata
        """
        return self.make_url(self.project_id, 'annotations', 'skeletons', 'list', **GET)

    def _get_annotations_for_skid_list2(self, **GET):
        """ Use to get annotations for given neuron. DOES need skid as postdata. """
        return self.make_url(self.project_id, 'skeleton', 'annotationlist', **GET)

    def _get_logs_url(self, **GET):
        """ Use to get logs. DOES need skid as postdata. """
        return self.make_url(self.project_id, 'logs', 'list', **GET)

    def _get_transactions_url(self, **GET):
        """ Use to get transactions. GET request."""
        return self.make_url(self.project_id, 'transactions/', **GET)

    def _get_annotation_list(self, **GET):
        """ Use to parse url for retrieving list of all annotations
        (and their IDs!!!).
        """
        return self.make_url(self.project_id, 'annotations/', **GET)

    def _get_contributions_url(self, **GET):
        """ Use to parse url for retrieving contributor statistics for given
        skeleton (does need post data). """
        return self.make_url(self.project_id, 'skeleton', 'contributor_statistics_multiple', **GET)

    def _get_annotated_url(self, **GET):
        """ Use to parse url for retrieving annotated neurons (NEEDS post data). """
        return self.make_url(self.project_id, 'annotations', 'query-targets', **GET)

    def _get_skid_from_tnid(self, treenode_id, **GET):
        """ Use to parse url for retrieving the skeleton id to a single
        treenode id (does not need postdata) API returns dict:
        {"count": integer, "skeleton_id": integer}
        """
        return self.make_url(self.project_id, 'skeleton', 'node', treenode_id, 'node_count', **GET)

    def _get_node_list_url(self, **GET):
        """ Use to parse url for retrieving list of nodes (NEEDS post data). """
        return self.make_url(self.project_id, 'node', 'list', **GET)

    def _get_node_info_url(self, **GET):
        """ Use to parse url for retrieving user info on a single node (needs post data). """
        return self.make_url(self.project_id, 'node', 'user-info', **GET)

    def _treenode_add_tag_url(self, treenode_id, **GET):
        """ Use to parse url adding labels (tags) to a given treenode (needs post data)."""
        return self.make_url(self.project_id, 'label', 'treenode', treenode_id, 'update', **GET)

    def _delete_neuron_url(self, neuron_id, **GET):
        """ Use to parse url for deleting a single neurons"""
        return self.make_url(self.project_id, 'label', 'treenode', treenode_id, 'update', **GET)

    def _delete_treenode_url(self, **GET):
        """ Use to parse url for deleting treenodes"""
        return self.make_url(self.project_id, 'treenode', 'delete', **GET)

    def _delete_connector_url(self, **GET):
        """ Use to parse url for deleting connectors"""
        return self.make_url(self.project_id, 'connector', 'delete', **GET)

    def _connector_add_tag_url(self, treenode_id, **GET):
        """ Use to parse url adding labels (tags) to a given treenode (needs post data)."""
        return self.make_url(self.project_id, 'label', 'connector', treenode_id, 'update', **GET)

    def _get_compact_skeleton_url(self, skid, connector_flag=1, tag_flag=1, **GET):
        """ Use to parse url for retrieving all info the 3D viewer gets (does NOT need post data).
        Returns, in JSON, [[nodes], [connectors], [tags]], with connectors and tags being empty when 0 == with_connectors and 0 == with_tags, respectively.
        Deprecated but kept for backwards compability!
        """
        return self.make_url(self.project_id, skid, connector_flag, tag_flag, 'compact-skeleton', **GET)

    def _get_compact_details_url(self, skid, **GET):
        """ Similar to compact-skeleton but if 'with_history':True is passed as GET request, returned data will include all positions a nodes/connector has ever occupied plus the creation time and last modified.
        """
        return self.make_url(self.project_id, 'skeletons', skid, 'compact-detail', **GET)

    def _get_compact_arbor_url(self, skid, nodes_flag=1, connector_flag=1, tag_flag=1, **GET):
        """ The difference between this function and get_compact_skeleton is that the connectors contain the whole chain from the skeleton of interest to the
        partner skeleton: contains [treenode_id, confidence_to_connector, connector_id, confidence_from_connector, connected_treenode_id, connected_skeleton_id, relation1, relation2]
        relation1 = 1 means presynaptic (this neuron is upstream), 0 means postsynaptic (this neuron is downstream)
        """
        return self.make_url(self.project_id, skid, nodes_flag, connector_flag, tag_flag, 'compact-arbor', **GET)

    def _get_edges_url(self, **GET):
        """ Use to parse url for retrieving edges between given skeleton ids (does need postdata).
        Returns list of edges: [source_skid, target_skid, weight]
        """
        return self.make_url(self.project_id, 'skeletons', 'confidence-compartment-subgraph', **GET)

    def _get_skeletons_from_neuron_id(self, neuron_id, **GET):
        """ Use to get all skeletons of a given neuron (neuron_id). """
        return self.make_url(self.project_id, 'neuron', neuron_id, 'get-all-skeletons', **GET)

    def _get_history_url(self, **GET):
        """ Use to get user history. """
        return self.make_url(self.project_id, 'stats', 'user-history', **GET)

    def _get_stats_node_count(self, **GET):
        """ Use to get nodecounts per user. """
        return self.make_url(self.project_id, 'stats', 'nodecount', **GET)

    def _rename_neuron_url(self, neuron_id, **GET):
        """ Use to rename a single neuron. Does need postdata."""
        return self.make_url(self.project_id, 'neurons', neuron_id, 'rename', **GET)

    def _get_label_list_url(self, **GET):
        """ Use to rename a single neuron. Does need postdata."""
        return self.make_url(self.project_id, 'labels', 'stats', **GET)


def get_neuron(x, remote_instance=None, connector_flag=1, tag_flag=1,
               get_history=False, get_merge_history=False, get_abutting=False,
               return_df=False, kwargs={}):
    """ Retrieve 3D skeleton data.

    Parameters
    ----------
    x
                        Can be either:

                        1. list of skeleton ID(s), int or str
                        2. list of neuron name(s), str, exact match
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    connector_flag :    0 | False | 1 | True, optional
                        Set if connector data should be retrieved.
                        Note: the CATMAID API endpoint does currently not
                        support retrieving abutting connectors this way.
                        Please use ``get_abutting=True`` to set an additional
                        flag.
    tag_flag :          0 | False | 1 | True, optional
                        Set if tags should be retrieved.
    get_history:        bool, optional
                        If True, the returned node data will contain
                        creation date and last modified for each
                        node.

                        ATTENTION: if ``get_history=True``, nodes/connectors
                        that have been moved since their creation will have
                        multiple entries reflecting their changes in position!
                        Each state has the date it was modified as creation
                        date and the next state's date as last modified. The
                        most up to date state has the original creation date
                        as last modified.
                        The creator_id is always the original creator though.
    get_abutting:       bool, optional
                        If True, will retrieve abutting connectors.
                        For some reason they are not part of compact-json, so
                        they have to be retrieved via a separate API endpoint
                        -> will show up as connector type 3!
    return_df :         bool, optional
                        If True, a ``pandas.DataFrame`` instead of
                        ``CatmaidNeuron``/``CatmaidNeuronList`` is returned.
    **kwargs
                        Above BOOLEAN parameters can also be passed as dict.
                        This is then used in CatmaidNeuron objects to
                        override implicitly set parameters!

    Returns
    -------
    :class:`~pymaid.CatmaidNeuron`
                        For single neurons.
    :class:`~pymaid.CatmaidNeuronList`
                        For a list of neurons.
    pandas.DataFrame
                        If ``return_df=True``

    Notes
    -----
    The returned objects contain for each neuron::

        neuron_name :           str
        skeleton_id :           str
        nodes / connectors :    pandas.DataFrames containing treenode/connector
                                ID, coordinates, parent nodes, etc.
        tags :                  dict containing the treenode tags:
                                {'tag': [ treenode_id, treenode_id, ... ]}

    Dataframe column titles for ``nodes`` and ``connectors`` should be
    self-explanatory with the exception of ``relation`` in connector table::

        connectors['relation']

                    0 = presynapse
                    1 = postsynapse
                    2 = gap junction
                    3 = abutting connector

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    # Update from kwargs if available
    tag_flag = kwargs.get('tag_flag', tag_flag)
    connector_flag = kwargs.get('connector_flag', connector_flag)
    get_history = kwargs.get('get_history', get_history)
    get_merge_history = kwargs.get('get_merge_history', get_merge_history)
    get_abutting = kwargs.get('get_abutting', get_abutting)
    return_df = kwargs.get('return_df', return_df)

    # Convert tag_flag, connector_tag, get_history and get_merge_history to
    # bool if necessary
    if isinstance(tag_flag, int):
        tag_flag = tag_flag == 1
    if isinstance(connector_flag, int):
        connector_flag = connector_flag == 1
    if isinstance(get_history, int):
        get_history = get_history == 1
    if isinstance(get_merge_history, int):
        get_merge_history = get_merge_history == 1

    # Generate URLs to retrieve
    urls = [remote_instance._get_compact_details_url(s) for s in x]
    GET = urllib.parse.urlencode(
                {'with_history': str(get_history).lower(),
                'with_tags': str(tag_flag).lower(),
                'with_connectors': str(connector_flag).lower(),
                'with_merge_history': str(get_merge_history).lower()})
    urls = [u + '?%s' % GET for u in urls]

    skdata = remote_instance.fetch(urls, desc='Fetch neurons')

    # Retrieve abutting
    if get_abutting:
        urls = [remote_instance._get_connector_links_url() for s in x]

        GET = [urllib.parse.urlencode({'skeleton_ids[0]': str(s),
                                       'relation_type': 'abutting'})
                                                          for s in x]
        urls = [u + '?%s' % g for u, g in zip(urls, GET)]

        cn_data = remote_instance.fetch(urls, desc='Fetch abbutt.')

        # Add abutting to other connectors in skdata with type == 3
        for i, cn in enumerate(cn_data):
            if not get_history:
                skdata[i][1] += [[c[7], c[1], 3, c[2], c[3], c[4]]
                                 for c in cn['links']]
            else:
                skdata[i][1] += [[c[7], c[1], 3, c[2],
                                  c[3], c[4], c[8], None]
                                 for c in cn['links']]

    # Get neuron names
    names = get_names(x, remote_instance)

    if not get_history:
        df = pd.DataFrame([[
            names[str(x[i])],
            str(x[i]),
            pd.DataFrame(n[0],
                         columns=['treenode_id', 'parent_id',
                                  'creator_id', 'x', 'y', 'z',
                                  'radius', 'confidence'],
                         dtype=object),
            pd.DataFrame(n[1],
                         columns=['treenode_id', 'connector_id',
                                  'relation', 'x', 'y', 'z'],
                         dtype=object),
            n[2]]
            for i, n in enumerate(skdata)
        ],
            columns=['neuron_name', 'skeleton_id',
                     'nodes', 'connectors', 'tags'],
            dtype=object
        )
    else:
        df = pd.DataFrame([[
            names[str(x[i])],
            str(x[i]),
            pd.DataFrame(n[0],
                         columns=['treenode_id', 'parent_id',
                                  'creator_id', 'x', 'y', 'z',
                                  'radius', 'confidence',
                                  'last_modified',
                                  'creation_date'],
                         dtype=object),
            pd.DataFrame(n[1],
                         columns=['treenode_id', 'connector_id',
                                  'relation', 'x', 'y', 'z',
                                  'last_modified',
                                  'creation_date'],
                         dtype=object),
            n[2]]
            for i, n in enumerate(skdata)
        ],
            columns=['neuron_name', 'skeleton_id',
                     'nodes', 'connectors', 'tags'],
            dtype=object
        )

    # Convert data to respective dtypes
    dtypes = {'treenode_id': int, 'parent_id': object,
              'creator_id': int, 'relation': int,
              'connector_id': int, 'x': int, 'y': int, 'z': int,
              'radius': int, 'confidence': int}

    for k, v in dtypes.items():
        for t in ['nodes', 'connectors']:
            for i in range(df.shape[0]):
                if k in df.loc[i, t]:
                    df.loc[i, t][k] = df.loc[i, t][k].astype(v)

    if return_df:
        return df

    if df.shape[0] > 1:
        return core.CatmaidNeuronList(df, remote_instance=remote_instance,)
    else:
        return core.CatmaidNeuron(df.loc[0], remote_instance=remote_instance,)


# This is for legacy reasons -> will remove eventually
get_3D_skeleton = get_3D_skeletons = get_neurons = get_neuron


def get_arbor(x, remote_instance=None, node_flag=1, connector_flag=1,
              tag_flag=1):
    """ Retrieve skeleton data for a list of skeleton ids.
    Similar to :func:`pymaid.get_neuron` but the connector data includes
    the whole chain::

        treenode1 -> (link_confidence) -> connector -> (link_confidence)
        -> treenode2

    This means that connectors can shop up multiple times (i.e. if they have
    multiple postsynaptic targets). Does include connector x,y,z coordinates!

    Parameters
    ----------
    x
                        Neurons to retrieve. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    connector_flag :    0 | 1, optional
                        Set if connector data should be retrieved.
    tag_flag :          0 | 1, optional
                        Set if tags should be retrieved.


    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron:

        >>> df
        ...  neuron_name   skeleton_id   nodes      connectors   tags
        ... 0  str             str     DataFrame     DataFrame   dict

    Notes
    -----
    - nodes and connectors are pandas.DataFrames themselves
    - tags is a dict: ``{ 'tag' : [ treenode_id, treenode_id, ... ] }``

    Dataframe (df) column titles should be self explanatory with these exception:

    - ``df['relation_1']`` describes treenode_1 to/from connector
    - ``df['relation_2']`` describes treenode_2 to/from connector
    - ``relation`` can be: ``0`` (presynaptic), ``1`` (postsynaptic), ``2`` (gap junction)

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    skdata = []

    for s in config.tqdm(x, desc='Retrieving arbors', disable=config.pbar_hide,
                  leave=config.pbar_leave):
        # Create URL for retrieving example skeleton from server
        remote_compact_arbor_url = remote_instance._get_compact_arbor_url(
            s, node_flag, connector_flag, tag_flag)

        # Retrieve node_data for example skeleton
        arbor_data = remote_instance.fetch(remote_compact_arbor_url)

        skdata.append(arbor_data)

        logger.debug('%s retrieved' % str(s))

    names = get_names(x, remote_instance)

    df = pd.DataFrame([[
        names[str(x[i])],
        str(x[i]),
        pd.DataFrame(n[0], columns=['treenode_id', 'parent_id', 'creator_id',
                                    'x', 'y', 'z', 'radius', 'confidence']),
        pd.DataFrame(n[1], columns=['treenode_1', 'link_confidence',
                                    'connector_id', 'link_confidence',
                                    'treenode_2', 'other_skeleton_id',
                                    'relation_1', 'relation_2']),
        n[2]]
        for i, n in enumerate(skdata)
    ],
        columns=['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags'],
        dtype=object
    )
    return df


def get_partners_in_volume(x, volume, remote_instance=None, threshold=1,
                           min_size=2):
    """ Retrieve the synaptic/gap junction partners of neurons
    of interest **within** a given CATMAID Volume.

    Important
    ---------
    Connectivity (total number of connections) returned is restricted to
    that volume.

    Parameters
    ----------
    x
                        Neurons to check. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    volume :            str | list of str | core.Volume
                        Name of the CATMAID volume to test OR volume dict with
                        {'vertices':[],'faces':[]} as returned by e.g.
                        :func:`~pymaid.get_volume()`.
    remote_instance :   CATMAID instance
                        If not passed directly, will try using global.
    threshold :         int, optional
                        Does not seem to have any effect on CATMAID API and is
                        therefore filtered afterwards. This threshold is
                        applied to the TOTAL number of synapses across all
                        neurons!
    min_size :          int, optional
                        Minimum node count of partner
                        (default = 2 -> hide single-node partner).

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of
        synapses with the query neurons:

        >>> df
        ...  neuron_name  skeleton_id  num_nodes   relation     skid1  skid2 ...
        ... 1  name1         skid1    node_count1  upstream     n_syn  n_syn ..
        ... 2  name2         skid2    node_count2  downstream   n_syn  n_syn .
        ... 3  name3         skid3    node_count3  gapjunction  n_syn  n_syn .

        - Relation can be: upstream (incoming), downstream (outgoing) of the neurons of interest or gap junction
        - partners can show up multiple times if they are e.g. pre- AND postsynaptic
        - the number of connections between two partners is restricted to the volume

    See Also
    --------
    :func:`~pymaid.get_neurons_in_volume`
            Get all neurons within given volume.
    :func:`~pymaid.filter_connectivity`
            Filter connectivity table or adjacency matrix by volume(s) or to
            parts of neuron(s).

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    # First, get list of connectors
    cn_data = get_connectors(x, remote_instance=remote_instance)

    # Find out which connectors are in the volume of interest
    iv = in_volume(
        cn_data[['x', 'y', 'z']], volume, remote_instance)

    # Get the subset of connectors within the volume
    cn_in_volume = cn_data[iv].copy()

    logger.info(
        '%i unique connectors in volume. Reconstructing connectivity...' % len(cn_in_volume.connector_id.unique()))

    # Get details for connectors in volume
    cn_details = get_connector_details(
        cn_in_volume.connector_id.unique(),
        remote_instance=remote_instance)

    # Filter those connectors that don't have a presynaptic node
    cn_details = cn_details[~cn_details.presynaptic_to.isnull()]

    # Now reconstruct connectivity table from connector details

    # Some connectors may be connected to the same neuron multiple times
    # In those cases there will be more treenode IDs in "postsynaptic_to_node"
    # than there are skeleton IDs in "postsynaptic_to". Then we need to map
    # treenode IDs to neurons
    mismatch = cn_details[cn_details.postsynaptic_to.apply(
        len) < cn_details.postsynaptic_to_node.apply(len)]
    match = cn_details[cn_details.postsynaptic_to.apply(
        len) >= cn_details.postsynaptic_to_node.apply(len)]

    if not mismatch.empty:
        logger.info(
            'Retrieving additional details for {0} connectors'.format(mismatch.shape[0]))
        tn_to_skid = get_skid_from_treenode([tn for l in mismatch.postsynaptic_to_node.values for tn in l],
                                            remote_instance=remote_instance)
    else:
        tn_to_skid = []

    # Now collect edges
    edges = [[cn.presynaptic_to, skid]
             for cn in match.itertuples() for skid in cn.postsynaptic_to]
    edges += [[cn.presynaptic_to, tn_to_skid[tn]]
              for cn in mismatch.itertuples() for tn in cn.postsynaptic_to_node]

    # Turn edges into synaptic connections
    unique_edges, counts = np.unique(edges, return_counts=True, axis=0)
    unique_skids = np.unique(edges).astype(str)
    unique_edges = unique_edges.astype(str)

    # Create empty adj_mat
    adj_mat = pd.DataFrame(np.zeros((len(unique_skids), len(unique_skids))),
                           columns=unique_skids, index=unique_skids)

    for i, e in enumerate(config.tqdm(unique_edges, disable=config.pbar_hide,
                               desc='Adj. matrix', leave=config.pbar_leave)):
        # using df.at here speeds things up tremendously!
        adj_mat.loc[str(e[0]), str(e[1])] = counts[i]

    # There is a chance that our original neurons haven't made it through filtering
    # (i.e. they don't have partners in the volume ). We will simply add these
    # rows and columns and set them to 0
    missing = [n for n in x if n not in adj_mat.columns]
    for n in missing:
        adj_mat[n] = 0

    missing = [n for n in x if n not in adj_mat.index]
    for n in missing:
        adj_mat.loc[n] = [0 for i in range(adj_mat.shape[1])]

    # Generate connectivity table
    all_upstream = adj_mat.T[adj_mat.T[x].sum(axis=1) > 0][x]
    all_upstream['skeleton_id'] = all_upstream.index
    all_upstream['relation'] = 'upstream'

    all_downstream = adj_mat[adj_mat[x].sum(axis=1) > 0][x]
    all_downstream['skeleton_id'] = all_downstream.index
    all_downstream['relation'] = 'downstream'

    # Merge tables
    df = pd.concat([all_upstream, all_downstream], axis=0, ignore_index=True)

    # We will use this to get name and size of neurons
    logger.info('Collecting additional info for {0} neurons'.format(
        len(df.skeleton_id.unique())))
    review = get_review(df.skeleton_id.unique(),
                        remote_instance=remote_instance).set_index('skeleton_id')

    df['neuron_name'] = [review.loc[str(s), 'neuron_name']
                         for s in df.skeleton_id.values]
    df['num_nodes'] = [review.loc[str(s), 'total_node_count']
                       for s in df.skeleton_id.values]
    df['total'] = df[x].sum(axis=1)

    # Filter for min size
    df = df[df.num_nodes >= min_size]

    # Reorder columns
    df = df[['neuron_name', 'skeleton_id', 'num_nodes', 'relation', 'total'] + x]

    df.sort_values(['relation', 'total'], inplace=True, ascending=False)

    return df.reset_index(drop=True)


def get_partners(x, remote_instance=None, threshold=1, min_size=2, filt=[],
                 min_confidence=1, directions=['incoming', 'outgoing',
                 'gapjunctions', 'attachments']):
    """ Retrieve partners connected by synapses, gap junctions or attachments
    to a set of neurons.

    Note
    ----
    This function treats multiple fragments with the same skeleton ID
    (e.g. from splits into axon & dendrites) as a single neuron when fetching
    data from the server. For "fragmented" connectivity use
    :func:`~pymaid.cn_table_from_connectors` instead.

    Parameters
    ----------
    x
                        Neurons for which to retrieve partners. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    threshold :         int, optional
                        Minimum # of links (synapses/gap-junctions/etc).
    min_size :          int, optional
                        Minimum node count of partner
                        (default=2 to hide single-node partners).
    filt :              list of str, optional
                        Filters partners for neuron names (must be exact) or
                        skeleton_ids.
    min_confidence :    int | None, optional
                        If set, edges with lower confidence will be ignored.
    directions :        'incoming' | 'outgoing' | 'gapjunctions' | 'attachments', optional
                        Use to restrict to either up- or downstream partners.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of
        synapses with the query neurons:

        >>> df
          neuron_name  skeleton_id    num_nodes    relation   total  skid1  skid2 ...
        0   name1         skid1      node_count1  upstream    n_syn  n_syn  ...
        1   name2         skid2      node_count2  downstream  n_syn  n_syn  ..
        2   name3         skid3      node_count3  gapjunction n_syn  n_syn  .
        ... ...

        ``relation`` can be ``'upstream'`` (incoming), ``'downstream'``
        (outgoing), ``'attachment'`` or ``'gapjunction'`` (gap junction).

    Warning
    -------
    By default, will exclude single node partners! Set ``min_size=1`` to return
    ALL partners including placeholder nodes.

    Notes
    -----
    Partners can show up multiple times if they are e.g. pre- AND postsynaptic!

    Examples
    --------
    >>> example_skids = [16,201,150,20]
    >>> cn = pymaid.get_partners( example_skids, remote_instance )
    >>> # Get only upstream partners
    >>> subset = cn[ cn.relation == 'upstream' ]
    >>> # Get partners with more than e.g. 5 synapses across all neurons
    >>> subset2 = cn[ cn[ example_skids ].sum(axis=1) > 5 ]
    >>> # Combine above conditions (watch parentheses!)
    >>> subset3 = cn[ (cn.relation=='upstream') &
    ... (cn[example_skids].sum(axis=1) > 5) ]

    See Also
    --------
    :func:`~pymaid.adjacency_matrix`
                    Use if you need an adjacency matrix instead of a table.
    :func:`~pymaid.get_partners_in_volume`
                    Use if you only want connectivity within a given volume.
    :func:`~pymaid.filter_connectivity`
                   Use to restrict connector table to given part of a neuron
                   or a volume.
    :func:`~cn_table_from_connectors`
                   Returns "fragmented" connectivity. Use e.g. if you are working
                   with multiple fragments from the same neuron.

    """

    if not isinstance(min_confidence, (float, int)) or min_confidence < 0 or min_confidence > 5:
        raise ValueError('min_confidence must be 0-5.')

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_connectivity_url = remote_instance._get_connectivity_url()

    connectivity_post = {}
    connectivity_post['boolean_op'] = 'OR'
    connectivity_post['with_nodes'] = False

    for i, skid in enumerate(x):
        tag = 'source_skeleton_ids[{0}]'.format(i)
        connectivity_post[tag] = skid

    logger.info(
        'Fetching connectivity table for {0} neurons'.format(len(x)))
    connectivity_data = remote_instance.fetch(
        remote_connectivity_url, connectivity_post)

    # Delete directions that we don't want
    connectivity_data.update(
        {d: [] for d in connectivity_data if d not in directions})

    # Get neurons' names
    names = get_names([n for d in connectivity_data for n in connectivity_data[
                      d]] + list(x), remote_instance)

    df = pd.DataFrame(columns=['neuron_name', 'skeleton_id',
                               'num_nodes', 'relation'] + [str(s) for s in x])

    relations = {
        'incoming': 'upstream',
        'outgoing': 'downstream',
        'gapjunctions': 'gapjunction',
        'attachments': 'attachment'
    }

    # Number of synapses is returned as list of links with 0-5 confidence:
    # {'skid': [0,1,2,3,4,5]}
    # This is being collapsed into a single value before returning it.
    for d in relations:
        df_temp = pd.DataFrame([[
            names[str(n)],
            str(n),
            int(connectivity_data[d][n]['num_nodes']),
            relations[d]]
            + [sum(connectivity_data[d][n]['skids'].get(str(s),
                                                        [0, 0, 0, 0, 0])[min_confidence - 1:]) for s in x]
            for i, n in enumerate(connectivity_data[d])
        ],
            columns=['neuron_name', 'skeleton_id', 'num_nodes',
                     'relation'] + [str(s) for s in x],
            dtype=object
        )

        df = pd.concat([df, df_temp], axis=0)

    df['total'] = df[x].sum(axis=1).values

    # Now filter for synapse threshold and size
    df = df[(df.num_nodes >= min_size) & (df.total >= threshold)]

    df.sort_values(['relation', 'total'], inplace=True, ascending=False)

    if filt:
        if not isinstance(filt, (list, np.ndarray)):
            filt = [filt]

        filt = [str(s) for s in filt]

        df = df[df.skeleton_id.isin(filt) | df.neuron_name.isin(filt)]

    df.datatype = 'connectivity_table'

    # Return reindexed concatenated dataframe
    df.reset_index(drop=True, inplace=True)

    logger.info('Done. Found {0} pre-, {1} postsynaptic and {2} gap junction-connected neurons'.format(
        *[df[df.relation == r].shape[0] for r in ['upstream', 'downstream', 'gapjunction']]))

    return df


def get_names(x, remote_instance=None):
    """ Retrieve neurons names for a list of skeleton ids.

    Parameters
    ----------
    x
                        Neurons for wich to retrieve names. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        Either pass directly to function or define
                        globally as ``remote_instance``.

    Returns
    -------
    dict
                    ``{skid1: 'neuron_name', skid2: 'neuron_name', ...}``

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    x = list(set(x))

    remote_get_names_url = remote_instance._get_neuronnames()

    get_names_postdata = {}
    get_names_postdata['self.project_id'] = remote_instance.project_id

    for i in range(len(x)):
        key = 'skids[%i]' % i
        get_names_postdata[key] = x[i]

    names = remote_instance.fetch(remote_get_names_url, get_names_postdata)

    logger.debug(
        'Names for %i of %i skeleton IDs retrieved' % (len(names), len(x)))

    return(names)


def get_node_details(x, remote_instance=None, chunk_size=10000):
    """ Retrieve detailed treenode info for a list of treenodes and/or
    connectors.

    Parameters
    ----------
    x :                 list | CatmaidNeuron | CatmaidNeuronList
                        List of node ids: can be treenode or connector ids!
                        If CatmaidNeuron/List will get both, treenodes and
                        connectors!
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    chunk_size :        int, optional
                        Querying large number of nodes will result in server
                        errors. We will thus query them in amenable bouts.


    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a treenode:

        >>> df
        ...   node_id  creation_time  creator  edition_time
        ... 0
        ... 1
        ...   editor  reviewers  review_times
        ... 0
        ... 1

    """

    if isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        node_ids = np.append(x.nodes.treenode_id.values,
                             x.connectors.connector_id.values)
    elif not isinstance(x, (list, tuple, np.ndarray)):
        node_ids = [x]
    else:
        node_ids = x

    remote_instance = utils._eval_remote_instance(remote_instance)

    logger.debug(
        'Retrieving details for %i nodes...' % len(node_ids))

    remote_nodes_details_url = remote_instance._get_node_info_url()

    data = dict()

    urls = []
    post = []

    for ix in range(0, len(node_ids), chunk_size):
        urls.append(remote_instance._get_node_info_url())
        post.append({'node_ids[{}]'.format(k): tn for k, tn in  enumerate(node_ids[ix:ix + chunk_size])})

    # Get responses
    resp = remote_instance.fetch(urls, post, desc='Chunks')
    # Merge into a single dictionary
    data = {k: d[k] for d in resp for k in d}

    # Generate dataframe
    data_columns = ['creation_time', 'user', 'edition_time',
                    'editor', 'reviewers', 'review_times']
    df = pd.DataFrame(
        [[e] + [d[k] for k in data_columns] for e, d in data.items()],
        columns=['node_id'] + data_columns,
        dtype=object
    )

    # Rename column 'user' to 'creator'
    df.rename({'user': 'creator'}, axis='columns', inplace=True)

    df['creation_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['creation_time'].values]
    df['edition_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['edition_time'].values]
    df['review_times'] = [[datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in lst] for lst in df['review_times'].values]

    return df


def get_skid_from_treenode(treenode_ids, remote_instance=None):
    """ Retrieve skeleton IDs from a list of nodes.

    Parameters
    ----------
    treenode_ids :      int | list of int
                        Treenode ID(s) to retrieve skeleton IDs for.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
            ``{ treenode_ID : skeleton_ID, .... }``. If treenode does not
            exists, ``skeleton_ID`` will be ``None``.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    treenode_ids = utils.eval_node_ids(
        treenode_ids, connectors=False, treenodes=True)

    if not isinstance(treenode_ids, (list, np.ndarray)):
        treenode_ids = [treenode_ids]

    urls = [remote_instance._get_skid_from_tnid(tn) for tn in treenode_ids]

    data = remote_instance.fetch(urls, desc='Fetch skids')

    return {treenode_ids[i]: d.get('skeleton_id', None) for i, d in enumerate(data)}


def get_treenode_table(x, include_details=True, remote_instance=None):
    """ Retrieve treenode table(s) for a list of neurons.

    Parameters
    ----------
    x
                        Catmaid Neuron(s) as single or list of either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    include_details :   bool, optional
                        If True, tags and reviewer are included in the table.
                        For larger lists, it is recommended to set this to
                        False to improve performance.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a treenode:

        >>> df
        ...   skeleton_id  treenode_id  parent_id  confidence  x  y  z /
        ... 0
        ... 1
        ... 2
        ...
        ...   radius creator last_edition reviewers tag
        ... 0
        ... 1
        ... 2

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    logger.info(
        'Retrieving %i treenode table(s)...' % len(x))

    user_list = get_user_list(remote_instance)

    user_dict = user_list.set_index('id').T.to_dict()

    # Generate URLs to retrieve
    urls = []
    for skid in x:
        remote_nodes_list_url = remote_instance._get_skeleton_nodes_url(skid)
        urls.append(remote_nodes_list_url)

    node_list = remote_instance.fetch(urls, desc='Get tbls')

    logger.info(
        '%i treenodes retrieved. Creating table...' % sum([len(nl[0]) for nl in node_list]))

    all_tables = []

    for i, nl in enumerate(config.tqdm(node_list,
                                       desc='Creating table',
                                       leave=config.pbar_leave,
                                       disable=config.pbar_hide)):
        if include_details:
            tag_dict = {n[0]: [] for n in nl[0]}
            reviewer_dict = {n[0]: [] for n in nl[0]}
            [tag_dict[n[0]].append(n[1]) for n in nl[2]]
            [reviewer_dict[n[0]].append(user_list[user_list.id == n[1]]['login'].values[
                                        0]) for n in nl[1]]

            this_df = pd.DataFrame([[x[i]] + n + [reviewer_dict[n[0]], tag_dict[n[0]]] for n in nl[0]],
                                   columns=['skeleton_id', 'treenode_id', 'parent_node_id', 'confidence',
                                            'x', 'y', 'z', 'radius', 'creator', 'last_edited', 'reviewers', 'tags'],
                                   dtype=object
                                   )
        else:
            this_df = pd.DataFrame([[x[i]] + n for n in nl[0]],
                                   columns=['skeleton_id', 'treenode_id', 'parent_node_id', 'confidence',
                                            'x', 'y', 'z', 'radius', 'creator', 'last_edited'],
                                   dtype=object
                                   )

        # Replace creator_id with their login
        this_df['creator'] = [user_dict[u]['login']
                              for u in this_df['creator']]

        # Replace timestamp with datetime object
        this_df['last_edited'] = [datetime.datetime.fromtimestamp(
            t, tz=datetime.timezone.utc) for t in this_df['last_edited']]

        all_tables.append(this_df)

    tn_table = pd.concat(all_tables, axis=0)

    return tn_table


def get_edges(x, remote_instance=None):
    """ Retrieve edges (synaptic connections only!) between sets of neurons.

    Parameters
    ----------
    x
                        Neurons for which to retrieve edges. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents an edge:

        >>> df
        ...   source_skid     target_skid     weight
        ... 1
        ... 2
        ... 3

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_edges_url = remote_instance._get_edges_url()

    get_edges_postdata = {}
    get_edges_postdata['confidence_threshold'] = '0'

    for i in range(len(x)):
        key = 'skeleton_ids[%i]' % i
        get_edges_postdata[key] = x[i]

    edges = remote_instance.fetch(remote_get_edges_url, get_edges_postdata)

    df = pd.DataFrame([[e[0], e[1], sum(e[2])] for e in edges['edges']],
                      columns=['source_skid', 'target_skid', 'weight']
                      )

    return df


def get_connectors(x, relation_type=None, tags=None, remote_instance=None):
    """ Retrieve connectors based on a set of filters.

    Parameters
    ----------
    x
                        Neurons for which to retrieve connectors. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
                        5. ``None`` if you want all fetch connectors that
                           match other criteria
    relation_type :     'presynaptic_to' | 'postsynaptic_to' | 'gapjunction_with' | 'abutting' | 'attached_to', optional
                        If provided, will filter for these connection types.
    tags :              str | list of str, optional
                        If provided, will filter connectors for tag(s).
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector:

        >>> df
        ...   connector_id  x  y  z  confidence  creator,
        ... 0
        ... 1
        ...
        ...   editor  creation_time  edition_time
        ... 0
        ... 1

    See Also
    --------
    :func:`~pymaid.get_connector_details`
        If you need details about the connectivity of a connector
    :func:`~pymaid.get_connectors_between`
        If you need to find the connectors between sets of neurons.
    :func:`~pymaid.get_connector_links`
        If you ned details about links for each connector.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, type(None)):
        x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_connectors_url = remote_instance._get_connectors_url()

    postdata = {'with_tags': 'true', 'with_partners': 'true'}

    # Add skeleton IDs filter (if applicable)
    if not isinstance(x, type(None)):
        postdata.update(
            {'skeleton_ids[{0}]'.format(i): s for i, s in enumerate(x)})

    # Add tags filter (if applicable)
    if not isinstance(tags, type(None)):
        if not isinstance(tags, (list, np.ndarray)):
            tags = [tags]
        postdata.update({'tags[{0}]'.format(i): str(t)
                         for i, t in enumerate(tags)})

    # Add relation_type filter (if applicable)
    allowed_relations = ['presynaptic_to', 'postsynaptic_to',
                         'gapjunction_with', 'abutting', 'attached_to']
    if not isinstance(relation_type, type(None)):
        if relation_type not in allowed_relations:
            raise ValueError('Unknown relation type "{0}". Must be in {1}'.format(
                relation_type, allowed_relations))
        postdata.update({'relation_type': relation_type})

    data = remote_instance.fetch(remote_get_connectors_url, post=postdata)

    # creator_id and editor_id will be replaced with logins later
    df = pd.DataFrame(data=data['connectors'],
                      columns=['connector_id', 'x', 'y', 'z', 'confidence',
                               'creator_id', 'editor_id', 'creation_time',
                               'edition_time'])

    # Add tags
    df['tags'] = df.connector_id.map(data['tags'])

    # Hard-wired connector types. Use this to check for what's on the server:
    # types = remote_instance.fetch(remote_instance._get_connector_types_url())
    rel_ids = {8: {'relation': 'postsynaptic_to', 'type': 'synaptic'},
               14: {'relation': 'presynaptic_to', 'type': 'synaptic'},
               54650: {'relation': 'abutting', 'type': 'abutting'},
               686364: {'relation': 'gapjunction_with', 'type': 'gap_junction'},
               # ATTENTION: "attachment" can be part of any connector type
               5989640: {'relation': 'attached_to', 'type': 'attachment'},
               6216812: {'relation': 'close_to', 'type': 'close_to'}
               }

    # Get connector type IDs
    cn_ids = {k: v[0][-2] for k, v in data['partners'].items()}

    # Map type ID to relation (also note conversion of connector ID to integer)
    cn_type = {int(k): rel_ids.get(v, {'type': 'unknown'})['type']
               for k, v in cn_ids.items()}

    # Map connector ID to connector type
    df['type'] = df.connector_id.map(cn_type)

    # Add creator login instead of id
    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()
    df['creator'] = df.creator_id.map(user_dict)
    df['editor'] = df.editor_id.map(user_dict)
    df.drop(['creator_id', 'editor_id'], inplace=True, axis=1)

    # Convert timestamps to datetimes
    df['creation_time'] = df['creation_time'].apply(
        datetime.datetime.fromtimestamp)
    df['edition_time'] = df['edition_time'].apply(
        datetime.datetime.fromtimestamp)

    df.datatype = 'connector_table'

    return df


def get_connector_links(x, with_tags=False, chunk_size=50, remote_instance=None):
    """ Retrieve connectors links for a set of neurons.

    Note
    ----
    In essence, this will get you all "arrows" that point from a connector to
    your neuron or from your neuron to a connector. It does NOT give you the
    entire battery of connectors for a set of connectors. For that you have
    to use :func:`~pymaid.get_connector_details`.

    Parameters
    ----------
    x :                 int | CatmaidNeuron | CatmaidNeuronList
                        Neurons/Skeleton IDs to retrieve link details for. If
                        CatmaidNeuron/List will respect changes made to
                        original neurons (e.g. pruning)!
    with_tags :         bool, optional
                        If True will also return dictionary of connector tags.
    chunk_size :        int, optional
                        Neurons are split into chunks of this size and then
                        queried sequentially to prevent server from returning
                        an error.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector link:

        >>> df
        ...   skeleton_id  relation  connector_id  x  y  z  confidence
        ... 0
        ... 1
        ... 2
        ...
        ...   creator_id  treenode_id  creation_time  edition_time
        ... 0
        ... 1
        ... 2

    (links, tags)
            If ``with_tags=True``, will return above DataFrame and tags dict.

    See Also
    --------
    :func:`~pymaid.get_connectors`
        If you just need the connector table (ID, x, y, z, creator, etc).
    :func:`~pymaid.get_connector_details`
        Get the same data but by connector, not by link.
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, warn_duplicates=False,
                             remote_instance=remote_instance)

    df_collection = []
    tags = {}
    with config.tqdm(desc='Fetching links', total=len(skids),
                     disable=config.pbar_hide, leave=config.pbar_leave) as pbar:
        for chunk in [skids[i:i+chunk_size] for i in range(0, len(skids), chunk_size)]:
            # Generate URLs
            GET = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(chunk)}
            link_types = ['presynaptic_to', 'postsynaptic_to',
                          'abutting', 'gapjunction_with']
            urls = [remote_instance._get_connector_links_url(relation_type=cn,
                                                             **GET) for cn in link_types]

            # Fetch data
            responses = remote_instance.fetch(urls, disable_pbar=True)

            # Extract tags
            if with_tags:
                for r in responses:
                    tags.update(r['tags'])

            # Generate separate DataFrames
            data = [pd.DataFrame(r['links'],
                                 columns=['skeleton_id', 'connector_id',
                                          'x', 'y', 'z', 'confidence',
                                          'creator_id', 'treenode_id',
                                          'creation_time', 'edition_time']
                                ) for r in responses]

            # Add link type to each DataFrame
            for t, d in zip(link_types, data):
                d['relation'] = t

            # Concatenate DataFrames
            df = pd.concat(data, axis=0)

            # Store
            df_collection.append(df)

            # Update progress bar
            pbar.update(len(chunk))

    # Merge DataFrames
    df = pd.concat(df_collection, axis=0)

    # Cater for cases in which the original neurons have been edited
    if isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        df = df[df.connector_id.isin(x.connectors)]

    # Convert to timestamps
    df['creation_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['creation_time'].values]
    df['edition_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['edition_time'].values]

    if with_tags:
        return df, tags

    return df


def get_connector_details(x, remote_instance=None):
    """ Retrieve details on sets of connectors.

    Parameters
    ----------
    x :                 list of connector IDs | CatmaidNeuron | CatmaidNeuronList
                        Connector ID(s) to retrieve details for. If
                        CatmaidNeuron/List, will use their connectors.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector:

        >>> df
        ...   connector_id  presynaptic_to  postsynaptic_to
        ... 0
        ... 1
        ... 2
        ...
        ...   presynaptic_to_node  postsynaptic_to_node
        ... 0
        ... 1
        ... 2

    See Also
    --------
    :func:`~pymaid.get_connectors`
        If you just need the connector table (ID, x, y, z, creator, etc).
    :func:`~pymaid.get_connector_links`
        Get the same data but by link, not by connector.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    connector_ids = utils.eval_node_ids(x, connectors=True, treenodes=False)

    connector_ids = list(set(connector_ids))

    remote_get_connectors_url = remote_instance._get_connector_details_url()

    # Depending on DATA_UPLOAD_MAX_NUMBER_FIELDS of your CATMAID server
    # (default = 1000), we have to cut requests into batches smaller than that
    DATA_UPLOAD_MAX_NUMBER_FIELDS = min(50000, len(connector_ids))

    connectors = []
    with config.tqdm(total=len(connector_ids), desc='CN details',
              disable=config.pbar_hide, leave=config.pbar_leave) as pbar:
        for b in range(0, len(connector_ids), DATA_UPLOAD_MAX_NUMBER_FIELDS):
            get_connectors_postdata = {}
            for i, s in enumerate(connector_ids[b:b + DATA_UPLOAD_MAX_NUMBER_FIELDS]):
                key = 'connector_ids[%i]' % i
                get_connectors_postdata[key] = s  # connector_ids[i]

            connectors += remote_instance.fetch(remote_get_connectors_url,
                                                get_connectors_postdata)

            pbar.update(DATA_UPLOAD_MAX_NUMBER_FIELDS)

    logger.info('Data for %i of %i unique connector IDs retrieved' % (
        len(connectors), len(set(connector_ids))))

    columns = ['connector_id', 'presynaptic_to', 'postsynaptic_to',
               'presynaptic_to_node', 'postsynaptic_to_node']

    df = pd.DataFrame([[cn[0]] + [cn[1][e] for e in columns[1:]] for cn in connectors],
                      columns=columns,
                      dtype=object
                      )

    return df


def get_connectors_between(a, b, directional=True, remote_instance=None):
    """ Retrieve connectors between sets of neurons.

    Important
    ---------
    This function does currently *not* return gap junctions between neurons.

    Notes
    -----
    Connectors can show up multiple times if it is connecting to more than one
    treenodes of the same neuron.

    Parameters
    ----------
    a,b
                        Neurons for which to retrieve connectors. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    directional :       bool, optional
                        If True, only connectors a -> b are listed,
                        otherwise it is a <-> b.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector:

        >>> df
        ...   connector_id  connector_loc  treenode1_id  source_neuron
        ... 0
        ... 1
        ... 2
        ...
        ...   confidence1  creator1 treenode1_loc treenode2_id  target_neuron
        ... 0
        ... 1
        ... 2
        ...
        ...  confidence2  creator2  treenode2_loc
        ... 0
        ... 1
        ... 2

    See Also
    --------
    :func:`~pymaid.get_edges`
        If you just need the number of synapses between neurons, this is much
        faster.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    a = utils.eval_skids(a, remote_instance=remote_instance)
    b = utils.eval_skids(b, remote_instance=remote_instance)

    if len(a) == 0:
        raise ValueError('No source neurons provided')

    if len(b) == 0:
        raise ValueError('No target neurons provided')

    post = {'relation': 'presynaptic_to'}
    post.update({'skids1[{0}]'.format(i): s for i, s in enumerate(a)})
    post.update({'skids2[{0}]'.format(i): s for i, s in enumerate(b)})

    url = remote_instance._get_connectors_between_url()

    data = remote_instance.fetch(url, post=post)

    if not directional:
        post['relation'] = 'postsynaptic_to'
        data += remote_instance.fetch(url, post=post)

    df = pd.DataFrame(data,
                      columns=['connector_id', 'connector_loc', 'treenode1_id',
                               'source_neuron', 'confidence1', 'creator1',
                               'treenode1_loc', 'treenode2_id',
                               'target_neuron', 'confidence2', 'creator2',
                               'treenode2_loc'])

    # Get user list and replace IDs with logins
    user_list = get_user_list(remote_instance=remote_instance).set_index('id')
    df['creator1'] = [user_list.loc[u, 'login'] for u in df.creator1.values]
    df['creator2'] = [user_list.loc[u, 'login'] for u in df.creator2.values]

    return df


def get_review(x, remote_instance=None):
    """ Retrieve review status for a set of neurons.

    Parameters
    ----------
    x
                        Neurons for which to get review status. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron:

        >>> df
           skeleton_id neuron_name total_node_count nodes_reviewed percent_reviewed
        0
        1
        2

    See Also
    --------
    :func:`~pymaid.get_review_details`
        Gives you review status for individual nodes of a given neuron.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_reviews_url = remote_instance._get_review_status_url()

    names = {}
    review_status = {}

    CHUNK_SIZE = 1000

    with config.tqdm(total=len(x), disable=config.pbar_hide, desc='Rev. status',
              leave=config.pbar_leave) as pbar:
        for j in range(0, len(x), CHUNK_SIZE):
            get_review_postdata = {}

            for i in range(j, min(j + CHUNK_SIZE, len(x))):
                key = 'skeleton_ids[%i]' % i
                get_review_postdata[key] = str(x[i])

            names.update(get_names(x[j:j + CHUNK_SIZE], remote_instance))

            review_status.update(remote_instance.fetch(
                remote_get_reviews_url, get_review_postdata))

            pbar.update(CHUNK_SIZE)

    df = pd.DataFrame([[s,
                        names[str(s)],
                        review_status[s][0],
                        review_status[s][1],
                        int(review_status[s][1] / review_status[s][0] * 100)
                        ] for s in review_status],
                      columns=['skeleton_id', 'neuron_name',
                               'total_node_count', 'nodes_reviewed',
                               'percent_reviewed']
                      )

    return df


def remove_annotations(x, annotations, remote_instance=None):
    """ Remove annotation(s) from a list of neuron(s).

    Parameters
    ----------
    x
                        Neurons to remove given annotation(s) from. Can be
                        either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    annotations :       list
                        Annotation(s) to remove from neurons.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Nothing

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    annotations = utils._make_iterable(annotations)

    # Translate into annotations ID
    an_list = get_annotation_list().set_index('annotation')

    an_ids = []
    for a in annotations:
        if a not in an_list.index:
            logger.warning(
                'Annotation {0} not found. Skipping.'.format(a))
            continue
        an_ids.append(an_list.loc[a, 'annotation_id'])

    remove_annotations_url = remote_instance._get_remove_annotations_url()

    remove_annotations_postdata = {}

    for i in range(len(x)):
        # This requires neuron IDs (skeleton ID + 1)
        key = 'entity_ids[%i]' % i
        remove_annotations_postdata[key] = str(int(x[i]) + 1)

    for i in range(len(an_ids)):
        key = 'annotation_ids[%i]' % i
        remove_annotations_postdata[key] = str(an_ids[i])

    if an_ids:
        resp = remote_instance.fetch(
            remove_annotations_url, remove_annotations_postdata)

        an_list = an_list.reset_index().set_index('annotation_id')

        if len(resp['deleted_annotations']) == 0:
            logger.info('No annotations removed.')

        for a in resp['deleted_annotations']:
            logger.info('Removed "{0}" from {1} entities ({2} uses left)'
                               .format(an_list.loc[int(a), 'annotation'],
                                       len(resp['deleted_annotations']
                                           [a]['targetIds']),
                                       resp['left_uses'][a]
                                       )
                               )
    else:
        logger.info('No annotations removed.')

    return


def add_annotations(x, annotations, remote_instance=None):
    """ Add annotation(s) to a list of neuron(s)

    Parameters
    ----------
    x
                        Neurons to add new annotation(s) to. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    annotations :       list
                        Annotation(s) to add to neurons.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Nothing

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    annotations = utils._make_iterable(annotations)

    add_annotations_url = remote_instance._get_add_annotations_url()

    add_annotations_postdata = {}

    for i in range(len(x)):
        key = 'entity_ids[%i]' % i
        add_annotations_postdata[key] = str(int(x[i]) + 1)

    for i in range(len(annotations)):
        key = 'annotations[%i]' % i
        add_annotations_postdata[key] = str(annotations[i])

    logger.info(remote_instance.fetch(
        add_annotations_url, add_annotations_postdata))

    return


def get_user_annotations(x, remote_instance=None):
    """ Retrieve annotations used by given user(s).

    Parameters
    ----------
    x
                        User(s) to get annotation for. Can be either:

                        1. single or list of user IDs
                        2. single or list of user login names
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame (df) in which each row represents a single annotation:

        >>> df
           annotation annotated_on times_used user_id annotation_id user_login
        0
        1

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    # Get user list
    user_list = get_user_list(remote_instance=remote_instance)

    try:
        ids = [int(e) for e in x]
    except BaseException:
        ids = [user_list.set_index('login').loc[e, 'user_id'] for e in x]

    # This works with neuron_id NOT skeleton_id
    # neuron_id can be requested via neuron_names
    url_list = list()
    postdata = list()

    iDisplayLength = 500

    for u in ids:
        url_list.append(remote_instance._get_annotation_table_url())
        postdata.append(dict(user_id=int(u),
                             iDisplayLength=iDisplayLength))

    # Get data
    annotations = [e['aaData'] for e in remote_instance.fetch(
                   url_list, post=postdata, desc='Get annot')]

    # Add user login
    for i, u in enumerate(ids):
        for an in annotations[i]:
            an.append(user_list.set_index('id').loc[u, 'login'])

    # Now flatten the list of lists
    annotations = [an for sublist in annotations for an in sublist]

    # Create dataframe
    df = pd.DataFrame(annotations,
                      columns=['annotation', 'annotated_on', 'times_used',
                               'user_id', 'annotation_id', 'user_login'],
                      dtype=object
                      )

    df['annotated_on'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['annotated_on'].values]

    return df.sort_values('times_used').reset_index(drop=True)


def get_annotation_details(x, remote_instance=None):
    """ Retrieve annotations for a set of neuron. Returns more
    details than :func:`~pymaid.get_annotations` but is slower.
    Contains timestamps and user IDs (same API as neuron navigator).

    Parameters
    ----------
    x
                        Neurons to get annotation details for. Can be either:

                        1. List of skeleton ID(s) (int or str)
                        2. List of neuron name(s) (str, exact match)
                        3. An annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a single annotation:

        >>> df
        ...   annotation skeleton_id time_annotated user_id annotation_id user
        ... 0
        ... 1

    See Also
    --------
    :func:`~pymaid.get_annotations`
                        Gives you annotations for a list of neurons (faster)

    Examples
    --------
    >>> # Get annotations for a set of neurons
    >>> an = pymaid.get_annotation_details([ 12, 57003 ])
    >>> # Get those for a single neuron
    >>> an[ an.skeleton_id == '57003' ]
    >>> # Get annotations given by set of users
    >>> an[ an.user.isin( ['schlegelp', 'lif'] )]
    >>> # Get most recent annotations
    >>> import datetime
    >>> an[ an.time_annotated > datetime.date(2017,6,1) ]

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    # This works with neuron_id NOT skeleton_id
    # neuron_id can be requested via neuron_names
    url_list = list()
    postdata = list()

    for s in skids:
        remote_get_neuron_name = remote_instance._get_single_neuronname_url(s)
        neuronid = remote_instance.fetch(remote_get_neuron_name)['neuronid']

        url_list.append(remote_instance._get_annotation_table_url())
        postdata.append(dict(neuron_id=int(neuronid)))

    # Get data
    annotations = [e['aaData'] for e in remote_instance.fetch(
                   url_list, post=postdata, desc='Get annot')]

    # Get user list
    user_list = get_user_list(remote_instance).set_index('id')

    # Add skeleton ID and user login
    for i, s in enumerate(skids):
        for an in annotations[i]:
            an.insert(1, s)
            an.append(user_list.loc[an[4], 'login'])

    # Now flatten the list of lists
    annotations = [an for sublist in annotations for an in sublist]

    # Create dataframe
    df = pd.DataFrame(annotations,
                      columns=['annotation', 'skeleton_id', 'time_annotated',
                               'times_used', 'user_id', 'annotation_id',
                               'user'],
                      dtype=object
                      )

    # Times used appears to not be working (always shows "1") - remove it
    df.drop('times_used', inplace=True, axis=1)

    df['time_annotated'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['time_annotated'].values]

    return df.sort_values('annotation').reset_index(drop=True)


def get_annotations(x, remote_instance=None):
    """ Retrieve annotations for a list of skeleton ids.
    If a neuron has no annotations, it will not show up in returned dict!

    Notes
    -----
    This API endpoint does not process more than 250 neurons at a time!

    Parameters
    ----------
    x
                        Neurons for which to retrieve annotations. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        ``{ skeleton_id : [ annnotation, annotation ], ... }``

    See Also
    --------
    :func:`~pymaid.get_annotation_details`
                        Gives you more detailed information about annotations
                        of a set of neuron (includes timestamp and user) but
                        is slower.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_annotations_url = remote_instance._get_annotations_for_skid_list2()

    get_annotations_postdata = {'metaannotations': 0, 'neuronnames': 0}

    for i in range(len(x)):
        #key = 'x[%i]' % i
        key = 'skeleton_ids[%i]' % i
        get_annotations_postdata[key] = str(x[i])

    annotation_list_temp = remote_instance.fetch(
        remote_get_annotations_url, get_annotations_postdata)

    annotation_list = {}

    try:
        for skid in annotation_list_temp['skeletons']:
            annotation_list[skid] = []
            # for entry in annotation_list_temp['skeletons'][skid]:
            for entry in annotation_list_temp['skeletons'][skid]['annotations']:
                annotation_id = entry['id']
                annotation_list[skid].append(
                    annotation_list_temp['annotations'][str(annotation_id)])

        return(annotation_list)
    except BaseException:
        logger.error(
            'No annotations retrieved. Make sure that the skeleton IDs exist.')
        raise Exception(
            'No annotations retrieved. Make sure that the skeleton IDs exist.')


def get_annotation_id(annotations, remote_instance=None, allow_partial=False):
    """ Retrieve the annotation ID for single or list of annotation(s).

    Parameters
    ----------
    annotations :       str | list of str
                        Single annotations or list of multiple annotations.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    allow_partial :     bool
                        If True, will allow partial matches.

    Returns
    -------
    dict
                        ``{'annotation_name' : 'annotation_id', ....}``

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    logger.debug('Retrieving list of annotations...')

    remote_annotation_list_url = remote_instance._get_annotation_list()
    an_list = remote_instance.fetch(remote_annotation_list_url)

    # Turn into pandas array
    an_list = pd.DataFrame.from_records(an_list['annotations'])

    annotations = utils._make_iterable(annotations)
    annotation_ids = {}
    for an in annotations:
        # '/' indicates regex
        if an.startswith('/'):
            re_str = an[1:]
        # If allow partial just use the raw string
        elif allow_partial:
            re_str = an
        # If exact match, encode this in regex
        else:
            re_str = '^{}$'.format(an)

        # Search for matches
        res = an_list[an_list.name.str.match(re_str)].set_index('name').id.to_dict()
        if not res:
            logger.warning('No annotation found for "{}"'.format(an))
        annotation_ids.update(res)

    return annotation_ids


def has_soma(x, remote_instance=None, tag='soma', min_rad=500):
    """ Check if a neuron/a list of neurons have somas.

    Parameters
    ----------
    x
                        Neurons which to check for a soma. Can be either:

                        1. skeleton ID(s) (int or str)
                        2. neuron name(s) (str)
                        3. annotation(s): e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object

    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    tag :               str | None, optional
                        Tag we expect the soma to have. Set to ``None`` if
                        not applicable.
    min_rad :           int, optional
                        Minimum radius of soma.

    Returns
    -------
    dict
                        ``{'skid1': True, 'skid2': False, ...}``

    Note
    ----
    There is no shortcut to get this information - we have to load the 3D
    skeleton to get the soma. If you need the 3D skeletons anyway, it is more
    efficient to use :func:`~pymaid.get_neuron` to get a neuronlist and then
    use the :attr:`~pymaid.CatmaidNeuronList.soma` attribute.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    skdata = get_neuron(x,
                        remote_instance=remote_instance,
                        connector_flag=0, tag_flag=1,
                        get_history=False,
                        return_df=True  # no need to make proper neurons
                        )

    d = {}
    for s in skdata.itertuples():
        if tag:
            if tag in s.tags:
                tn_with_tag = s.tags['soma']
            else:
                tn_with_tag = []
        else:
            tn_with_tag = s.nodes.treenode_id.values

        tn_with_rad = s.nodes[s.nodes.radius > min_rad].treenode_id.values

        if set(tn_with_tag) & set(tn_with_rad):
            d[s.skeleton_id] = True
        else:
            d[s.skeleton_id] = False

    return d


def get_skids_by_name(names, remote_instance=None, allow_partial=True):
    """ Retrieve the all neurons with matching name.

    Parameters
    ----------
    names :             str | list of str
                        Name(s) to search for.
    allow_partial :     bool, optional
                        If True, partial matches are returned too.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron:

        >>> df
           name   skeleton_id
        0
        1
        2

    See Also
    --------
    :func:`pymaid.find_neurons`
                            Use to retrieve neurons by combining various
                            search criteria. For example names, reviewers,
                            annotations, etc.
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    # Prepare names for regex search on the backend
    re_str = []
    for n in utils._make_iterable(names, force_type=str):
        # '/' indicates regex - not necessary for the backend, so remove
        if n.startswith('/'):
            re_str.append(n[1:])
        # If allow partial just use the raw string
        elif allow_partial:
            re_str.append(n)
        # If exact match, encode this in regex
        else:
            re_str.append('^{}$'.format(n))

    urls = [remote_instance._get_annotated_url() for n in re_str]
    POST = [{'name': n, 'with_annotations': False} for n in re_str]
    responses = remote_instance.fetch(urls, post=POST, desc='Get nms')

    df = pd.DataFrame([[n['name'], n['skeleton_ids'][0]] for res in responses for
                        n in res['entities'] if n['type'] == 'neuron'],
                      columns=['name', 'skeleton_id'])

    return df.sort_values(['name']).drop_duplicates().reset_index(drop=True)


def get_skids_by_annotation(annotations, remote_instance=None,
                            allow_partial=False, intersect=False):
    """ Retrieve the all neurons annotated with given annotation(s).

    Parameters
    ----------
    annotations :           str | list
                            Single annotation or list of multiple annotations.
    remote_instance :       CATMAID instance, optional
                            If not passed directly, will try using global.
    allow_partial :         bool, optional
                            If True, allow partial match of annotation.
    intersect :             bool, optional
                            If True, neurons must have ALL provided annotations.

    Returns
    -------
    list
                            ``[skid1, skid2, skid3]``

    See Also
    --------
    :func:`pymaid.find_neurons`
                            Use to retrieve neurons by combining various
                            search criteria. For example names, reviewers,
                            annotations, etc.
    """

    """
    logger.warning(
            "Deprecationwarning: get_skids_by_annotation() is deprecated, use find_neurons() instead."
        )
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    logger.debug(
        'Looking for annotation(s): ' + str(annotations))

    if intersect and utils._is_iterable(annotations):
        current_level = logger.level
        logger.setLevel('WARNING')
        skids = set.intersection(*[set(get_skids_by_annotation(a,
                                                               remote_instance=remote_instance,
                                                               allow_partial=allow_partial,
                                                               intersect=False))
                                   for a in annotations])
        logger.setLevel(current_level)
        logger.info(
            'Found %i skeletons with matching annotation(s)' % len(skids))
        return list(skids)

    annotation_ids = get_annotation_id(
        annotations, remote_instance, allow_partial=allow_partial)

    if not annotation_ids:
        logger.error(
            'No matching annotation found! Returning None')
        raise Exception('No matching annotation found!')

    if allow_partial is True:
        logger.debug(
            'Found id(s): %s (partial matches included)' % len(annotation_ids))
    elif utils._is_iterable(annotations):
        logger.debug('Found id(s): %s | Unable to retrieve: %i' % (
            str(annotation_ids), len(annotations) - len(annotation_ids)))
    elif isinstance(annotations, str):
        logger.debug('Found id: %s | Unable to retrieve: %i' % (
            list(annotation_ids.keys())[0], 1 - len(annotation_ids)))

    annotated_skids = []
    logger.debug(
        'Retrieving skids for annotationed neurons...')
    for an_id in annotation_ids.values():
        #annotation_post = {'neuron_query_by_annotation': annotation_id, 'display_start': 0, 'display_length':500}
        annotation_post = {'annotated_with0': an_id, 'with_annotations': False}
        remote_annotated_url = remote_instance._get_annotated_url()
        neuron_list = remote_instance.fetch(
            remote_annotated_url, annotation_post)
        for entry in neuron_list['entities']:
            if entry['type'] == 'neuron':
                annotated_skids.append(str(entry['skeleton_ids'][0]))

    annotated_skids = list(set(annotated_skids))

    logger.info(
        'Found %i skeletons with matching annotation(s)' % len(annotated_skids))

    return annotated_skids


def neuron_exists(x, remote_instance=None):
    """ Check if neurons exist in CATMAID.

    Parameters
    ----------
    x
                        Neurons to check if they exist in Catmaid. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    bool :
                        True if skeleton exists, False if not. If multiple
                        neurons are queried, returns a dict
                        ``{ skid1 : True, skid2 : False, ... }``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    if len(x) > 1:
        return {n: neuron_exists(n) for n in x}
    else:
        x = x[0]

    remote_get_neuron_name = remote_instance._get_single_neuronname_url(x)
    response = remote_instance.fetch(remote_get_neuron_name)

    if 'error' in response:
        return False
    else:
        return True


def get_treenode_info(x, remote_instance=None):
    """ Retrieve info for a set of treenodes.

    Parameters
    ----------
    x                   CatmaidNeuron | CatmaidNeuronList | list of treenode IDs
                        Single or list of treenode IDs. If CatmaidNeuron/List,
                        details for all it's treenodes are requested.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas DataFrame
                DataFrame in which each row represents a queried treenode:

                >>> df
                  treenode_id neuron_name skeleton_id skeleton_name neuron_id
                0
                1

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    treenode_ids = utils.eval_node_ids(x, connectors=False, treenodes=True)

    urls = [remote_instance._get_treenode_info_url(tn) for tn in treenode_ids]

    data = remote_instance.fetch(urls, desc='Get info')

    df = pd.DataFrame([[treenode_ids[i]] + list(n.values()) for i, n in enumerate(data)],
                      columns=['treenode_id'] + list(data[0].keys())
                      )

    return df


def get_node_tags(node_ids, node_type, remote_instance=None):
    """ Retrieve tags for a set of treenodes.

    Parameters
    ----------
    node_ids
                        Single or list of treenode or connector IDs.
    node_type :         'TREENODE' | 'CONNECTOR'
                        Set which node type of IDs you have provided as they
                        use different API endpoints!
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                dictionary containing tags for each node

    Examples
    --------
    >>> pymaid.get_node_tags( ['6626578', '6633237']
    ...                        'TREENODE',
    ...                        remote_instance )
    {'6633237': ['ends'], '6626578': ['ends'] }

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(node_ids, (list, np.ndarray)):
        node_ids = [node_ids]

    # Make sure node_ids are strings
    node_ids = [str(n) for n in node_ids]

    url = remote_instance._get_node_labels_url()

    if node_type in ['TREENODE', 'TREENODES']:
        key = 'treenode_ids'
    elif node_type in ['CONNECTOR', 'CONNECTORS']:
        key = 'connector_ids'
    else:
        raise TypeError('Unknown node_type parameter: %s' % str(node_type))

    POST = {key: ','.join([str(tn) for tn in node_ids])}

    return remote_instance.fetch(url, post=POST)


def delete_neuron(x, no_prompt=False, remote_instance=None):
    """ Completely delete neurons. Use this with EXTREME caution
    as this is irreversible!

    Important
    ---------
    Deletes a neuron if (and only if!) two things are the case:

     1. You own all treenodes of the skeleton making up the neuron in question
     2. The neuron is not annotated by other users

    Parameters
    ----------
    x
                        Neurons to check if they exist in Catmaid. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    no_prompt :         bool, optional
                        By default, you will be prompted to confirm before
                        deleting the neuron(s). Set this to True to skip that
                        step.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    server response

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    if len(x) > 1:
        return {n: delete_neuron(n, remote_instance=remote_instance)
                for n in x}
    else:
        x = x[0]

    # Need to get the neuron ID
    remote_get_neuron_name = remote_instance._get_single_neuronname_url(x)
    neuronid = remote_instance.fetch(remote_get_neuron_name)['neuronid']

    if not no_prompt:
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("Please confirm deletion [Y/N] ").lower()

        if answer != 'y':
            return

    url = remote_instance._delete_neuron_url(neuronid)

    return remote_instance.fetch(url)


def delete_tags(node_list, tags, node_type, remote_instance=None):
    """ Remove tag(s) for a list of treenode(s) or connector(s).
    Works by getting existing tags, removing given tag(s) and then using
    pymaid.add_tags() to push updated tags back to CATMAID.

    Parameters
    ----------
    node_list :         list
                        Treenode or connector IDs to delete tags from.
    tags :              list
                        Tags(s) to delete from provided treenodes/connectors.
                        Use ``tags=None`` and to remove all tags from a set of
                        nodes.
    node_type :         'TREENODE' | 'CONNECTOR'
                        Set which node type of IDs you have provided as they
                        use different API endpoints!
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    str
                        Confirmation from Catmaid server.

    See Also
    --------
    :func:`~pymaid.add_tags`
            Function to add tags to nodes.

    Examples
    --------
    Remove end-related tags from non-end treenodes

    >>> import pymaid
    >>> # Load neuron
    >>> n = pymaid.get_neuron( 16 )
    >>> # Get non-end nodes
    >>> non_leaf_nodes = n.nodes[n.nodes.type != 'end']
    >>> # Define which tags to remove
    >>> tags_to_remove = ['ends', 'uncertain end', 'uncertain continuation', 'TODO']
    >>> # Remove tags
    >>> resp = pymaid.delete_tags( non_leaf_nodes.treenode_id.values,
    ...                            tags_to_remove,
    ...                            'TREENODE')
    2017-08-09 14:08:36,102 - pymaid.pymaid - WARNING - Skipping 8527 nodes without tags
    >>> # Above warning means that most nodes did not have ANY tags

    """

    PERM_NODE_TYPES = ['TREENODE', 'CONNECTOR']

    if node_type not in PERM_NODE_TYPES:
        raise ValueError('Unknown node_type "{0}". Please use either: {1}'.format(node_type,
                                                                                  ','.join(
                                                                                      PERM_NODE_TYPES)
                                                                                  ))

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(node_list, (list, np.ndarray)):
        node_list = [node_list]

    # Make sure node list is strings
    node_list = [str(n) for n in node_list]

    if not isinstance(tags, (list, np.ndarray)):
        tags = [tags]

    if tags != [None]:
        # First, get existing tags for these nodes
        existing_tags = get_node_tags(
            node_list, node_type, remote_instance=remote_instance)

        # Check if our treenodes actually exist
        if [n for n in node_list if n not in existing_tags]:
            logger.warning('Skipping %i nodes without tags' % len(
                [n for n in node_list if n not in existing_tags]))
            [node_list.remove(n) for n in [
                n for n in node_list if n not in existing_tags]]

        # Remove tags from that list that we want to have deleted
        existing_tags = {n: [t for t in existing_tags[
            n] if t not in tags] for n in node_list}
    else:
        existing_tags = ''

    # Use the add_tags function to override existing tags
    return add_tags(node_list, existing_tags, node_type,
                    remote_instance=remote_instance, override_existing=True)


def add_tags(node_list, tags, node_type, remote_instance=None,
             override_existing=False):
    """ Add or edit tag(s) for a list of treenode(s) or connector(s).

    Parameters
    ----------
    node_list :         list
                        Treenode or connector IDs to edit.
    tags :              str | list | dict
                        Tags(s) to add to provided treenode/connector ids. If
                        a dictionary is provided `{node_id: [tag1, tag2], ...}`
                        each node gets individual tags. If string or list
                        are provided, all nodes will get the same tags.
    node_type :         'TREENODE' | 'CONNECTOR'
                        Set which node type of IDs you have provided as they
                        use different API endpoints!
    override_existing : bool, default = False
                        This needs to be set to True if you want to delete a
                        tag. Otherwise, your tags (even if empty) will not
                        override existing tags.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    str
                        Confirmation from Catmaid server

    Notes
    -----
    Use ``tags=''`` and ``override_existing=True`` to delete all tags from
    nodes.

    See Also
    --------
    :func:`~pymaid.delete_tags`
            Function to delete given tags from nodes.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(node_list, (list, np.ndarray)):
        node_list = [node_list]

    if not isinstance(tags, (list, np.ndarray, dict)):
        tags = [tags]

    if node_type in ['TREENODE', 'TREENODES']:
        add_tags_urls = [
            remote_instance._treenode_add_tag_url(n) for n in node_list]
    elif node_type in ['CONNECTOR', 'CONNECTORS']:
        add_tags_urls = [
            remote_instance._connector_add_tag_url(n) for n in node_list]
    else:
        raise TypeError('Unknown node_type parameter: %s' % str(node_type))

    if isinstance(tags, dict):
        post_data = [
            {'tags': ','.join(tags[n]), 'delete_existing': override_existing} for n in node_list]
    else:
        post_data = [
            {'tags': ','.join(tags), 'delete_existing': override_existing} for n in node_list]

    return remote_instance.fetch(add_tags_urls,
                                 post=post_data,
                                 desc='Modifying tags')


def get_segments(x, remote_instance=None):
    """ Retrieve list of segments for a neuron just like the review widget.

    Parameters
    -----------
    x
                        Neurons to retrieve. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    list
                List of treenode IDs, ordered by length. If multiple neurons
                are requested, returns a dict { skid : [], ... }

    See Also
    --------
    ``CatmaidNeuron.segments``
    ``CatmaidNeuron.short_segments``
                Use these :class:`pymaid.CatmaidNeuron` attributes to access
                segments generated by pymaid (faster).
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    urls = []
    post_data = []

    for s in x:
        urls.append(remote_instance._get_review_details_url(s))
        # For some reason this needs to fetched as POST (even though actual
        # POST data is not necessary)
        post_data.append({'placeholder': 0})

    rdata = remote_instance.fetch(urls, post=post_data, desc='Get segs')

    if len(x) > 1:
        return {x[i]: [[tn['id'] for tn in arb['sequence']] for arb in rdata[i]] for i in range(len(x))}
    else:
        return [[tn['id'] for tn in arb['sequence']] for arb in rdata[0]]


def get_review_details(x, remote_instance=None):
    """ Retrieve review status (reviewer + timestamp) for each node
    of a given skeleton. Uses the review API.

    Parameters
    -----------
    x
                        Neurons to get review-details for. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas DataFrame
        DataFrame in which each row respresents a node.

        >>> print(df)
        treenode_id  skeleton_id  reviewer1  reviewer2  reviewer 3
           12345       12345123     datetime    NaT      datetime

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    node_list = []
    urls = []
    post_data = []

    for s in x:
        urls.append(remote_instance._get_review_details_url(s))
        # For some reason this needs to fetched as POST (even though actual
        # POST data is not necessary)
        post_data.append({'placeholder': 0})

    rdata = remote_instance.fetch(urls,
                                  post=post_data,
                                  desc='Get rev stats')

    for i, neuron in enumerate(rdata):
        # There is a small chance that nodes are counted twice but not
        # tracking node_id speeds up this extraction a LOT
        # node_ids = []
        for arbor in neuron:
            node_list += [(n['id'], x[i], n['rids'])
                          for n in arbor['sequence'] if n['rids']]

    tn_to_skid = {n[0]: n[1] for n in node_list}
    node_dict = {n[0]: {u[0]: datetime.datetime.strptime(
        u[1][:16], '%Y-%m-%dT%H:%M') for u in n[2]} for n in node_list}

    user_list = get_user_list(remote_instance=remote_instance).set_index('id')

    df = pd.DataFrame.from_dict(node_dict, orient='index').fillna(np.nan)
    df.columns = [user_list.loc[u, 'login'] for u in df.columns]
    df['skeleton_id'] = [tn_to_skid[tn] for tn in df.index.values]
    df.index.name = 'treenode_id'
    df = df.reset_index(drop=False)

    # Make sure we didn't count treenodes twice
    df = df[~df.duplicated('treenode_id')]

    return df


def get_logs(remote_instance=None, operations=[], entries=50, display_start=0,
             search=""):
    """ Retrieve logs (same data as in log widget).

    Parameters
    ----------
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    operations :        list of str, optional
                        If empty, all operations will be queried from server
                        possible operations: 'join_skeleton',
                        'change_confidence', 'rename_neuron', 'create_neuron',
                        'create_skeleton', 'remove_neuron', 'split_skeleton',
                        'reroot_skeleton', 'reset_reviews', 'move_skeleton'
    entries :           int, optional
                        Number of entries to retrieve.
    display_start :     int, optional
                        Sets range of entries to return:
                        ``display_start`` to ``display_start + entries``.
    search :            str, optional
                        Use to filter results for e.g. a specific skeleton ID
                        or neuron name.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a single operation:

        >>> df
            user   operation   timestamp   x   y   z   explanation
        0
        1

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not operations:
        operations = [-1]
    elif not isinstance(operations, (list, np.ndarray)):
        operations = [operations]

    logs = []
    for op in operations:
        get_logs_postdata = {'sEcho': 6,
                             'iColumns': 7,
                             'iDisplayStart': display_start,
                             'iDisplayLength': entries,
                             'mDataProp_0': 0,
                             'sSearch_0': '',
                             'bRegex_0': False,
                             'bSearchable_0': False,
                             'bSortable_0': True,
                             'mDataProp_1': 1,
                             'sSearch_1': '',
                             'bRegex_1': False,
                             'bSearchable_1': False,
                             'bSortable_1': True,
                             'mDataProp_2': 2,
                             'sSearch_2': '',
                             'bRegex_2': False,
                             'bSearchable_2': False,
                             'bSortable_2': True,
                             'mDataProp_3': 3,
                             'sSearch_3': '',
                             'bRegex_3': False,
                             'bSearchable_3': False,
                             'bSortable_3': False,
                             'mDataProp_4': 4,
                             'sSearch_4': '',
                             'bRegex_4': False,
                             'bSearchable_4': False,
                             'bSortable_4': False,
                             'mDataProp_5': 5,
                             'sSearch_5': '',
                             'bRegex_5': False,
                             'bSearchable_5': False,
                             'bSortable_5': False,
                             'mDataProp_6': 6,
                             'sSearch_6': '',
                             'bRegex_6': False,
                             'bSearchable_6': False,
                             'bSortable_6': False,
                             'sSearch': '',
                             'bRegex': False,
                             'iSortCol_0': 2,
                             'sSortDir_0': 'desc',
                             'iSortingCols': 1,
                             'self.project_id': remote_instance.project_id,
                             'operation_type': op,
                             'search_freetext': search}

        remote_get_logs_url = remote_instance._get_logs_url()
        logs += remote_instance.fetch(remote_get_logs_url,
                                      get_logs_postdata)['aaData']

    df = pd.DataFrame(logs,
                      columns=['user', 'operation', 'timestamp',
                               'x', 'y', 'z', 'explanation']
                      )

    df['timestamp'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['timestamp'].values]

    return df


def get_contributor_statistics(x, remote_instance=None, separate=False,
                               max_threads=500):
    """ Retrieve contributor statistics for given skeleton ids.
    By default, stats are given over all neurons.

    Parameters
    ----------
    x
                        Neurons to get contributor stats for. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    separate :          bool, optional
                        If true, stats are given per neuron.
    max_threads :       int, optional
                        Maximum parallel data requests. Overrides
                        ``CatmaidInstance.max_threads``.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Series, if ``separate=False``. DataFrame, if ``separate=True``:

        >>> df
           skeleton_id  node_contributors  multiuser_review_minutes  ..
        1
        2
        3
           post_contributors  construction_minutes  min_review_minutes  ..
        1
        2
        3
           n_postsynapses  n_presynapses  pre_contributors  n_nodes  review_contributors
        1
        2
        3

    Examples
    --------
    >>> # Plot contributions as pie chart
    >>> import matplotlib.pyplot as plt
    >>> cont = pymaid.get_contributor_statistics( "annotation:uPN right" )
    >>> plt.subplot(131, aspect=1)
    >>> ax1 = plt.pie(  cont.node_contributors.values(),
                        labels=cont.node_contributors.keys(),
                        autopct='%.0f%%' )
    >>> plt.subplot(132, aspect=1)
    >>> ax2 = plt.pie(  cont.pre_contributors.values(),
                        labels=cont.pre_contributors.keys(),
                        autopct='%.0f%%' )
    >>> plt.subplot(133, aspect=1)
    >>> ax3 = plt.pie(  cont.post_contributors.values(),
                        labels=cont.post_contributors.keys(),
                        autopct='%.0f%%' )
    >>> plt.show()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    columns = ['skeleton_id', 'n_nodes', 'node_contributors', 'n_presynapses',
               'pre_contributors', 'n_postsynapses', 'post_contributors',
               'review_contributors', 'multiuser_review_minutes',
               'construction_minutes', 'min_review_minutes']

    user_list = get_user_list(remote_instance=remote_instance).set_index('id')

    if not separate:
        with config.tqdm(total=len(x), desc='Contr. stats', disable=config.pbar_hide,
                  leave=config.pbar_leave) as pbar:
            stats = []
            for j in range(0, len(x), max_threads):
                pbar.update(j)
                get_statistics_postdata = {}

                for i in range(j, min(len(x), j + max_threads)):
                    key = 'skids[%i]' % i
                    get_statistics_postdata[key] = x[i]

                remote_get_statistics_url = remote_instance._get_contributions_url()
                stats.append(remote_instance.fetch(
                    remote_get_statistics_url, get_statistics_postdata))

        # Now generate DataFrame
        node_contributors = {user_list.loc[int(u), 'login']: sum([st['node_contributors'][u] for st in stats if u in st[
            'node_contributors']]) for st in stats for u in st['node_contributors']}
        pre_contributors = {user_list.loc[int(u), 'login']: sum([st['pre_contributors'][u] for st in stats if u in st[
            'pre_contributors']]) for st in stats for u in st['pre_contributors']}
        post_contributors = {user_list.loc[int(u), 'login']: sum([st['post_contributors'][u] for st in stats if u in st[
            'post_contributors']]) for st in stats for u in st['post_contributors']}
        review_contributors = {user_list.loc[int(u), 'login']: sum([st['review_contributors'][u] for st in stats if u in st[
            'review_contributors']]) for st in stats for u in st['review_contributors']}

        df = pd.Series([
            x,
            sum([st['n_nodes'] for st in stats]),
            node_contributors,
            sum([st['n_pre'] for st in stats]),
            pre_contributors,
            sum([st['n_post'] for st in stats]),
            post_contributors,
            review_contributors,
            sum([st['multiuser_review_minutes'] for st in stats]),
            sum([st['construction_minutes'] for st in stats]),
            sum([st['min_review_minutes'] for st in stats])
        ],
            index=columns,
            dtype=object
        )
    else:
        get_statistics_postdata = [{'skids[0]': s} for s in x]
        remote_get_statistics_url = [
            remote_instance._get_contributions_url() for s in x]

        stats = remote_instance.fetch(remote_get_statistics_url,
                                      post=get_statistics_postdata,
                                      desc='Get contrib.')

        df = pd.DataFrame([[
            s,
            stats[i]['n_nodes'],
            {user_list.loc[int(u), 'login']: stats[i]['node_contributors'][u]
                for u in stats[i]['node_contributors']},
            stats[i]['n_pre'],
            {user_list.loc[int(u), 'login']: stats[i]['pre_contributors'][u]
                for u in stats[i]['pre_contributors']},
            stats[i]['n_post'],
            {user_list.loc[int(u), 'login']: stats[i]['post_contributors'][u]
                for u in stats[i]['post_contributors']},
            {user_list.loc[int(u), 'login']: stats[i]['review_contributors'][u]
                for u in stats[i]['review_contributors']},
            stats[i]['multiuser_review_minutes'],
            stats[i]['construction_minutes'],
            stats[i]['min_review_minutes']
        ] for i, s in enumerate(x)],
            columns=columns,
            dtype=object
        )
    return df


def get_neuron_list(remote_instance=None, user=None, node_count=1,
                    start_date=[], end_date=[], reviewed_by=None,
                    minimum_cont=None):
    """ Retrieves a list of all skeletons that fit given parameters (see
    variables). If no parameters are provided, all existing skeletons are
    returned!

    Parameters
    ----------
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.
    user :              int | str | list, optional
                        User ID(s) (int) or login(s) (str).
    minimum_cont :      int, optional
                        Minimum contribution (in nodes) to a neuron in order
                        for it to be counted. Only applicable if ``user`` is
                        provided. If multiple users are provided contribution
                        is calculated across all users. Minimum contribution
                        does NOT take start and end dates into account!
    node_count :        int, optional
                        Minimum size of returned neuron (number of nodes).
    start_date :        datetime | list of integers, optional
                        If list: ``[year, month, day]``
                        Only consider neurons created after.
    end_date :          datetime | list of integers, optional
                        If list: ``[year, month, day]``
                        Only consider neurons created before.
    reviewed_by :       int | str | list, optional
                        User ID(s) (int) or login name(s) (str) of reviewer.

    Returns
    -------
    list
                        ``[skid, skid, skid, ... ]``

    Examples
    --------
    Get all neurons a given users have worked on in the last week. This example
    assumes that you have already set up a CATMAID instance.

    >>> # We are using user IDs but you can also use login names
    >>> import datetime
    >>> last_week = datetime.date.today() - datetime.date.timedelta(days=7)
    >>> skids = pymaid.get_neuron_list( user = [16,20],
    ...                                 start_date = last_week,
    ...                                 remote_instance = remote_instance )

    """

    def _contribution_helper(skids):
        """ Helper to test if users have contributed more than X nodes to
        neuron.

        Returns
        -------
        Filtered list of skeleton IDs
        """
        nl = get_neuron(skids, remote_instance=remote_instance, return_df=True)
        return [n.skeleton_id for n in nl.itertuples() if n.nodes[n.nodes.creator_id.isin(user)].shape[0] > minimum_cont]

    remote_instance = utils._eval_remote_instance(remote_instance)

    get_skeleton_list_GET_data = {'nodecount_gt': node_count}

    if user:
        user = utils.eval_user_ids(
            user, user_list=None, remote_instance=remote_instance)
    if reviewed_by:
        reviewed_by = utils.eval_user_ids(
            reviewed_by, user_list=None, remote_instance=remote_instance)

    if not isinstance(user, type(None)) or not isinstance(reviewed_by,
                                                          type(None)):
        user_list = get_user_list(
            remote_instance=remote_instance).set_index('login')

    if not isinstance(user, type(None)):
        if utils._is_iterable(user):
            skid_list = list()
            for u in config.tqdm(user, desc='Get user', disable=config.pbar_hide,
                          leave=config.pbar_leave):
                skid_list += get_neuron_list(remote_instance=remote_instance,
                                             user=u,
                                             node_count=node_count,
                                             start_date=start_date,
                                             end_date=end_date,
                                             reviewed_by=reviewed_by,
                                             minimum_cont=None)

            if minimum_cont:
                skid_list = _contribution_helper(
                    list(set(skid_list)))

            return list(set(skid_list))
        else:
            if isinstance(user, str):
                user = user_list.loc[user, 'id']
            get_skeleton_list_GET_data['created_by'] = user

    if not isinstance(reviewed_by, type(None)):
        if utils._is_iterable(reviewed_by):
            skid_list = list()
            for u in config.tqdm(reviewed_by, desc='Get revs',
                                 disable=config.pbar_hide,
                                 leave=config.pbar_leave):
                skid_list += get_neuron_list(remote_instance=remote_instance,
                                             user=user,
                                             node_count=node_count,
                                             start_date=start_date,
                                             end_date=end_date,
                                             reviewed_by=u,
                                             minimum_cont=None)

            return list(set(skid_list))
        else:
            if isinstance(reviewed_by, str):
                reviewed_by = user_list.loc[reviewed_by, 'id']
            get_skeleton_list_GET_data['reviewed_by'] = reviewed_by

    if start_date and not end_date:
        today = datetime.date.today()
        end_date = (today.year, today.month, today.day)

    if isinstance(start_date, datetime.date):
        start_date = [start_date.year, start_date.month, start_date.day]

    if isinstance(end_date, datetime.date):
        end_date = [end_date.year, end_date.month, end_date.day]

    if start_date and end_date:
        get_skeleton_list_GET_data['from'] = ''.join(
            [str(d) for d in start_date])
        get_skeleton_list_GET_data['to'] = ''.join([str(d) for d in end_date])

    remote_get_list_url = remote_instance._get_list_skeletons_url()
    remote_get_list_url += '?%s' % urllib.parse.urlencode(
        get_skeleton_list_GET_data)
    skid_list = remote_instance.fetch(remote_get_list_url)

    if minimum_cont and user:
        skid_list = _contribution_helper(
            list(set(skid_list)))

    return list(set(skid_list))


def get_history(remote_instance=None,
                start_date=(datetime.date.today() - datetime.timedelta(days=7)).isoformat(),
                end_date=datetime.date.today().isoformat(), split=True):
    """ Retrieves CATMAID project history.

    Notes
    -----
    If the time window is too large, the connection might time out which will
    result in an error! Make sure ``split=True`` to avoid that.

    Parameters
    ----------
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    start_date :        datetime | str | tuple, optional, default=last week
                        dates can be either:
                            - ``datetime.date``
                            - ``datetime.datetime``
                            - str ``'YYYY-MM-DD'``, e.g. ``'2016-03-09'``
                            - tuple ``(YYYY,MM,DD)``, e.g. ``(2016,3,9)``
    end_date :          datetime | str | tuple, optional, default=today
                        See start_date.
    split :             bool, optional
                        If True, history will be requested in bouts of 6 months.
                        Useful if you want to look at a very big time window
                        as this can lead to gateway timeout.

    Returns
    -------
    pandas.Series
            A pandas.Series with the following entries::

            {
            cable :             DataFrame containing cable created in nm.
                                Rows = users, columns = dates
            connector_links :   DataFrame containing connector links created.
                                Rows = users, columns = dates
            reviewed :          DataFrame containing nodes reviewed.
                                Rows = users, columns = dates
            user_details :      user-list (see pymaid.get_user_list())
            treenodes :         DataFrame containing nodes created by user.
            }

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import pymaid
    >>> rm = pymaid.CatmaidInstance( 'server_url',
    ...                              'http_user',
    ...                              'http_pw',
    ...                              'token')
    >>> # Get last week's history (using the default start/end dates)
    >>> hist = pymaid.get_history( remote_instance = rm )
    >>> # Plot cable created by all users over time
    >>> import matplotlib.pyplot as plt
    >>> hist.cable.T.plot()
    >>> plt.show()
    >>> # Collapse users and plot sum of cable over time
    >>> hist.cable.sum(0).plot()
    >>> plt.show()
    >>> # Plot single users cable (index by user login name)
    >>> hist.cable.ix['schlegelp'].T.plot()
    >>> plt.show()
    >>> # Sum up cable created this week by all users
    >>> hist.cable.values.sum()
    >>> # Get number of active (non-zero) users
    >>> active_users = hist.cable.astype(bool).sum(axis=0)

    """

    def _constructor_helper(data, key, days):
        """ Helper to extract variable from data returned by CATMAID server
        """
        temp = []
        for d in days:
            try:
                temp.append(data[d][key])
            except BaseException:
                temp.append(0)
        return temp

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(start_date, datetime.date):
        start_date = start_date.isoformat()
    elif isinstance(start_date, datetime.datetime):
        start_date = start_date.isoformat()[:10]
    elif isinstance(start_date, (tuple, list)):
        start_date = datetime.date(start_date[0], start_date[
                                   1], start_date[2]).isoformat()

    if isinstance(end_date, datetime.date):
        end_date = end_date.isoformat()
    elif isinstance(end_date, datetime.datetime):
        end_date = end_date.isoformat()[:10]
    elif isinstance(end_date, (tuple, list)):
        end_date = datetime.date(end_date[0], end_date[
                                 1], end_date[2]).isoformat()

    rounds = []
    if split:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        logger.info(
            'Retrieving %i days of history in bouts!' % (end - start).days)

        # First make big bouts of roughly 6 months each
        while start < (end - datetime.timedelta(days=6 * 30)):
            rounds.append(
                (start.isoformat(), (start + datetime.timedelta(days=6 * 30)).isoformat()))
            start += datetime.timedelta(days=6 * 30)

        # Append the last bit
        if start < end:
            rounds.append((start.isoformat(), end.isoformat()))
    else:
        rounds = [(start_date, end_date)]

    data = []
    for r in config.tqdm(rounds, desc='Retrieving history', disable=config.pbar_hide,
                  leave=config.pbar_leave):
        get_history_GET_data = {'self.project_id': remote_instance.project_id,
                                'start_date': r[0],
                                'end_date': r[1]
                                }

        remote_get_history_url = remote_instance._get_history_url()

        remote_get_history_url += '?%s' % urllib.parse.urlencode(
            get_history_GET_data)

        logger.debug(
            'Retrieving user history from %s to %s ' % (r[0], r[1]))

        data.append(remote_instance.fetch(remote_get_history_url))

    # Now merge data into a single dict
    stats = dict(data[0])
    for d in data:
        stats['days'] += [e for e in d['days'] if e not in stats['days']]
        stats['daysformatted'] += [e for e in d['daysformatted']
                                   if e not in stats['daysformatted']]

        for u in d['stats_table']:
            stats['stats_table'][u].update(d['stats_table'][u])

    user_list = get_user_list(remote_instance).set_index('id')
    user_list.index = user_list.index.astype(str)

    df = pd.Series([
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_cable_length', stats['days']) for u in stats['stats_table']],
                     index=[user_list.loc[u, 'login'] for u in stats[
                         'stats_table'].keys()],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_treenodes', stats['days']) for u in stats['stats_table']],
                     index=[user_list.loc[u, 'login'] for u in stats[
                         'stats_table'].keys()],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_connectors', stats['days']) for u in stats['stats_table']],
                     index=[user_list.loc[u, 'login'] for u in stats[
                         'stats_table'].keys()],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_reviewed_nodes', stats['days']) for u in stats['stats_table']],
                     index=[user_list.loc[u, 'login'] for u in stats[
                         'stats_table'].keys()],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        user_list.reset_index(drop=True)
    ],
        index=['cable', 'treenodes', 'connector_links',
               'reviewed', 'user_details']
    )

    return df


def get_nodes_in_volume(left, right, top, bottom, z1, z2, remote_instance=None,
                        coord_format='NM', resolution=(4, 4, 50)):
    """ Retrieve treenodes in given bounding box.

    Parameters
    ----------
    left :                  int | float
    right :                 int | float
    top :                   int | float
    bottom :                int | float
    z1 :                    int | float
    z2 :                    int | float
                            Coordinates defining the volume
                            Can be given in nm or pixels+slices.
    remote_instance :       CATMAID instance, optional
                            If not passed directly, will try using global.
    coord_format :          str, optional
                            Define whether provided coordinates are in
                            nanometer ('NM') or in pixels/slices ('PIXEL').
    resolution :            tuple of floats, optional
                            x/y/z resolution in nm (default = ( 4, 4, 50 ) )
                            Used to transform to nm if limits are given in
                            pixels.

    Returns
    -------
    dict
        Dictionary (d) containing the following entries::

        "treenodes" : pandas.DataFrame

        >>> d['treenodes']
          treenode_id parent_id x y z confidence radius skeleton_id edition_time user_id
        0
        1
        2

        "connectors" : pandas.DataFrame

        >>> d['connectors']
           connector_id x y z confidence edition_time user_id partners
        0
        1
        2

        >>> d['labels'] : dictionary
        { treenode_id : [ labels ] }

        >>> d['node_limit_reached'] : boolean
        True/False; if True, node limit was exceeded

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    # Set resolution to 1:1 if coordinates are already in nm
    if coord_format == 'NM':
        resolution = (1, 1, 1)

    remote_nodes_list = remote_instance._get_node_list_url()

    node_list_postdata = {
        'left': left * resolution[0],
        'right': right * resolution[0],
        'top': top * resolution[1],
        'bottom': bottom * resolution[1],
        'z1': z1 * resolution[2],
        'z2': z2 * resolution[2],
        # Atnid seems to be related to fetching the
        # active node too (will be ignored if atnid =
        # -1)
        'atnid': -1,
        'labels': False,
        'limit': 3500,  # This limits the number of nodes -> default is 3500

    }

    node_list = remote_instance.fetch(remote_nodes_list, node_list_postdata)

    data = {'treenodes': pd.DataFrame([[i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], datetime.datetime.fromtimestamp(int(i[8]), tz=datetime.timezone.utc), i[9], ]
                                       for i in node_list[0]],
                                      columns=['treenode_id', 'parent_id', 'x', 'y', 'z', 'confidence',
                                               'radius', 'skeleton_id', 'edition_time', 'user_id'],
                                      dtype=object),
            'connectors': pd.DataFrame([[i[0], i[1], i[2], i[3], i[4], datetime.datetime.fromtimestamp(int(i[5]), tz=datetime.timezone.utc), i[6], i[7]]
                                        for i in node_list[1]],
                                       columns=[
                'connector_id', 'x', 'y', 'z', 'confidence', 'edition_time', 'user_id', 'partners'],
        dtype=object),
        'labels': node_list[3],
        'node_limit_reached': node_list[4],
    }

    return data


def find_neurons(names=None, annotations=None, volumes=None, users=None,
                 from_date=None, to_date=None, reviewed_by=None, skids=None,
                 intersect=False, partial_match=False, only_soma=False,
                 min_size=1, minimum_cont=None, remote_instance=None):
    """ Find neurons matching given search criteria. Returns a CatmaidNeuronList.

    Warning
    -------
    Depending on the parameters, this can take quite a while! Also: by default,
    will return single-node neurons! Use the ``min_size`` parameter to change
    that behaviour.

    Parameters
    ----------
    names :             str | list of str
                        Neuron name(s) to search for.
    annotations :       str | list of str
                        Annotation(s) to search for.
    volumes :           str | core.Volume | list of either
                        CATMAID volume(s) to look into.
    users :             int | str | list of either, optional
                        User ID(s) (int) or login(s) (str).
    reviewed_by :       int | str | list of either, optional
                        User ID(s) (int) or login(s) (str) of reviewer.
    from_date :         datetime | list of integers, optional
                        Format: [year, month, day]. Return neurons created
                        after this date. This works ONLY if also querying by
                        ``users`` or ``reviewed_by``!
    to_date :           datetime | list of integers, optional
                        Format: [year, month, day]. Return neurons created
                        before this date. This works ONLY if also querying by
                        ``users`` or ``reviewed_by``!
    skids :             list of skids, optional
                        Can be a list of skids or pandas object with
                        "skeleton_id" columns.
    intersect :         bool, optional
                        If multiple search parameters are provided, this
                        parameter determines if neurons have to meet all of
                        them or just a single one in order to be returned. This
                        applies between AND within search parameters!
    partial_match :     bool, optional
                        If True, partial *names* AND *annotations* matches are
                        returned.
    minimum_cont :      int, optional
                        If looking for specific ``users``: minimum contribution
                        (in nodes) to a neuron in order for it to be counted.
                        Only applicable if ``users`` is provided. If multiple
                        users are provided contribution is calculated across
                        all users. Minimum contribution does NOT take start
                        and end dates into account! This is applied AFTER
                        intersecting!
    min_size :          int, optional
                        Minimum size (in nodes) for neurons to be returned.
                        The lower this value, the longer it will take to
                        filter.
    only_soma :         bool, optional
                        If True, only neurons with a soma are returned. This
                        is a very time-consuming step!
    remote_instance :   CATMAID instance
                        If not passed directly, will try using globally
                        defined CatmaidInstance.
    Returns
    -------
    :class:`~pymaid.CatmaidNeuronList`

    Examples
    --------
    >>> # Simple request for neurons with given annotations
    >>> to_find = ['glomerulus DA1','glomerulus DL4']
    >>> skids = pymaid.get_skids(annotations=to_find)
    >>> # Get only neurons that have both annotations
    >>> skids = pymaid.get_skids(annotations=to_find, intersect=True)
    >>> # Get all neurons with more than 1000 nodes
    >>> skids = pymaid.get_skids(min_size=500)
    >>> # Get all neurons that have been traced recently by given user
    >>> skids = pymaid.get_skids(users='schlegelp',
    ...                          from_date=[2017,10,1])
    >>> # Get all neurons traced by a given user within a certain volume
    >>> skids = pymaid.get_skids(users='schlegelp',
    ...                          minimum_cont=500,
    ...                          volumes='LH_R')

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    # Fist, we have to prepare a whole lot of parameters
    if users:
        users = utils.eval_user_ids(users, remote_instance=remote_instance)
    if reviewed_by:
        reviewed_by = utils.eval_user_ids(
            reviewed_by, remote_instance=remote_instance)
    if annotations and not isinstance(annotations, (list, np.ndarray)):
        annotations = [annotations]
    if names and not isinstance(names, (list, np.ndarray)):
        names = [names]
    if volumes and not isinstance(volumes, (list, np.ndarray)):
        volumes = [volumes]

    # Bring dates into the correct format
    if from_date and not to_date:
        today = datetime.date.today()
        to_date = (today.year, today.month, today.day)
    elif to_date and not from_date:
        from_date = (1900, 1, 1)

    if isinstance(from_date, datetime.date):
        from_date = [from_date.year, from_date.month, from_date.day]

    if isinstance(to_date, datetime.date):
        to_date = [to_date.year, to_date.month, to_date.day]

    # Warn if from/to_date are used without also querying by user or reviewer
    if from_date and not (users or reviewed_by):
        logger.warning(
            'Start/End dates can only be used for queries against <users> or <reviewed_by>')

    # Now go over all parameters and get sets of skids
    sets_of_skids = []

    if not isinstance(skids, type(None)):
        skids = utils.eval_skids(skids)
        sets_of_skids.append(set(skids))

    # Get skids by name
    if names:
        urls = [remote_instance._get_annotated_url() for n in names]
        post_data = [{'name': str(n), 'with_annotations': 'false'}
                     for n in names]

        results = remote_instance.fetch(urls,
                                        post=post_data,
                                        desc='Get names')

        this_name = []
        for i, r in enumerate(results):
            for e in r['entities']:
                if partial_match and e['type'] == 'neuron' and names[i].lower() in e['name'].lower():
                    this_name.append(e['skeleton_ids'][0])
                if not partial_match and e['type'] == 'neuron' and e['name'] == names[i]:
                    this_name.append(e['skeleton_ids'][0])

        sets_of_skids.append(set(this_name))

    # Get skids by annotation
    if annotations:
        annotation_ids = get_annotation_id(
            annotations, remote_instance, allow_partial=partial_match)

        if not annotation_ids:
            logger.error(
                'No matching annotation(s) found! Returning None')
            raise Exception('No matching annotation(s) found!')

        if partial_match is True:
            logger.debug(
                'Found {0} id(s) (partial matches included)'.format(len(annotation_ids)))
        else:
            logger.debug('Found id(s): %s | Unable to retrieve: %i' % (
                str(annotation_ids), len(annotations) - len(annotation_ids)))

        logger.debug(
            'Retrieving skids for annotationed neurons')

        urls = [remote_instance._get_annotated_url() for an in annotation_ids]
        post_data = [{'annotated_with0': str(an), 'with_annotations': 'false'}
                     for an in annotation_ids.values()]
        results = remote_instance.fetch(urls,
                                        post=post_data,
                                        desc='Get annot')
        sets_of_skids.append(set([e['skeleton_ids'][0] for res in results for e in res['entities'] if e['type'] == 'neuron']))

    # Get skids by user
    if users:
        urls = [remote_instance._get_list_skeletons_url() for u in users]
        GET_data = [{'nodecount_gt': min_size,
                     'created_by': u} for u in users]

        if from_date and to_date:
            dates = {'from' : ''.join([str(d) for d in from_date]),
                     'to': ''.join([str(d) for d in to_date])}
            GET_data = [{**d, **dates} for d in GET_data]
        urls = [u + '?%s' % urllib.parse.urlencode(g) for u, g in zip(urls, GET_data)]

        results = remote_instance.fetch(urls, desc='Get users')

        sets_of_skids.append(set([s for res in results for s in res]))

    # Get skids by reviewer
    if reviewed_by:
        urls = [remote_instance._get_list_skeletons_url() for u in reviewed_by]
        GET_data = [{'nodecount_gt': min_size,
                     'reviewed_by': u} for u in reviewed_by]

        if from_date and to_date:
            dates = {'from' : ''.join([str(d) for d in from_date]),
                     'to': ''.join([str(d) for d in to_date])}
            GET_data = [{**d, **dates} for d in GET_data]
        urls = [u + '?%s' % urllib.parse.urlencode(g) for u, g in zip(urls, GET_data)]

        results = remote_instance.fetch(urls, desc='Get reviewers')

        sets_of_skids.append(set([s for res in results for s in res]))

    # Get by volume
    if volumes:
        for v in config.tqdm(volumes, desc='Get by vols', disable=config.pbar_hide,
                      leave=config.pbar_leave):
            if not isinstance(v, core.Volume):
                vol = get_volume(v, remote_instance)
            else:
                vol = v

            temp = get_neurons_in_bbox(
                vol.bbox, remote_instance=remote_instance)

            sets_of_skids.append(set(temp))

    # Get neurons by size if only min_size and no other no parameters were provided
    if False not in [isinstance(param, type(None)) for param in [names, annotations, volumes, users, reviewed_by, skids]]:
        # Make sure people don't accidentally request ALL neurons in the dataset
        if min_size <= 1:
            answer = ""
            while answer not in ["y", "n"]:
                answer = input(
                    "Your search parameters will retrieve ALL neurons in the dataset. Proceed? [Y/N] ").lower()

            if answer != 'y':
                logger.info('Query cancelled')
                return

        logger.info(
            'Get all neurons with at least {0} nodes'.format(min_size))
        get_skeleton_list_GET_data = {'nodecount_gt': min_size}
        remote_get_list_url = remote_instance._get_list_skeletons_url()
        remote_get_list_url += '?%s' % urllib.parse.urlencode(
            get_skeleton_list_GET_data)
        these_neurons = set(remote_instance.fetch(remote_get_list_url))

        sets_of_skids.append(these_neurons)

    # Now merge the different neurons that we have
    if intersect:
        logger.info('Intersecting by search parameters')
        skids = list(set.intersection(*sets_of_skids))
    else:
        skids = list(set.union(*sets_of_skids))

    # Filtering by size was already done for users and reviewed_by and dates
    # If we queried by annotations, names or volumes we need to do this explicitly here
    if min_size > 1 and (volumes or annotations or names):
        logger.info('Filtering neurons for size')

        get_skeleton_list_GET_data = {'nodecount_gt': min_size}
        remote_get_list_url = remote_instance._get_list_skeletons_url()
        remote_get_list_url += '?%s' % urllib.parse.urlencode(
            get_skeleton_list_GET_data)
        neurons_by_size = set(remote_instance.fetch(remote_get_list_url))

        skids = set.intersection(set(skids), neurons_by_size)

    nl = core.CatmaidNeuronList(list(skids), remote_instance=remote_instance)
    nl.get_names()

    if only_soma:
        logger.info('Filtering neurons for somas...')
        nl.get_skeletons(skip_existing=True)
        nl = nl[nl.soma != None]

    if users and minimum_cont:
        nl.get_skeletons(skip_existing=True)
        nl = core.CatmaidNeuronList([n for n in nl if n.nodes[n.nodes.creator_id.isin(users)].shape[0] >= minimum_cont],
                                    remote_instance=remote_instance)

    if nl.empty:
        logger.warning(
            'No neurons matching the search parameters were found')
    else:
        logger.info(
            'Found {0} neurons matching the search parameters'.format(len(nl)))

    return nl


def get_neurons_in_volume(volumes, intersect=False, min_nodes=2,
                          only_soma=False, remote_instance=None):
    """ Retrieves neurons with processes within CATMAID volumes. This function
    uses the **BOUNDING BOX** around volume as proxy and queries for neurons
    that are within that volume. See examples on how to work around this.

    Warning
    -------
    Depending on the number of nodes in that volume, this can take quite a
    while! Also: by default, will NOT return single-node neurons - use the
    ``min_nodes`` parameter to change that behaviour.

    Parameters
    ----------
    volumes :               str | core.Volume | list of either
                            Single or list of CATMAID volumes.
    intersect :             bool, optional
                            If multiple volumes are provided, this parameter
                            determines if neurons have to be in all of the
                            neuropils or just a single.
    min_nodes :             int, optional
                            Minimum number of node these neurons need to have
                            in given volumes.
    only_soma :             bool, optional
                            If True, only neurons with a soma will be returned.
                            In case you are going to retrieve skeleton data
                            anyway, it is better to do that for all neurons
                            and then filter for soma.
    remote_instance :       CATMAID instance
                            If not passed directly, will try using global.

    Returns
    -------
    list
                            ``[skeleton_id, skeleton_id, ...]``

    See Also
    --------
    :func:`~pymaid.get_partners_in_volume`
                            Get only partners that make connections within a
                            given volume.
    :func:`pymaid.find_neurons`
                            Use to retrieve neurons by combining various
                            search criteria. For example names, reviewers,
                            annotations, etc.

    Examples
    --------
    >>> # Get a volume
    >>> lh = pymaid.get_volume('LH_R')
    >>> # Get neurons within the bounding box of a volume
    >>> skids = pymaid.get_neurons_in_volume(lh, min_nodes = 10)
    >>> # Retrieve 3D skeletons of these neurons
    >>> lh_neurons = pymaid.get_neurons(skids)
    >>> # Prune by volume
    >>> lh_pruned = lh_neurons.copy()
    >>> lh_pruned.prune_by_volume(lh)
    >>> # Filter neurons with more than 100um of cable in the volume
    >>> n = lh_neurons[ lh_pruned.cable_length > 100  ]

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(volumes, (list, np.ndarray)):
        volumes = [volumes]

    for i, v in enumerate(volumes):
        if not isinstance(v, core.Volume):
            volumes[i] = get_volume(v)

    neurons = []

    for v in volumes:
        logger.info('Retrieving nodes in volume {0}'.format(v.name))
        temp = get_neurons_in_bbox(v.bbox, min_nodes=min_nodes,
                                   remote_instance=remote_instance)

        if not intersect:
            neurons += list(temp)
        else:
            neurons += [temp]

    if intersect:
        # Filter for neurons that show up in all neuropils
        neurons = [n for l in neurons for n in l if False not in [
            n in v for v in neurons]]

    # Need to do this in case we have several volumes
    neurons = list(set(neurons))

    if only_soma:
        logger.info('Filtering neurons for somas...')
        soma = has_soma(neurons, remote_instance)
        neurons = [n for n in neurons if soma[n] is True]

    logger.info('Done. {0} unique neurons found in volume(s) {1}'.format(
        len(neurons), ','.join([v.name for v in volumes])))

    return neurons


def get_neurons_in_bbox(bbox, unit='NM', min_nodes=1, remote_instance=None,
                        **kwargs):
    """ Retrieves neurons with processes within a defined box volume. Because the
    API returns only a limited number of neurons at a time, the defined volume
    has to be chopped into smaller pieces for crowded areas - may thus take
    some time! This function will retrieve ALL neurons within the box - not
    just the once entering/exiting.

    Parameters
    ----------
    bbox :                  np.ndarray | list, dict
                            Coordinates of the bounding box. Can be either:

                              1. List/np.array: ``[[left, right], [top, bottom], [z1, z2]]``
                              2. Dictionary with above entries.
    unit :                  'NM' | 'PIXEL'
                            Unit of your coordinates. Attention:
                            'PIXEL' will also assume that Z1/Z2 is in slices.
                            By default, a X/Y resolution of 3.8nm and a Z
                            resolution of 35nm is assumed. Pass 'xy_res' and
                            'z_res' as **kwargs to override this.
    min_nodes :             int, optional
                            Minimum node count for a neuron within given box
                            to be returned.
    remote_instance :       CATMAID instance
                            If not passed directly, will try using global.

    Returns
    --------
    list
                            ``[ skeleton_id, skeleton_id, ... ]``

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x_y_resolution = kwargs.get('xy_res', 3.8)
    z_resolution = kwargs.get('z_res', 35)

    if isinstance(bbox, dict):
        bbox = np.array([[bbox['left'], bbox['right']],
                         [bbox['top'], bbox['bottom']],
                         [bbox['z1'], bbox['z2']]
                         ])

    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)

    if unit == 'PIXEL':
        bbox *= x_y_resolution

    # Subset the volume into boxes of 50**3 um^3
    boxes = _subset_volume(bbox, max_vol=25**3)

    node_list = []
    while boxes.any():
        # Prepare urls and postdata for each box
        urls = [remote_instance._get_node_list_url() for u in boxes]
        postdata = [{
                    'left': bbox[0][0],
                    'right': bbox[0][1],
                    'top': bbox[1][0],
                    'bottom': bbox[1][1],
                    'z1': bbox[2][0],
                    'z2': bbox[2][1],
                    # Atnid seems to be related to fetching the
                    # active node too (will be ignored if atnid
                    # = -1)
                    'atnid': -1,
                    'labels': False,
                    # Maximum number of nodes returned per
                    # query - default appears to be 3500 (on
                    # Github) but it appears as if this is
                    # overriden by server settings anyway!
                    'limit': 1000000
                    } for bbox in boxes]

        data = remote_instance.fetch(urls, post=postdata, disable_pbar=False,
                                     leave_pbar=False, desc='Fetching nodes')

        # Collect data and check if we need more fetching
        new_boxes = np.empty((0, 3, 2))
        for nl, bbox in zip(data, boxes):
            # Subdivide if too many nodes returned
            if nl[3] is True:
                # Divided this box into 8 smaller boxes
                new_boxes = np.append(new_boxes, _subset_volume(bbox), axis=0)
            else:
                # If limit not reached, return node list
                node_list += nl[0]
        boxes = new_boxes

    # Collapse list into unique skeleton ids
    unique, counts = np.unique([n[7] for n in node_list], return_counts=True)
    skeletons = unique[counts >= min_nodes]

    logger.info("Done: %i nodes from %i unique neurons retrieved." % (
        len(node_list), len(skeletons)))

    return skeletons


def _subset_volume(bbox, max_vol=None):
    """ Subdivide a bounding box into smaller subvolumes. Can provide a max
    volume size.

    Parameters
    ----------
    bbox :      dict
                Must contain these entries: left, right, top, bottom, z1, z2.
    max_vol :   int, optional
                Maximum volume per subvolume in cubic microns [um^3].

    Returns
    -------
    subvolumes :    np.ndarray

    """

    # Tile space
    b_x = zip(np.linspace(bbox[0][0], bbox[0][1], 3)[:-1],
              np.linspace(bbox[0][0], bbox[0][1], 3)[1:])

    b_y = zip(np.linspace(bbox[1][0], bbox[1][1], 3)[:-1],
              np.linspace(bbox[1][0], bbox[1][1], 3)[1:])

    b_z = zip(np.linspace(bbox[2][0], bbox[2][1], 3)[:-1],
              np.linspace(bbox[2][0], bbox[2][1], 3)[1:])

    # For some reason we need to convert list, otherwise coordinates are
    # dropped
    b_x = list(b_x)
    b_y = list(b_y)
    b_z = list(b_z)

    new_boxes = []
    for x in b_x:
        for y in b_y:
            for z in b_z:
                new_boxes.append([x, y, z])
    new_boxes = np.array(new_boxes)

    # Check the volume of the new boxes. If volume is too big, do another
    # round of subsetting
    if max_vol:
        this_volume = (new_boxes[0][:, 1] -
                       new_boxes[0][:, 0]).prod() / 1000**3
        if this_volume > max_vol:
            new_boxes = np.array(
                [b for box in new_boxes for b in _subset_volume(box,
                                                                max_vol=max_vol)])

    return new_boxes


def get_user_list(remote_instance=None):
    """ Get list of users for given CATMAID server (not project specific).

    Parameters
    ----------
    remote_instance :   CATMAID instance
                        If not passed directly, will try using global.

    Returns
    ------
    pandas.DataFrame
        DataFrame in which each row represents a user:

        >>> print(user_list)
          id   login   full_name   first_name   last_name   color
        0
        1

    Examples
    --------
    >>> user_list = pymaid.get_user_list(remote_instance = rm)
    >>> # To search for e.g. user ID 22
    >>> user_list.set_index('id',inplace=True)
    >>> user_list.ix[ 22 ]
    id                                  22
    login                      mustermannm
    full_name          Michaela Mustermann
    first_name                     Michael
    last_name                   Mustermann
    color         [0.91389, 0.877853, 1.0]
    >>> user_list.reset_index(inplace=True)
    >>> # To convert into a classic dict
    >>> d = user_list.set_index('id').T.to_dict()
    >>> d[22]['first_name']
    ... Michaela

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    user_list = remote_instance.fetch(remote_instance._get_user_list_url())

    columns = ['id', 'login', 'full_name', 'first_name', 'last_name', 'color']

    df = pd.DataFrame([[e[c] for c in columns] for e in user_list],
                      columns=columns
                      )

    df.sort_values(['login'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def get_paths(sources, targets, remote_instance=None, n_hops=2, min_synapses=1,
              return_graph=False, remove_isolated=False):
    """ Retrieves paths between two sets of neurons.

    Parameters
    ----------
    sources
                        Source neurons.
    targets
                        Target neurons. ``sources`` and ``targets`` can be:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object

    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.
    n_hops :            int | list | range, optional
                        Number of hops allowed between sources and
                        targets. Direct connection would be 1 hop.

                        1. int, e.g. ``n_hops=3`` will return paths with
                        EXACTLY 3 hops
                        2. list, e.g. ``n_hops=[2,4]`` will return all
                        paths with 2 and 4 hops
                        3. range, e.g. ``n_hops=range(2,4)`` will be converted
                        to a list and return paths with 2 and 3 hops.
    min_synapses :      int, optional
                        Minimum number of synpases between source and target.
    return_graph :      bool, optional
                        If True, will return NetworkX Graph (see below).
    remove_isolated :   bool, optional
                        Remove isolated nodes from NetworkX Graph. Only
                        relevant if ``return_graph=True``.

    Returns
    -------
    paths :     list
                List of skeleton IDs that constitute paths from
                sources to targets::

                    [[source1, ..., target1], [source2, ..., target2], ...]

    networkx.DiGraph
                Only if ``return_graph=True``. Graph contains all neurons that
                connect sources and targets. **Important**: Does only contain
                edges that connect sources and targets via max ``n_hops``!
                Other edges have been removed.

    Examples
    --------
    >>> # This assumes that you have already set up a Catmaid Instance
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> g, paths = pymaid.get_paths( ['annotation:glomerulus DA1'],
    ...                              ['2333007'] )
    >>> g
    <networkx.classes.digraph.DiGraph at 0x127d12390>
    >>> paths
    [['57381', '4376732', '2333007'], ['57323', '630823', '2333007'], ...
    >>> nx.draw(g)
    >>> plt.show()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    sources = utils.eval_skids(sources, remote_instance=remote_instance)
    targets = utils.eval_skids(targets, remote_instance=remote_instance)

    if not isinstance(targets, (list, np.ndarray)):
        targets = [targets]

    if not isinstance(sources, (list, np.ndarray)):
        sources = [sources]

    n_hops = utils._make_iterable(n_hops)

    response = []
    if min(n_hops) <= 0:
        raise ValueError('n_hops must not be <= 0')

    url = remote_instance._get_graph_dps_url()
    for h in range(1, max(n_hops) + 1):
        if h == 1:
            response += list(sources) + list(targets)
            continue

        post_data = {
            'n_hops': h,
            'min_synapses': min_synapses
        }

        for i, s in enumerate(sources):
            post_data['sources[%i]' % i] = s

        for i, t in enumerate(targets):
            post_data['targets[%i]' % i] = t

        # Response is just a set of skeleton IDs
        response += remote_instance.fetch(url, post=post_data)

    response = list(set(response))

    # Turn neurons into an NetworkX graph
    g = graph.network2nx(
        response, remote_instance=remote_instance, threshold=min_synapses)

    # Get all paths between sources and targets
    all_paths = [p for s in sources for t in targets for p in nx.all_simple_paths(
        g, s, t, cutoff=max(n_hops)) if len(p) - 1 in n_hops]

    if not return_graph:
        return all_paths

    # Turn into edges
    edges_to_keep = set([e for l in all_paths for e in nx.utils.pairwise(l)])

    # Remove edges
    g.remove_edges_from([e for e in g.edges if e not in edges_to_keep])

    if remove_isolated:
        # Remove isolated nodes
        g.remove_nodes_from(list(nx.isolates(g)))

    return all_paths, g


def get_volume(volume_name=None, remote_instance=None,
               color=(120, 120, 120, .6), combine_vols=False):
    """ Retrieves volume (mesh) from Catmaid server and converts to set of
    vertices and faces.

    Parameters
    ----------
    volume_name :       str | list of str
                        Name(s) of the volume to import - must be EXACT!
                        If ``volume_name=None``, will return list of all
                        available CATMAID volumes. If list of volume names,
                        will return a dictionary ``{ name : Volume, ... }``
    remote_instance :   CATMAIDInstance, optional
                        If not passed directly, will try using global.
    color :             tuple, optional
                        R,G,B,alpha values used by :func:`~pymaid.plot3d`.
    combine_vols :      bool, optional
                        If True and multiple volumes are requested, the will
                        be combined into a single volume.

    Returns
    -------
    :class:`~pymaid.Volume`
            If ``volume_name`` is list of volumes, returns a dictionary of
            Volumes: ``{ name1 : Volume1, name2 : Volume2, ...}``

    Examples
    --------
    >>> import pymaid
    >>> rm = CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
    >>> # Retrieve volume
    >>> vol = pymaid.get_volume('LH_R')
    >>> # Plot volume
    >>> vol.plot3d()

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(volume_name, type(None)):
        logger.info('Retrieving list of available volumes.')
    elif not isinstance(volume_name, (str, list, np.ndarray)):
        raise TypeError('Volume name must be str or list of str.')

    volume_names = utils._make_iterable(volume_name)

    # First, get volume IDs
    get_volumes_url = remote_instance._get_volumes()
    response = remote_instance.fetch(get_volumes_url)

    if not volume_name:
        return pd.DataFrame.from_dict(response)

    volume_ids = [e['id'] for e in response if e['name'] in volume_names]

    if len(volume_ids) != len(volume_names):
        not_found = [v for v in volume_names if v not in response.name.values]
        raise Exception(
            'No volume(s) found for: {}'.format(not_found.split(',')))

    url_list = [remote_instance._get_volume_details(v) for v in volume_ids]

    # Get data
    responses = remote_instance.fetch(url_list, desc='Volumes')

    # Generate volume(s) from responses
    volumes = {}
    for r in responses:
        mesh_str = r['mesh']
        mesh_name = r['name']
        mesh_id = r['id']

        mesh_type = re.search('<(.*?) ', mesh_str).group(1)

        # Now reverse engineer the mesh
        if mesh_type == 'IndexedTriangleSet':
            t = re.search("index='(.*?)'", mesh_str).group(1).split(' ')
            faces = [(int(t[i]), int(t[i + 1]), int(t[i + 2]))
                     for i in range(0, len(t) - 2, 3)]

            v = re.search("point='(.*?)'", mesh_str).group(1).split(' ')
            vertices = [(float(v[i]), float(v[i + 1]), float(v[i + 2]))
                        for i in range(0, len(v) - 2, 3)]

        elif mesh_type == 'IndexedFaceSet':
            # For this type, each face is indexed and an index of -1 indicates the
            # end of this face set
            t = re.search("coordIndex='(.*?)'", mesh_str).group(1).split(' ')
            faces = []
            this_face = []
            for f in t:
                if int(f) != -1:
                    this_face.append(int(f))
                else:
                    faces.append(this_face)
                    this_face = []

            # Make sure the last face is also appended
            faces.append(this_face)

            v = re.search("point='(.*?)'", mesh_str).group(1).split(' ')
            vertices = [(float(v[i]), float(v[i + 1]), float(v[i + 2]))
                        for i in range(0, len(v) - 2, 3)]

        else:
            logger.error("Unknown volume type: %s" % mesh_type)
            raise Exception("Unknown volume type: %s" % mesh_type)

        # For some reason, in this format vertices occur multiple times - we have
        # to collapse that to get a clean mesh
        final_faces = []
        final_vertices = []

        for t in faces:
            this_faces = []
            for v in t:
                if vertices[v] not in final_vertices:
                    final_vertices.append(vertices[v])

                this_faces.append(final_vertices.index(vertices[v]))

            final_faces.append(this_faces)

        logger.debug('Volume type: %s' % mesh_type)
        logger.debug(
            '# of vertices after clean-up: %i' % len(final_vertices))
        logger.debug(
            '# of faces after clean-up: %i' % len(final_faces))

        v = core.Volume(name=mesh_name,
                        volume_id=mesh_id,
                        vertices=final_vertices,
                        faces=final_faces,
                        color=color)

        volumes[mesh_name] = v

    # Return just the volume if a single one was requested
    if len(volumes) == 1:
        return list(volumes.values())[0]

    return volumes


def get_annotation_list(remote_instance=None):
    """ Get a list of all annotations in the project.

    Parameters
    ----------
    remote_instance : CatmaidInstance, optional
                      If not passed directly, will try using global.

    Returns
    -------
    pandas DataFrame
            DataFrame in which each row represents an annotation:

            >>> print(user_list)
              annotation_id   annotation   users
            0
            1
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    an = remote_instance.fetch(remote_instance._get_annotation_list())[
        'annotations']

    df = pd.DataFrame.from_dict(an)
    df.columns = ['annotation_id', 'annotation', 'users']

    return df


def url_to_coordinates(coords, stack_id, active_skeleton_id=None,
                       active_node_id=None, remote_instance=None, zoom=0,
                       tool='tracingtool'):
    """ Generate URL to a location.

    Parameters
    ----------
    coords :                list | np.ndarray | pandas.DataFrame
                            ``x``, ``y``, ``z`` coordinates.
    stack_id :              int | list/array of ints
                            ID of the image stack you want to link to.
                            Depending on your setup this parameter might be
                            overriden by local user settings.
    active_skeleton_id :    int | list/array of ints, optional
                            Skeleton ID of the neuron that should be selected.
    active_node_id :        int | list/array of ints, optional
                            Treenode/Connector ID of the node that should be
                            active.
    zoom :                  int, optional
    tool :                  str, optional
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    str | list of str
                URL(s) to the coordinates provided.

    Examples
    --------
    >>> # Get URL for a single coordinate
    >>> url = pymaid.url_to_coordinates([1000,1000,1000], stack_id=5)
    >>> # Get URLs for all low-confidence nodes of a neuron
    >>> n = pymaid.get_neuron(27295)
    >>> low_c = n.nodes.loc[n.nodes.confidence < 5]
    >>> urls = pymaid.url_to_coordinates(low_c[['x','y','z']].values,
                                         stack_id = 5,
                                         active_node_id = low_c.treenode_id.values)
    """

    def gen_url(c, stid, nid, sid):
        """ This function generates the actual urls
        """
        GET_data = {'pid': remote_instance.project_id,
                    'xp': int(c[0]),
                    'yp': int(c[1]),
                    'zp': int(c[2]),
                    'tool': tool,
                    'sid0': stid,
                    's0': zoom
                    }

        if sid:
            GET_data['active_skeleton_id'] = sid
        if nid:
            GET_data['active_node_id'] = nid

        return(remote_instance.make_url('?%s' % urllib.parse.urlencode(GET_data)))

    def list_helper(x):
        """ Helper function to turn variables into lists matching length of coordinates
        """
        if not isinstance(x, (list, np.ndarray)):
            return [x] * len(coords)
        elif len(x) != len(coords):
            raise ValueError('Parameters must be the same shape as coords.')
        else:
            return x

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(coords, (pd.DataFrame, pd.Series)):
        try:
            coords = coords[['x', 'y', 'z']].values
        except:
            raise ValueError(
                'Pandas DataFrames must have "x","y" and "z" columns.')
    elif isinstance(coords, list):
        coords = np.array(coords)

    if isinstance(coords, np.ndarray) and coords.ndim > 1:
        stack_id = list_helper(stack_id)
        active_skeleton_id = list_helper(active_skeleton_id)
        active_node_id = list_helper(active_node_id)

        return [gen_url(c, stid, nid, sid) for c, stid, nid, sid in zip(coords, stack_id, active_node_id, active_skeleton_id)]
    else:
        return gen_url(coords, stack_id, active_node_id, active_skeleton_id)


def rename_neurons(x, new_names, remote_instance=None, no_prompt=False):
    """ Rename neuron(s).

    Parameters
    ----------
    x
                        Neuron(s) to rename. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    new_names :         str | list | dict
                        New name(s). If renaming multiple neurons this
                        needs to be a dict mapping skeleton IDs to new
                        names or a list of the same size as provided skeleton
                        IDs.
    no_prompt :         bool, optional
                        By default, you will be prompted to confirm before
                        renaming neuron(s). Set this to True to skip that step.
    remote_instance :   CATMAID instance, optional
                        If not passed directly, will try using global.

    Examples
    --------
    Add your initials to a set of neurons

    >>> # Get their current names
    >>> names = pymaid.get_names('annotation:Philipps neurons')
    >>> # Add initials to names
    >>> new_names = {skid : name + ' PS' for skid, name in names.items()}
    >>> # Rename neurons
    >>> pymaid.rename_neurons(list(names.keys()), new_names)

    Returns
    -------
    Nothing

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    if isinstance(new_names, dict):
        # First make sure that dictionary maps strings
        _ = {str(n): new_names[n] for n in new_names}
        # Generate a list from the dict
        new_names = [_[n] for n in x if n in _]
    elif not isinstance(new_names, (list, np.ndarray)):
        new_names = [new_names]

    if len(x) != len(new_names):
        raise ValueError('Need a name for every single neuron to rename.')

    if not no_prompt:
        old_names = get_names(x)
        df = pd.DataFrame(data=[[old_names[n], new_names[i], n] for i, n in enumerate(x)],
                          columns=['Current name', 'New name', 'Skeleton ID']
                          )
        print(df)
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("Please confirm above renaming [Y/N] ").lower()

        if answer != 'y':
            return

    url_list = []
    postdata = []
    for skid, name in zip(x, new_names):
        # Renaming works with neuron ID, which we can get via this API endpoint
        remote_get_neuron_name = remote_instance._get_single_neuronname_url(
            skid)
        neuron_id = remote_instance.fetch(remote_get_neuron_name)['neuronid']
        url_list.append(remote_instance._rename_neuron_url(neuron_id))
        postdata.append({'name': name})

    # Get data
    responses = [r for r in remote_instance.fetch(url_list,
                                                  post=postdata,
                                                  desc='Renaming')]

    if False not in [r['success'] for r in responses]:
        logger.info('All neurons successfully renamed.')
    else:
        failed = [n for i, n in enumerate(
            x) if responses[i]['success'] is False]
        logger.error(
            'Error renaming neuron(s): {0}'.format(','.join(failed)))

    return


def get_label_list(remote_instance=None):
    """ Retrieves all labels (TREENODE tags only) in a project.

    Parameters
    ----------
    remote_instance :   CatmaidInstance, optional
                        If not provided, will search for globally defined
                        remote instance.

    Returns
    -------
    pandas.DataFrame
            DataFrame in which each row represents a label:

            >>> df
                label_id  tag  skeleton_id  treenode_id
            0
            1
            2

    Examples
    --------
    >>> # Get all labels
    >>> labels = pymaid.get_label_list()
    >>> # Get all nodes with a given tag
    >>> treenodes = labels[ labels.tag == 'my_label' ].treenode_id
    >>> # Get neuron that have at least a single node with a given tag
    >>> neurons = labels[ labels.tag == 'my_label' ].skeleton_id.unique()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    labels = remote_instance.fetch(remote_instance._get_label_list_url())

    return pd.DataFrame(labels, columns=['label_id', 'tag', 'skeleton_id',
                                         'treenode_id'])


def get_transactions(range_start=None, range_length=25, remote_instance=None):
    """ Retrieve individual transactions with server. **This API endpoint is
    extremely slow!**

    Parameters
    ----------
    range_start :       int, optional
                        Start of table. Transactions are returned in
                        chronological order (most recent transactions first)
    range_length :      int, optional
                        End of table. If None, will return all.
    remote_instance :   CatmaidInstance, optional

    Returns
    -------
    pandas.DataFrame
            >>> df
               change_type      execution_time          label
            0  Backend       2017-12-26 03:37:00     labels.update
            1  Backend       2017-12-26 03:37:00  treenodes.create
            2  Backend       2017-12-26 03:37:00  treenodes.create
            3  Backend       2017-12-26 03:37:00  treenodes.create
            4  Backend       2017-12-26 03:32:00  treenodes.create
               project_id  transaction_id  user_id    user
            0  1            404899166        151     dacksa
            1  1            404899165        151     dacksa
            2  1            404899164        151     dacksa
            3  1            404899163        151     dacksa
            4  1            404899162        151     dacksa

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    remote_transactions_url = remote_instance._get_transactions_url()

    desc = {'range_start': range_start, 'range_length': range_length}
    desc = {k: v for k, v in desc.items() if v is not None}

    remote_transactions_url += '?%s' % urllib.parse.urlencode(desc)

    data = remote_instance.fetch(remote_transactions_url)

    df = pd.DataFrame.from_dict(data['transactions'])

    user_list = get_user_list(remote_instance).set_index('id')

    df['user'] = [user_list.loc[uid, 'login'] for uid in df.user_id.values]

    df['execution_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['execution_time'].values]

    return df
