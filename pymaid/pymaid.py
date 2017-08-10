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

""" Low-level wrappers to request data from Catmaid server

Examples
--------
>>> from pymaid.pymaid import CatmaidInstance, get_neuron
>>> # HTTP_USER AND HTTP_PASSWORD are only necessary if your server requires a 
... # http authentification
>>> myInstance = CatmaidInstance(   'www.your.catmaid-server.org' , 
...                                 'HTTP_USER' , 
...                                 'HTTP_PASSWORD', 
...                                 'TOKEN' )
>>> neuron_list = get_neuron ( ['12345','67890'] , myInstance )
>>> neuron_list[0]
type              <class 'pymaid.core.CatmaidNeuron'>
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

Notes
-----
Also see https://github.com/schlegelp/PyMaid for Jupyter notebooks
"""

import urllib
import json
import http.cookiejar as cj
import time
import base64
import threading
import datetime
import logging
import re
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

from pymaid import core, morpho, igraph_catmaid


class CatmaidInstance:
    """ A class giving access to a CATMAID instance.

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
                    ID of your project. Default = 1
    logger :        optional
                    provide name for logging.getLogger if you like the CATMAID 
                    instance to log to a specific logger by default (None), a 
                    dedicated logger __name__ is created
    logger_level :  {'DEBUG','INFO','WARNING','ERROR'}, optional
                    Sets logger level
    time_out :      integer or None
                    Time in seconds after which fetching data will time-out 
                    (so as to not block the system).
                    If set to None, time-out will be max([ 30, len(requests) ])

    Notes
    -----
    CatmaidInstance holds credentials and performs fetch operations. 
    You can either pass this object to each function individually or 
    define module wide by using e.g:: 

        pymaid.remote_instance = CatmaidInstance

    Examples
    --------
    Ordinarily, you would use one of the wrapper functions in 
    :mod:`pymaid.pymaid` but if you feel like getting your hands on the raw
    data, here is how it goes:

    >>> # 1.) Fetch raw skeleton data for a single neuron
    >>> from pymaid.pymaid import CatmaidInstance
    >>> myInstance = CatmaidInstance(   'www.your.catmaid-server.org', 
    ...                                 'user', 
    ...                                 'password', 
    ...                                 'token' 
    ...                             )
    >>> skeleton_id = 12345
    >>> 3d_skeleton_url = myInstance._get_compact_skeleton_url( skeleton_id )
    >>> raw_data = myInstance.fetch( 3d_skeleton_url )
    >>> # 2.) Use wrapper to generate CatmaidNeuron objects
    >>> from pymaid.pymaid import CatmaidInstance, get_neuron
    >>> myInstance = CatmaidInstance(   'www.your.catmaid-server.org', 
    ...                                 'user', 
    ...                                 'password', 
    ...                                 'token' )
    >>> neuron_list = get_neuron ( ['12345','67890'] , myInstance )    
    >>> # Print summary
    >>> print(neuron_list)
    """

    def __init__(self, server, authname, authpassword, authtoken, project_id=1, logger=None, logger_level='INFO', time_out=None):
        self.server = server
        self.authname = authname
        self.authpassword = authpassword
        self.authtoken = authtoken
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPRedirectHandler())
        self.time_out = time_out
        self.project_id = project_id

        # If pymaid is not run as module, make sure logger has a at least a
        # StreamHandler
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(logger)

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            # Create formatter and add it to the handlers
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)

            self.logger.addHandler(sh)

        self.logger.setLevel(logger_level)

        self.logger.info(
            'CATMAID instance created. See help(CatmaidInstance) to learn how to define globally.')

    def djangourl(self, path):
        """ Expects the path to lead with a slash '/'. """
        return self.server + path

    def auth(self, request):
        if self.authname:
            base64str = base64.encodestring(
                ('%s:%s' % (self.authname, self.authpassword)).encode()).decode().replace('\n', '')
            request.add_header("Authorization", "Basic %s" % base64str)
        if self.authtoken:
            request.add_header("X-Authorization",
                               "Token {}".format(self.authtoken))

    def fetch(self, url, post=None):
        """ Requires the url to connect to and the variables for POST, if any, in a dictionary. """
        if post:
            # Convert bools into lower case str
            for v in post:
                if isinstance(post[v], bool):
                    post[v] = str(post[v]).lower()

            data = urllib.parse.urlencode(post)
            data = data.encode('utf-8')
            self.logger.debug('Encoded postdata: %s' % data)
            # headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
            request = urllib.request.Request(url, data=data)
        else:
            request = urllib.request.Request(url)

        self.auth(request)

        response = self.opener.open(request)

        return json.loads(response.read().decode("utf-8"))

    def _get_stack_info_url(self, sid):
        """ Use to parse url for retrieving stack infos. """
        return self.djangourl("/" + str(self.project_id) + "/stack/" + str(sid) + "/info")

    def _get_treenode_info_url(self, tn_id):
        """ Use to parse url for retrieving treenode infos. Needs empty post!"""
        return self.djangourl("/" + str(self.project_id) + "/treenodes/" + str(tn_id) + "/info")

    def _get_node_labels_url(self):
        """ Use to parse url for retrieving treenode infos. Needs postdata!"""
        return self.djangourl("/" + str(self.project_id) + "/labels-for-nodes")

    def _get_skeleton_nodes_url(self, skid):
        """ Use to parse url for retrieving skeleton nodes (no info on parents or synapses, does need post data). """
        return self.djangourl("/" + str(self.project_id) + "/treenode/table/" + str(skid) + "/content")

    def _get_skeleton_for_3d_viewer_url(self, skid):
        """ ATTENTION: this url doesn't work properly anymore as of 07/07/14
        use compact-skeleton instead
        Used to parse url for retrieving all info the 3D viewer gets (does NOT need post data)
        Format: name, nodes, tags, connectors, reviews
        """
        return self.djangourl("/" + str(self.project_id) + "/skeleton/" + str(skid) + "/compact-json")

    def _get_add_annotations_url(self):
        """ Use to parse url to add annotations to skeleton IDs. """
        return self.djangourl("/" + str(self.project_id) + "/annotations/add")

    def _get_connectivity_url(self):
        """ Use to parse url for retrieving connectivity (does need post data). """
        return self.djangourl("/" + str(self.project_id) + "/skeletons/connectivity")

    def _get_connectors_url(self):
        """ Use to retrieve list of connectors either pre- or postsynaptic a set of neurons - GET request
        Format: { 'links': [ skeleton_id, connector_id, x,y,z, S(?), confidence, creator, treenode_id, creation_date ] }
        """
        return self.djangourl("/" + str(self.project_id) + "/connectors/")

    def _get_connector_details_url(self):
        """ Use to parse url for retrieving info connectors (does need post data). """
        return self.djangourl("/" + str(self.project_id) + "/connector/skeletons")

    def _get_neuronnames(self):
        """ Use to parse url for names for a list of skeleton ids (does need post data: self.project_id, skid). """
        return self.djangourl("/" + str(self.project_id) + "/skeleton/neuronnames")

    def _get_list_skeletons_url(self):
        """ Use to parse url for names for a list of skeleton ids (does need post data: self.project_id, skid). """
        return self.djangourl("/" + str(self.project_id) + "/skeletons/")

    def _get_graph_dps_url(self):
        """ Use to parse url for getting connections between source and targets. """
        return self.djangourl("/" + str(self.project_id) + "/graph/dps")

    def _get_completed_connector_links(self):
        """ Use to parse url for retrieval of completed connector links by given user 
        GET request: 
        Returns list: [ connector_id, [x,z,y], node1_id, skeleton1_id, link1_confidence, creator_id, [x,y,z], node2_id, skeleton2_id, link2_confidence, creator_id ]
        """
        return self.djangourl("/" + str(self.project_id) + "/connector/list/")

    def _url_to_coordinates(self, coords, stack_id=0, tool='tracingtool', active_skeleton_id=None, active_node_id=None):
        """ Use to generate URL to a location

        Parameters
        ----------
        coords :    list of integers
                    (x, y, z)

        """
        GET_data = {'pid': self.project_id,
                    'xp': coords[0],
                    'yp': coords[1],
                    'zp': coords[2],
                    'tool': tool,
                    'sid0': stack_id,
                    's0': 0
                    }

        if active_skeleton_id:
            GET_data['active_skeleton_id'] = active_skeleton_id
        if active_node_id:
            GET_data['active_node_id'] = active_node_id

        return(self.djangourl('?%s' % urllib.parse.urlencode(GET_data)))

    def _get_user_list_url(self):
        """ Get user list for project. """
        return self.djangourl("/user-list")

    def _get_single_neuronname_url(self, skid):
        """ Use to parse url for a SINGLE neuron (will also give you neuronID). """
        return self.djangourl("/" + str(self.project_id) + "/skeleton/" + str(skid) + "/neuronname")

    def _get_review_status_url(self):
        """ Use to get skeletons review status. """
        return self.djangourl("/" + str(self.project_id) + "/skeletons/review-status")

    def _get_annotation_table_url(self):
        """ Use to get annotations for given neuron. DOES need skid as postdata. """
        return self.djangourl("/" + str(self.project_id) + "/annotations/table-list")

    def _get_intersects(self, vol_id, x, y, z):
        """ Use to test if point intersects with volume. """
        return self.djangourl("/" + str(self.project_id) + "/volumes/" + str(vol_id) + "/intersect") + '?%s' % urllib.parse.urlencode({'skids': x, 'y': y, 'z': z})

    def _get_volumes(self):
        """ Get list of all volumes in project. """
        return self.djangourl("/" + str(self.project_id) + "/volumes/")

    # Get details on a given volume (mesh)
    def _get_volume_details(self, volume_id):
        return self.djangourl("/" + str(self.project_id) + "/volumes/" + str(volume_id))

    def _get_annotations_for_skid_list(self):
        """ ATTENTION: This does not seem to work anymore as of 20/10/2015 -> although it still exists in CATMAID code
            use get_annotations_for_skid_list2    
            Use to get annotations for given neuron. DOES need skid as postdata
        """
        return self.djangourl("/" + str(self.project_id) + "/annotations/skeletons/list")

    def _get_review_details_url(self, skid):
        """ Use to retrieve review status for every single node of a skeleton.      
        For some reason this needs to fetched as POST (even though actual POST data is not necessary)
        Returns list of arbors, the nodes the contain and who has been reviewing them at what time  
        """
        return self.djangourl("/" + str(self.project_id) + "/skeletons/" + str(skid) + "/review")

    def _get_annotations_for_skid_list2(self):
        """ Use to get annotations for given neuron. DOES need skid as postdata. """
        return self.djangourl("/" + str(self.project_id) + "/skeleton/annotationlist")

    def _get_logs_url(self):
        """ Use to get logs. DOES need skid as postdata. """
        return self.djangourl("/" + str(self.project_id) + "/logs/list")

    def _get_annotation_list(self):
        """ Use to parse url for retrieving list of all annotations (and their IDs!!!). """
        return self.djangourl("/" + str(self.project_id) + "/annotations/")

    def _get_contributions_url(self):
        """ Use to parse url for retrieving contributor statistics for given skeleton (does need post data). """
        return self.djangourl("/" + str(self.project_id) + "/skeleton/contributor_statistics_multiple")

    def _get_annotated_url(self):
        """ Use to parse url for retrieving annotated neurons (NEEDS post data). """
        return self.djangourl("/" + str(self.project_id) + "/annotations/query-targets")

    def _get_skid_from_tnid(self, treenode_id):
        """ Use to parse url for retrieving the skeleton id to a single treenode id (does not need postdata) 
        API returns dict: {"count": integer, "skeleton_id": integer}
        """
        return self.djangourl("/" + str(self.project_id) + "/skeleton/node/" + str(treenode_id) + "/node_count")

    def _get_node_list_url(self):
        """ Use to parse url for retrieving list of nodes (NEEDS post data). """
        return self.djangourl("/" + str(self.project_id) + "/node/list")

    def _get_node_info_url(self):
        """ Use to parse url for retrieving user info on a single node (needs post data). """
        return self.djangourl("/" + str(self.project_id) + "/node/user-info")

    def _treenode_add_tag_url(self, treenode_id):
        """ Use to parse url adding labels (tags) to a given treenode (needs post data)."""
        return self.djangourl("/" + str(self.project_id) + "/label/treenode/" + str(treenode_id) + "/update")

    def _connector_add_tag_url(self, treenode_id):
        """ Use to parse url adding labels (tags) to a given treenode (needs post data)."""
        return self.djangourl("/" + str(self.project_id) + "/label/connector/" + str(treenode_id) + "/update")

    def _get_compact_skeleton_url(self, skid, connector_flag=1, tag_flag=1):
        """ Use to parse url for retrieving all info the 3D viewer gets (does NOT need post data).
        Returns, in JSON, [[nodes], [connectors], [tags]], with connectors and tags being empty when 0 == with_connectors and 0 == with_tags, respectively.
        Deprecated but kept for backwards compability!
        """
        return self.djangourl("/" + str(self.project_id) + "/" + str(skid) + "/" + str(connector_flag) + "/" + str(tag_flag) + "/compact-skeleton")

    def _get_compact_details_url(self, skid):
        """ Similar to compact-skeleton but if 'with_history':True is passed as GET request, returned data will include all positions a nodes/connector has ever occupied plus the creation time and last modified.        
        """
        return self.djangourl("/" + str(self.project_id) + "/skeletons/" + str(skid) + "/compact-detail")

    def _get_compact_arbor_url(self, skid, nodes_flag=1, connector_flag=1, tag_flag=1):
        """ The difference between this function and get_compact_skeleton is that the connectors contain the whole chain from the skeleton of interest to the
        partner skeleton: contains [treenode_id, confidence_to_connector, connector_id, confidence_from_connector, connected_treenode_id, connected_skeleton_id, relation1, relation2]
        relation1 = 1 means presynaptic (this neuron is upstream), 0 means postsynaptic (this neuron is downstream)
        """
        return self.djangourl("/" + str(self.project_id) + "/" + str(skid) + "/" + str(nodes_flag) + "/" + str(connector_flag) + "/" + str(tag_flag) + "/compact-arbor")

    def _get_edges_url(self):
        """ Use to parse url for retrieving edges between given skeleton ids (does need postdata).
        Returns list of edges: [source_skid, target_skid, weight]
        """
        return self.djangourl("/" + str(self.project_id) + "/skeletons/confidence-compartment-subgraph")

    def _get_skeletons_from_neuron_id(self, neuron_id):
        """ Use to get all skeletons of a given neuron (neuron_id). """
        return self.djangourl("/" + str(self.project_id) + "/neuron/" + str(neuron_id) + '/get-all-skeletons')

    def _get_history_url(self):
        """ Use to get user history. """
        return self.djangourl("/" + str(self.project_id) + "/stats/user-history")


def _get_urls_threaded(urls, remote_instance, post_data=[], desc='data'):
    """ Wrapper to retrieve a list of urls in parallel using threads

    Parameters
    ----------
    urls :              list of str
                        Urls to retrieve.
    remote_instance :   CATMAID instance
                        Either pass directly to function or define globally 
                        as 'remote_instance'       
    post_data :         list of dicts, optional
                        Needs to be the same size as urls
    desc :              str, optional
                        Description to show on status bar

    Returns
    -------
    data               
                        Data retrieved for each url -> order is kept!
    """

    data = [None for u in urls]
    threads = {}
    threads_closed = []

    if remote_instance.time_out is None:
        time_out = max([len(urls), 30])
    else:
        time_out = remote_instance.time_out

    remote_instance.logger.debug(
        'Creating %i threads to retrieve data' % len(urls))
    for i, url in enumerate(urls):
        if post_data:
            t = _retrieveUrlThreaded(
                url, remote_instance, post_data=post_data[i])
        else:
            t = _retrieveUrlThreaded(url, remote_instance)
        t.start()
        threads[str(i)] = t
        remote_instance.logger.debug('Threads: %i' % len(threads))
    remote_instance.logger.debug('%i threads generated.' % len(threads))

    remote_instance.logger.debug('Joining threads...')

    start = cur_time = time.time()
    joined = 0

    with tqdm(total=len(threads), desc='Fetching %s' % desc) as pbar:
        while cur_time <= (start + time_out) and len([d for d in data if d is not None]) != len(threads):
            for t in threads:
                if t in threads_closed:
                    continue
                if not threads[t].is_alive():
                    # Make sure we keep the order
                    data[int(t)] = threads[t].join()
                    threads_closed.append(t)
            time.sleep(1)
            cur_time = time.time()
            pbar.update(len(threads_closed))

            remote_instance.logger.debug('Closing Threads: %i ( %is until time out )' % (
                len(threads_closed), round(time_out - (cur_time - start))))

    if cur_time > (start + time_out):
        remote_instance.logger.warning('Timeout while joining threads. Retrieved only %i of %i urls' % (
            len([d for d in data if d != None]), len(threads)))
        remote_instance.logger.warning(
            'Consider increasing time to time-out via remote_instance.time_out')
        for t in threads:
            if t not in threads_closed:
                remote_instance.logger.warning(
                    'Did not close thread for url: ' + urls[int(t)])
    else:
        remote_instance.logger.debug(
            'Success! %i of %i urls retrieved.' % (len(threads_closed), len(urls)))

    return data


class _retrieveUrlThreaded(threading.Thread):
    """ Class to retrieve a URL by threading
    """

    def __init__(self, url, remote_instance, post_data=None):
        try:
            self.url = url
            self.post_data = post_data
            threading.Thread.__init__(self)
            self.connector_flag = 1
            self.tag_flag = 1
            self.remote_instance = remote_instance
        except:
            remote_instance.logger.error(
                'Failed to initiate thread for ' + self.url)

    def run(self):
        """
        Retrieve data from single url
        """
        if self.post_data:
            self.data = self.remote_instance.fetch(self.url, self.post_data)
        else:
            self.data = self.remote_instance.fetch(self.url)
        return

    def join(self):
        try:
            threading.Thread.join(self)
            return self.data
        except:
            remote_instance.logger.error(
                'Failed to join thread for ' + self.url)
            return None


def get_neuron(x, remote_instance=None, connector_flag=1, tag_flag=1, get_history=False, get_merge_history=False, get_abutting=False, return_df=False, kwargs={}):
    """ Wrapper to retrieve the skeleton data for a list of skeleton ids

    Parameters
    ----------
    x                 
                        Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        Either pass directly to function or define globally 
                        as 'remote_instance'
    connector_flag :    {0/False,1/True}, optional
                        Set if connector data should be retrieved.                         
                        Note: the CATMAID API endpoint does currently not
                        support retrieving abutting connectors this way.
                        Please use <get_abutting = True> to set an additional 
                        flag.
    tag_flag :          {0/False,1/True}, optional
                        Set if tags should be retrieved. 
                        Possible values = 0/False or 1/True                             
    get_history:        bool, optional
                        If True, the returned skeleton data will contain 
                        creation date ([8]) and last modified ([9]) for each 
                        node -> compact-details url the 'with_history' option 
                        is used in this case
                        ATTENTION: if get_history = True, nodes/connectors 
                        that have been moved since their creation will have 
                        multiple entries reflecting their changes in position! 
                        Each state has the date it was modified as creation 
                        date and the next state's date as last modified. The 
                        most up to date state has the original creation date 
                        as last modified (full circle).
                        The creator_id is always the original creator though.
    get_abutting:       bool, optional
                        If True, will retrieve abutting connectors. 
                        For some reason they are not part of compact-json, 
                        so they have to be retrieved via a separate API endpoint  
                        -> will show up as connector type 3!
    return_df :         bool, optional
                        If True, a ``pandas.DataFrame`` instead of 
                        ``CatmaidNeuron``/``CatmaidNeuronList`` is returned.
    **kwargs           
                        Above boolean parameters can also be passed as dict.
                        This is then used in CatmaidNeuron objects to
                        override implicitly set parameters!

    Returns
    -------
    :class:`pymaid.core.CatmaidNeuron`
                        For single neurons        
    :class:`pymaid.core.CatmaidNeuronList`
                        For a list of neurons

    Notes
    -----    
    The returned objects contain for each neuron::

        neuron_name :           str
        skeleton_id :           str
        nodes / connectors :    pandas.DataFrames containing treenode/connector 
                                ID, coordinates, parent nodes, etc.
        tags :                  dict containing the treenode tags: 
                                { 'tag' : [ treenode_id, treenode_id, ... ] }        

    Dataframe column titles for ``nodes`` and ``connectors`` should be self 
    explanatory with the exception of ``connectors['relation']``::

        connectors['relation']     

                    0 = presynapse
                    1 = postsynapse
                    2 = gap junction
                    3 = abutting connector

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        to_retrieve = [x]
    else:
        to_retrieve = x

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
    urls = []
    for i, skeleton_id in enumerate(to_retrieve):
        # Create URL for retrieving skeleton data from server with history
        # details
        remote_compact_skeleton_url = remote_instance._get_compact_details_url(
            skeleton_id)
        # For compact-details, parameters have to passed as GET
        remote_compact_skeleton_url += '?%s' % urllib.parse.urlencode({'with_history': str(get_history).lower(),
                                                                       'with_tags': str(tag_flag).lower(),
                                                                       'with_connectors': str(connector_flag).lower(),
                                                                       'with_merge_history': str(get_merge_history).lower()})
        # 'True'/'False' needs to be lower case
        urls.append(remote_compact_skeleton_url)

    skdata = _get_urls_threaded(urls, remote_instance, desc='neurons')

    # Retrieve abutting
    if get_abutting:
        remote_instance.logger.debug(
            'Retrieving abutting connectors for %i neurons' % len(to_retrieve))
        urls = []

        for s in to_retrieve:
            get_connectors_GET_data = {'skeleton_ids[0]': str(s),
                                       'relation_type': 'abutting'}
            urls.append(remote_instance._get_connectors_url() + '?%s' %
                        urllib.parse.urlencode(get_connectors_GET_data))
            print(urls)

        cn_data = _get_urls_threaded(urls, remote_instance, desc='abutting')

        # Add abutting to other connectors in skdata with type == 2
        for i, cn in enumerate(cn_data):
            if not get_history:
                skdata[i][1] += [[c[7], c[1], 3, c[2], c[3], c[4]]
                                 for c in cn['links']]
            else:
                skdata[i][1] += [[c[7], c[1], 3, c[2], c[3], c[4], c[8], None]
                                 for c in cn['links']]

    # Get neuron names
    names = get_names(to_retrieve, remote_instance)

    if not get_history:
        df = pd.DataFrame([[
            names[str(to_retrieve[i])],
            str(to_retrieve[i]),
            pd.DataFrame(n[0], columns=['treenode_id', 'parent_id', 'creator_id',
                                        'x', 'y', 'z', 'radius', 'confidence'], dtype=object),
            pd.DataFrame(n[1], columns=['treenode_id', 'connector_id',
                                        'relation', 'x', 'y', 'z'], dtype=object),
            n[2]]
            for i, n in enumerate(skdata)
        ],
            columns=['neuron_name', 'skeleton_id',
                     'nodes', 'connectors', 'tags'],
            dtype=object
        )
    else:
        df = pd.DataFrame([[
            names[str(to_retrieve[i])],
            str(to_retrieve[i]),
            pd.DataFrame(n[0], columns=['treenode_id', 'parent_id', 'creator_id', 'x', 'y',
                                        'z', 'radius', 'confidence', 'last_modified', 'creation_date'], dtype=object),
            pd.DataFrame(n[1], columns=['treenode_id', 'connector_id', 'relation',
                                        'x', 'y', 'z', 'last_modified', 'creation_date'], dtype=object),
            n[2]]
            for i, n in enumerate(skdata)
        ],
            columns=['neuron_name', 'skeleton_id',
                     'nodes', 'connectors', 'tags'],
            dtype=object
        )

    # Placeholder for igraph representations of neurons
    df['igraph'] = None

    if return_df:
        return df

    if df.shape[0] > 1:
        return core.CatmaidNeuronList(df, remote_instance=remote_instance,)
    else:
        return core.CatmaidNeuron(df.ix[0], remote_instance=remote_instance,)


# This is for legacy reasons -> will remove eventually
get_3D_skeleton = get_3D_skeletons = get_neurons = get_neuron


def get_arbor(x, remote_instance=None, node_flag=1, connector_flag=1, tag_flag=1):
    """ Wrapper to retrieve the skeleton data for a list of skeleton ids.    
    Similar to :func:`pymaid.pymaid.get_neuron` but the connector data includes 
    the whole chain::

        treenode1 -> (link_confidence) -> connector -> (link_confidence)
        -> treenode2. 

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
                        Either pass directly to function or define  
                        globally as ``remote_instance``
    connector_flag :    {0,1}, optional
                        Set if connector data should be retrieved. 
    tag_flag :          {0,1}, optional
                        Set if tags should be retrieved.


    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron

        >>> df
        ...  neuron_name   skeleton_id   nodes      connectors   tags
        ... 0  str             str      node_df      conn_df     dict 

    Notes
    -----
    - nodes and connectors are pandas.DataFrames themselves
    - tags is a dict: ``{ 'tag' : [ treenode_id, treenode_id, ... ] }``

    Dataframe column titles should be self explanatory with these exception:

    - conn_df['relation_1'] describes treenode_1 to/from connector
    - conn_df['relation_2'] describes treenode_2 to/from connector
    - 'relations' can be: 0 (presynaptic), 1 (postsynaptic), 2 (gap junction)

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    skdata = []

    for s in tqdm(x, desc='Retrieving arbors'):
        # Create URL for retrieving example skeleton from server
        remote_compact_arbor_url = remote_instance._get_compact_arbor_url(
            s, node_flag, connector_flag, tag_flag)

        # Retrieve node_data for example skeleton
        arbor_data = remote_instance.fetch(remote_compact_arbor_url)

        skdata.append(arbor_data)

        remote_instance.logger.debug('%s retrieved' % str(s))

    names = get_names(x, remote_instance)

    df = pd.DataFrame([[
        names[str(x[i])],
        str(x[i]),
        pd.DataFrame(n[0], columns=['treenode_id', 'parent_id',
                                    'creator_id', 'skids', 'y', 'z', 'radius', 'confidence']),
        pd.DataFrame(n[1], columns=['treenode_1', 'link_confidence', 'connector_id',
                                    'link_confidence', 'treenode_2', 'other_skeleton_id', 'relation_1', 'relation_2']),
        n[2]]
        for i, n in enumerate(skdata)
    ],
        columns=['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags'],
        dtype=object
    )
    return df


def get_partners_in_volume(x, volume, remote_instance=None, threshold=1, min_size=2, approximate=False):
    """ Wrapper to retrieve the synaptic/gap junction partners of neurons 
    of interest **within** a given CATMAID Volume. 

    Important
    ---------
    Connecivity (total number of connections) returned is not restricted to 
    that volume. It just means that at least a single connection is made
    within queried volume.

    Parameters
    ----------
    x              
                        Neurons to check. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    volume :            {str, volume dict, list of str} 
                        Name of the CATMAID volume to test OR dict with 
                        {'vertices':[],'faces':[]} as returned by e.g. 
                        :func:`pymaid.pymaid.get_volume()`
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``                        
    threshold :         int, optional
                        Does not seem to have any effect on CATMAID API and is 
                        therefore filtered afterwards. This threshold is 
                        applied to the TOTAL number of synapses across all
                        neurons!  
    min_size :          int, optional
                        Minimum node count of partner
                        (default = 2 -> hide single-node partner)
    approximate :       bool, optional
                        If True, bounding box around the volume is used. Will
                        speed up calculations a lot!

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of 
        synapses with the query neurons:

        >>> df
        ...  neuron_name  skeleton_id  num_nodes   relation     skid1  skid2 ...
        ... 1  name1         skid1    node_count1  upstream     n_syn  n_syn ...
        ... 2  name2         skid2    node_count2  downstream   n_syn  n_syn ...   
        ... 3  name3         skid3    node_count3  gapjunction  n_syn  n_syn ...        

        - Relation can be: upstream (incoming), downstream (outgoing) of the neurons of interest or gap junction
        - partners can show up multiple times if they are e.g. pre- AND postsynaptic
        - the number of connections between two partners is not restricted to the volume

    See Also
    --------
    :func:`pymaid.pymaid.get_neurons_in_volume`
                            Get neurons within given volume
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    # First, get list of connectors
    cn_data = get_connectors(x, remote_instance=remote_instance,
                             incoming_synapses=True, outgoing_synapses=True,
                             abutting=False, gap_junctions=True, )

    remote_instance.logger.info(
        '%i connectors retrieved - now checking for intersection with volume...' % cn_data.shape[0])

    # Find out which connectors are in the volume of interest
    iv = morpho.in_volume(
        cn_data[['skids', 'y', 'z']], volume, remote_instance, approximate=approximate)

    # Get the subset of connectors within the volume
    cn_in_volume = cn_data[iv].copy()

    remote_instance.logger.info(
        '%i connectors in volume - retrieving connected neurons...' % cn_in_volume.shape[0])

    # Get details and extract connected x
    cn_details = get_connector_details(
        cn_in_volume.connector_id.unique().tolist(), remote_instance=remote_instance,)
    skids_in_volume = list(set(cn_details.presynaptic_to.tolist(
    ) + [n for l in cn_details.postsynaptic_to.tolist() for n in l]))

    skids_in_volume = [str(s) for s in skids_in_volume]

    # Get all connectivity
    connectivity = get_partners(x, remote_instance=remote_instance,
                                threshold=threshold,
                                min_size=min_size)

    # Filter and return connectivity
    filtered_connectivity = connectivity[
        connectivity.skeleton_id.isin(skids_in_volume)].copy().reset_index()

    remote_instance.logger.info('%i unique partners left after filtering (%i of %i connectors in given volume)' % (
        len(filtered_connectivity.skeleton_id.unique()), cn_in_volume.shape[0], cn_data.shape[0]))

    return filtered_connectivity


def get_partners(x, remote_instance=None, threshold=1,  min_size=2, filt=[], directions=['incoming', 'outgoing']):
    """ Wrapper to retrieve the synaptic/gap junction partners of neurons 
    of interest

    Parameters
    ----------
    x            
                        Neurons for which to retrieve partners. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        Either pass directly to function or define  
                        globally as ``remote_instance``
    threshold :         int, optional
                        Does not seem to have any effect on CATMAID API and is 
                        therefore filtered afterwards. This threshold is 
                        applied to the total number of synapses. 
    min_size :          int, optional
                        Minimum node count of partner
                        (default = 2 -> hides single-node partners!)
    filt :              list of str, optional, default = []
                        Filters partners for neuron names (must be exact) or
                        skeleton_ids
    directions :        list of str, optional
                        Use to restrict to either up- or downstream partners

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of 
        synapses with the query neurons:

        >>> df
        ...   neuron_name  skeleton_id    num_nodes    relation    skid1  skid2 ....
        ... 0   name1         skid1      node_count1  upstream     n_syn  n_syn ...
        ... 1   name2         skid2      node_count2  downstream   n_syn  n_syn ..   
        ... 2   name3         skid3      node_count3  gapjunction  n_syn  n_syn .            
        ... ...

        ``relation`` can be ``upstream`` (incoming), ``downstream`` (outgoing) 
        or ``gapjunction`` (gap junction)

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
    >>> #Get only upstream partners
    >>> subset = cn[ cn.relation == 'upstream' ]
    >>> #Get partners with more than e.g. 5 synapses across all neurons
    >>> subset2 = cn[ cn[ example_skids ].sum(axis=1) > 5 ]
    >>> #Combine above conditions (watch parentheses!)
    >>> subset3 = cn[ (cn.relation=='upstream') & 
    ... (cn[example_skids].sum(axis=1) > 5) ]

    See Also
    --------
    :func:`pymaid.cluster.create_adjacency_matrix`
                        Use if you need an adjacency matrix instead of a table
    :func:`pymaid.pymaid.get_partners_in_volume`
                        Use if you only want partners within a given volume
    """

    def _constructor_helper(entry, skid):
        """ Helper to extract connectivity from data returned by CATMAID server
        """
        try:
            return entry['skids'][str(skid)]
        except:
            return 0

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    remote_connectivity_url = remote_instance._get_connectivity_url()

    connectivity_post = {}
    connectivity_post['boolean_op'] = 'OR'
    connectivity_post['with_nodes'] = False

    for i, skid in enumerate(x):
        tag = 'source_skeleton_ids[%i]' % i
        connectivity_post[tag] = skid

    remote_instance.logger.info('Fetching connectivity')
    connectivity_data = remote_instance.fetch(
        remote_connectivity_url, connectivity_post)

    # Delete directions that we don't want
    connectivity_data.update(
        {d: [] for d in connectivity_data if d not in directions})

    # As of 08/2015, # of synapses is returned as list of nodes with 0-5 confidence: {'skid': [0,1,2,3,4,5]}
    # This is being collapsed into a single value before returning it:

    for d in connectivity_data:
        pop = set()
        for entry in connectivity_data[d]:
            if sum([sum(connectivity_data[d][entry]['skids'][n]) for n in connectivity_data[d][entry]['skids']]) >= threshold:
                for skid in connectivity_data[d][entry]['skids']:
                    connectivity_data[d][entry]['skids'][skid] = sum(
                        connectivity_data[d][entry]['skids'][skid])
            else:
                pop.add(entry)

            if min_size > 1:
                if connectivity_data[d][entry]['num_nodes'] < min_size:
                    pop.add(entry)

        for n in pop:
            connectivity_data[d].pop(n)

    names = get_names([n for d in connectivity_data for n in connectivity_data[
                      d]] + x, remote_instance)

    remote_instance.logger.info('Done. Found %i up- %i downstream neurons' % (
        len(connectivity_data['incoming']), len(connectivity_data['outgoing'])))

    df = pd.DataFrame(columns=['neuron_name', 'skeleton_id',
                               'num_nodes', 'relation'] + [str(s) for s in x])

    relations = {
        'incoming': 'upstream',
        'outgoing': 'downstream',
                    'gapjunctions': 'gapjunction'
    }

    for d in relations:
        df_temp = pd.DataFrame([[
            names[str(n)],
            str(n),
            int(connectivity_data[d][n]['num_nodes']),
            relations[d]]
            + [_constructor_helper(connectivity_data[d][n], s) for s in x]
            for i, n in enumerate(connectivity_data[d])
        ],
            columns=['neuron_name', 'skeleton_id', 'num_nodes',
                     'relation'] + [str(s) for s in x],
            dtype=object
        )

        df = pd.concat([df, df_temp], axis=0)

    if filt:
        if not isinstance(filt, (list, np.ndarray)):
            filt = [filt]

        filt = [str(s) for s in f]

        df = df[df.skeleton_id.isin(filt) | df.neuron_name.isin(filt)]

    # Return reindexed concatenated dataframe
    return df.reset_index(drop=True)


def get_names(x, remote_instance=None):
    """ Wrapper to retrieve neurons names for a list of skeleton ids

    Parameters
    ----------
    x              
                        Neurons for wich to retrieve names. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    ------- 
    dict            
                        ``{ skid1 : 'neuron_name', skid2 : 'neuron_name',  .. }``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    x = list(set(x))

    remote_get_names_url = remote_instance._get_neuronnames()

    get_names_postdata = {}
    get_names_postdata['self.project_id'] = remote_instance.project_id

    for i in range(len(x)):
        key = 'skids[%i]' % i
        get_names_postdata[key] = x[i]

    names = remote_instance.fetch(remote_get_names_url, get_names_postdata)

    remote_instance.logger.debug(
        'Names for %i of %i skeleton IDs retrieved' % (len(names), len(x)))

    return(names)


def get_node_user_details(treenode_ids, remote_instance=None):
    """ Wrapper to retrieve user info for a list of treenode and/or connectors

    Parameters
    ----------
    treenode_ids :      list
                        list of treenode ids (can also be connector ids!)
    remote_instance :   CATMAID instance
                        Either pass directly to function or define globally as 
                        ``remote_instance``

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a treenode

        >>> df
        ...   treenode_id  creation_time  user  edition_time
        ... 0
        ... 1
        ...   editor  reviewers  review_times
        ... 0
        ... 1


    """

    if type(treenode_ids) != type(list()):
        treenode_ids = [treenode_ids]

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_instance.logger.info(
        'Retrieving details for %i nodes...' % len(treenode_ids))

    remote_nodes_details_url = remote_instance._get_node_info_url()

    get_node_details_postdata = {
    }

    for i, tn in enumerate(treenode_ids):
        key = 'node_ids[%i]' % i
        get_node_details_postdata[key] = tn

    data = remote_instance.fetch(
        remote_nodes_details_url, get_node_details_postdata)

    data_columns = ['creation_time', 'user',  'edition_time',
                    'editor', 'reviewers', 'review_times']

    df = pd.DataFrame(
        [[e] + [data[e][k] for k in data_columns] for e in data.keys()],
        columns=['treenode_id'] + data_columns,
        dtype=object
    )

    df['creation_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['creation_time'].tolist()]
    df['edition_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['edition_time'].tolist()]
    df['review_times'] = [[datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in lst] for lst in df['review_times'].tolist()]

    return df


def get_treenode_table(x, remote_instance=None):
    """ Wrapper to retrieve treenode table(s) for a list of neurons

    Parameters
    ----------
    x                 
                        Catmaid Neuron(s) as single or list of either:
                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a treenode::

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

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    remote_instance.logger.info(
        'Retrieving %i treenode table(s)...' % len(x))

    user_list = get_user_list(remote_instance)

    user_dict = user_list.set_index('id').T.to_dict()

    # Generate URLs to retrieve
    urls = []
    for i, skeleton_id in enumerate(list(set(x))):
        remote_nodes_list_url = remote_instance._get_skeleton_nodes_url(
            skeleton_id)
        urls.append(remote_nodes_list_url)

    node_list = _get_urls_threaded(urls, remote_instance, desc='tn tables')

    # Format of node_list: treenode_id, parent_node_id, confidence, x, y, z,
    # radius, creator, last_edited
    tag_dict = {n[0]: [] for nl in node_list for n in nl[0]}
    [tag_dict[n[0]].append(n[1]) for nl in node_list for n in nl[2]]

    reviewer_dict = {n[0]: [] for nl in node_list for n in nl[0]}
    [reviewer_dict[n[0]].append(user_list[user_list.id == n[1]]['login'].values[
                                0]) for nl in node_list for n in nl[1]]

    tn_table = pd.DataFrame([[x[i]] + n + [reviewer_dict[n[0]], tag_dict[n[0]]] for nl in node_list for n in nl[0]],
                            columns=['skeleton_id', 'treenode_id', 'parent_node_id', 'confidence',
                                     'skids', 'y', 'z', 'radius', 'creator', 'last_edited', 'reviewers', 'tags'],
                            dtype=object
                            )

    # Replace creator_id with their login
    tn_table['creator'] = [user_dict[u]['login'] for u in tn_table['creator']]

    # Replace timestamp with datetime object
    tn_table['last_edited'] = [datetime.datetime.fromtimestamp(
        int(t)) for t in tn_table['last_edited']]

    return tn_table


def get_edges(x, remote_instance=None):
    """ Wrapper to retrieve edges (synaptic connections) between sets of neurons

    Parameters
    ----------
    x
                        Neurons for which to retrieve edges. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    -------
    pandas.DataFrame 
        DataFrame in which each row represents an edge::

        >>> df
        ...   source_skid     target_skid     weight 
        ... 1     
        ... 2     
        ... 3     

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

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


def get_connectors(x, remote_instance=None, incoming_synapses=True, outgoing_synapses=True, abutting=False, gap_junctions=False):
    """ Wrapper to retrieve connectors for a set of neurons.    

    Parameters
    ----------
    x             
                        Neurons for which to retrieve connectors. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
                        Either pass directly to function or define 
                        globally as 'remote_instance'
    incoming_synapses : bool, optional
                        if True, incoming synapses will be retrieved
    outgoing_synapses : bool, optional
                        if True, outgoing synapses will be retrieved
    abutting :          bool, optional
                        if True, abutting connectors will be retrieved
    gap_junctions :     bool, optional
                        if True, gap junctions will be retrieved

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a connector::

        >>> df
        ... skeleton_id  connector_id  x  y  z  confidence  creator_id  
        ... 0
        ... 1
        ...        
        ... treenode_id  creation_time  edition_time type
        ... 0
        ... 1        

    Notes
    -----
    DataFrame in which each row represents a link (connector <-> treenode)! 
    Connectors may thus show up in multiple rows. Use e.g. 
    ``df.connector_id.unique()`` to get a set of unique connector IDs.
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    get_connectors_GET_data = {'with_tags': 'false'}

    cn_data = []

    # There seems to be some cap regarding how many x you can send to the
    # server, so we have to chop it into pieces
    for a in range(0, len(x), 50):
        for i, s in enumerate(x[a:a + 50]):
            tag = 'skeleton_ids[%i]' % i
            get_connectors_GET_data[tag] = str(s)

        if incoming_synapses is True:
            get_connectors_GET_data['relation_type'] = 'presynaptic_to'
            remote_get_connectors_url = remote_instance._get_connectors_url(
            ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [e + ['presynaptic_to']
                        for e in remote_instance.fetch(remote_get_connectors_url)['links']]

        if outgoing_synapses is True:
            get_connectors_GET_data['relation_type'] = 'postsynaptic_to'
            remote_get_connectors_url = remote_instance._get_connectors_url(
            ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [e + ['postsynaptic_to']
                        for e in remote_instance.fetch(remote_get_connectors_url)['links']]

        if abutting is True:
            get_connectors_GET_data['relation_type'] = 'abutting'
            remote_get_connectors_url = remote_instance._get_connectors_url(
            ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [e + ['abutting']
                        for e in remote_instance.fetch(remote_get_connectors_url)['links']]

        if gap_junctions is True:
            get_connectors_GET_data['relation_type'] = 'gapjunction_with'
            remote_get_connectors_url = remote_instance._get_connectors_url(
            ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [e + ['gap_junction']
                        for e in remote_instance.fetch(remote_get_connectors_url)['links']]

    df = pd.DataFrame(cn_data,
                      columns=['skeleton_id', 'connector_id', 'skids', 'y', 'z', 'confidence',
                               'creator_id', 'treenode_id', 'creation_time', 'edition_time', 'type'],
                      dtype=object
                      )

    remote_instance.logger.info(
        '%i connectors for %i neurons retrieved' % (df.shape[0], len(x)))

    return df


def get_connector_details(connector_ids, remote_instance=None):
    """ Wrapper to retrieve details on sets of connectors 

    Parameters
    ----------
    connector_ids :     list of connector ids 
                        Can be found e.g. in compact skeletons (get_neuron)
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define globally as 
                        'remote_instance'

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a connector

        >>> df    
        ...   connector_id  presynaptic_to  postsynaptic_to  presynaptic_to_node
        ... 0
        ... 1
        ... 2
        ...
        ... postsynaptic_to_node
        ... 0
        ... 1
        ... 2    
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_connectors_url = remote_instance._get_connector_details_url()

    # Depending on DATA_UPLOAD_MAX_NUMBER_FIELDS of your CATMAID server
    # (default = 1000), we have to cut requests into batches < 1000
    connectors = []
    # for b in range( 0, len( connector_ids ), 999 ):
    get_connectors_postdata = {}
    for i, s in enumerate(connector_ids):
        key = 'connector_ids[%i]' % i
        get_connectors_postdata[key] = s  # connector_ids[i]

    connectors += remote_instance.fetch(remote_get_connectors_url,
                                        get_connectors_postdata)

    remote_instance.logger.info('Data for %i of %i unique connector IDs retrieved' % (
        len(connectors), len(set(connector_ids))))

    columns = ['connector_id', 'presynaptic_to', 'postsynaptic_to',
               'presynaptic_to_node', 'postsynaptic_to_node']

    df = pd.DataFrame([[cn[0]] + [cn[1][e] for e in columns[1:]] for cn in connectors],
                      columns=columns,
                      dtype=object
                      )

    return df


def get_review(x, remote_instance=None):
    """ Wrapper to retrieve review status for a set of neurons

    Parameters
    ----------
    x
                        Neurons for which to get review status. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a neuron

        >>> df
           skeleton_id neuron_name total_node_count nodes_reviewed percent_reviewed
        0    
        1
        2

    See Also
    --------
    :func:`pymaid.pymaid.get_review_details`
        Gives you review status for individual nodes of a given neuron   

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    remote_get_reviews_url = remote_instance._get_review_status_url()

    get_review_postdata = {}

    for i in range(len(x)):
        key = 'skeleton_ids[%i]' % i
        get_review_postdata[key] = str(x[i])

    names = get_names(x, remote_instance)

    review_status = remote_instance.fetch(
        remote_get_reviews_url, get_review_postdata)

    df = pd.DataFrame([[s,
                        names[str(s)],
                        review_status[s][0],
                        review_status[s][1],
                        int(review_status[s][1] / review_status[s][0] * 100)
                        ] for s in review_status],
                      columns=['skeleton_id', 'neuron_name', 'total_node_count',
                               'nodes_reviewed', 'percent_reviewed']
                      )

    return df


def add_annotations(x, annotations, remote_instance=None):
    """ Wrapper to add annotation(s) to a list of neuron(s)

    Parameters
    ----------
    x                  
                        Neurons to add new annotation(s). Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    annotations :       list
                        Annotation(s) to add to neurons provided
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'

    Returns
    -------
    Nothing
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    if type(annotations) != type(list()):
        annotations = [annotations]

    add_annotations_url = remote_instance._get_add_annotations_url()

    add_annotations_postdata = {}

    for i in range(len(x)):
        key = 'skeleton_ids[%i]' % i
        add_annotations_postdata[key] = str(x[i])

    for i in range(len(annotations)):
        key = 'annotations[%i]' % i
        add_annotations_postdata[key] = str(annotations[i])

    remote_instance.logger.info(remote_instance.fetch(
        add_annotations_url, add_annotations_postdata))

    return


def get_user_annotations(x, remote_instance=None):
    """ Wrapper to retrieve annotations used by given user(s).     

    Parameters
    ----------
    x                  
                        Users to get annotationfor. Can be either:

                        1. single or list of user IDs 
                        2. single or list of user login names

    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'

    Returns
    ------- 
    pandas.DataFrame
        DataFrame (df) in which each row represents a single annotation

        >>> df
           annotation annotated_on times_used user_id annotation_id user_login
        0
        1        

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    # Get user list
    user_list = get_user_list(remote_instance=remote_instance)

    try:
        ids = [int(e) for e in x]
    except:
        ids = [user_list.set_index('login').ix[e] for e in x]

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
    annotations = [e['aaData'] for e in _get_urls_threaded(
        url_list, remote_instance, post_data=postdata, desc='annotations')]

    # Add user login
    for i, u in enumerate(ids):        
        for an in annotations[i]:
            an.append(user_list.set_index('id').ix[u].login)

    # Now flatten the list of lists
    annotations = [an for sublist in annotations for an in sublist]

    # Create dataframe
    df = pd.DataFrame(annotations,
                      columns=['annotation', 'annotated_on', 'times_used', 
                               'user_id' , 'annotation_id' ,'user_login'],
                      dtype=object
                      )

    df['annotated_on'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['annotated_on'].tolist()]

    return df.sort_values('times_used').reset_index(drop=True)


def get_annotation_details(x, remote_instance=None):
    """ Wrapper to retrieve annotations for a set of neuron. Returns more
    details than :func:`pymaid.pymaid.get_annotations` but is slower:
    Contains timestamps and user_id (same API as neuron navigator)

    Parameters
    ----------
    x                  
                        Neurons to get annotation details for. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object                        
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'

    Returns
    ------- 
    pandas.DataFrame
        DataFrame in which each row represents a single annotation

        >>> df
        ...   annotation skeleton_id time_annotated unknown user_id annotation_id user
        ... 0
        ... 1    

    See Also
    --------
    :func:`pymaid.pymaid.get_annotations`
                        Gives you annotations for a list of neurons (faster)

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    skids = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(skids, (list, np.ndarray)):
        skids = [skids]

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
    annotations = [e['aaData'] for e in _get_urls_threaded(
        url_list, remote_instance, post_data=postdata, desc='annotations')]

    # Get user list
    user_list = get_user_list(remote_instance).set_index('id')

    # Add skeleton ID and user login
    for i, s in enumerate(skids):
        for an in annotations[i]:
            an.insert(1, s)
            an.append(user_list.ix[an[3]].login)

    # Now flatten the list of lists
    annotations = [an for sublist in annotations for an in sublist]

    # Create dataframe
    df = pd.DataFrame(annotations,
                      columns=['annotation', 'skeleton_id', 'time_annotated',
                               'unknown', 'user_id', 'annotation_id', 'user'],
                      dtype=object
                      )

    df['time_annotated'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['time_annotated'].tolist()]

    return df.sort_values('annotation').reset_index(drop=True)


def get_annotations(x, remote_instance=None):
    """ Wrapper to retrieve annotations for a list of skeleton ids.
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
                        Either pass directly to function or define  
                        globally as 'remote_instance'

    Returns
    -------
    dict 
                        ``{ skeleton_id : [ annnotation, annotation ], ... }``

    See Also
    --------
    :func:`pymaid.pymaid.get_annotation_details`
                        Gives you more detailed information about annotations
                        of a set of neuron (includes timestamp and user) but 
                        is slower.
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

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
    except:
        remote_instance.logger.error(
            'No annotations retrieved. Make sure that the skeleton IDs exist.')
        raise Exception(
            'No annotations retrieved. Make sure that the skeleton IDs exist.')


def get_annotation_id(annotations, remote_instance=None,  allow_partial=False):
    """ Wrapper to retrieve the annotation ID for single or list of annotation(s)

    Parameters
    ----------
    annotations :       {str,list}
                        Single annotations or list of multiple annotations
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'
    allow_partial :     bool
                        If True, will allow partial matches

    Returns
    -------
    dict 
                        ``{ 'annotation_name' : 'annotation_id', ....}``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_instance.logger.debug('Retrieving list of annotations...')

    remote_annotation_list_url = remote_instance._get_annotation_list()
    annotation_list = remote_instance.fetch(remote_annotation_list_url)

    annotation_ids = {}
    annotations_matched = set()

    if type(annotations) == type(str()):
        for d in annotation_list['annotations']:
            if d['name'] == annotations and allow_partial is False:
                annotation_ids[d['name']] = d['id']
                remote_instance.logger.debug(
                    'Found matching annotation: %s' % d['name'])
                annotations_matched.add(d['name'])
                break
            elif annotations in d['name'] and allow_partial is True:
                annotation_ids[d['name']] = d['id']
                remote_instance.logger.debug(
                    'Found matching annotation: %s' % d['name'])

        if not annotation_ids:
            remote_instance.logger.warning(
                'Could not retrieve annotation id for: ' + annotations)

    elif type(annotations) == type(list()):
        for d in annotation_list['annotations']:
            if d['name'] in annotations and allow_partial is False:
                annotation_ids[d['name']] = d['id']
                annotations_matched.add(d['name'])
                remote_instance.logger.debug(
                    'Found matching annotation: %s' % d['name'])
            elif True in [a in d['name'] for a in annotations] and allow_partial is True:
                annotation_ids[d['name']] = d['id']
                annotations_matched |= set(
                    [a for a in annotations if a in d['name']])
                remote_instance.logger.debug(
                    'Found matching annotation: %s' % d['name'])

        if len(annotations) != len(annotations_matched):
            remote_instance.logger.warning('Could not retrieve annotation id(s) for: ' + str(
                [a for a in annotations if a not in annotations_matched]))

    return annotation_ids


def has_soma(x, remote_instance=None, tag='soma', min_rad=500):
    """ Quick function to check if a neuron/a list of neurons have somas. 

    Parameters
    ----------
    x                   
                        Neurons which to check for a soma. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'
    tag :               {str, None}, optional
                        Tag we expect the soma to have. Set to ``None`` if
                        not applicable.
    min_rad :           int, optional
                        Minimum radius of soma.

    Returns
    -------
    dict 
                        ``{ 'skid1' : True, 'skid2' : False, ...}``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    skdata = get_neuron(x, remote_instance=remote_instance,
                        connector_flag=0, tag_flag=1,
                        get_history=False,
                        )

    d = {}
    for s in skdata:
        if tag:
            if tag in s.tags:
                tn_with_tag = s.tags['soma']
            else:
                tn_with_tag = []
        else:
            tn_with_tag = s.nodes.treenode_id.tolist()

        tn_with_rad = s.nodes[s.nodes.radius > min_rad].treenode_id.tolist()

        if set(tn_with_tag) & set(tn_with_rad):
            d[s.skeleton_id] = True
        else:
            d[s.skeleton_id] = False

    return d


def get_skids_by_name(names, remote_instance=None, allow_partial=True):
    """ Wrapper to retrieve the all neurons with matching name

    Parameters
    ----------
    names :             {str, list of str}
                        Name(s) to search for
    allow_partial :     bool, optional
                        If True, partial matches are returned too    
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define globally as 
                        'remote_instance'

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron        

        >>> df
           name   skeleton_id
        0
        1
        2

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if isinstance(names, str):
        names = [names]

    urls = []
    post_data = []

    for n in names:
        urls.append(remote_instance._get_annotated_url())
        post_data.append({'name': str(n), 'rangey_start': 0,
                          'range_length': 500, 'with_annotations': False}
                         )

    results = _get_urls_threaded(
        urls, remote_instance, post_data=post_data, desc='names')

    match = []
    for i, r in enumerate(results):
        for e in r['entities']:
            if allow_partial and e['type'] == 'neuron' and names[i].lower() in e['name'].lower():
                match.append([e['name'], e['skeleton_ids'][0]])
            if not allow_partial and e['type'] == 'neuron' and e['name'] == names[i]:
                match.append([e['name'], e['skeleton_ids'][0]])

    df = pd.DataFrame(match,
                      columns=['name', 'skeleton_id']
                      )

    return df.sort_values(['name']).reset_index(drop=True)


def get_skids_by_annotation(annotations, remote_instance=None, allow_partial=False):
    """ Wrapper to retrieve the all neurons annotated with given annotation(s)

    Parameters
    ----------
    annotations :           {str,list}
                            single annotation or list of multiple annotations    
    remote_instance :       CATMAID instance, optional
                            Either pass directly to function or define globally 
                            as 'remote_instance'
    allow_partial :         bool, optional
                            If True, allow partial match of annotation

    Returns
    -------
    list :
                            ``[skid1, skid2, skid3 ]``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_instance.logger.info(
        'Looking for Annotation(s): ' + str(annotations))
    annotation_ids = get_annotation_id(
        annotations, remote_instance,  allow_partial=allow_partial)

    if not annotation_ids:
        remote_instance.logger.error(
            'No matching annotation found! Returning None')
        raise Exception('No matching annotation found!')

    if allow_partial is True:
        remote_instance.logger.debug(
            'Found id(s): %s (partial matches included)' % len(annotation_ids))
    elif type(annotations) == type(list()):
        remote_instance.logger.debug('Found id(s): %s | Unable to retrieve: %i' % (
            str(annotation_ids), len(annotations) - len(annotation_ids)))
    elif type(annotations) == type(str()):
        remote_instance.logger.debug('Found id: %s | Unable to retrieve: %i' % (
            list(annotation_ids.keys())[0], 1 - len(annotation_ids)))

    annotated_skids = []
    remote_instance.logger.debug(
        'Retrieving x for annotationed neurons...')
    for an_id in tqdm(annotation_ids.values(),desc='annotations'):
        #annotation_post = {'neuron_query_by_annotation': annotation_id, 'display_start': 0, 'display_length':500}
        annotation_post = {'annotated_with0': an_id, 'rangey_start': 0,
                           'range_length': 500, 'with_annotations': False}
        remote_annotated_url = remote_instance._get_annotated_url()
        neuron_list = remote_instance.fetch(
            remote_annotated_url, annotation_post)
        count = 0
        for entry in neuron_list['entities']:
            if entry['type'] == 'neuron':
                annotated_skids.append(str(entry['skeleton_ids'][0]))

    remote_instance.logger.info(
        'Found %i skeletons with matching annotation(s)' % len(annotated_skids))

    return(annotated_skids)


def neuron_exists(x, remote_instance=None):
    """ Quick function to check if neurons exist in CATMAID

    Parameters
    ----------
    x
                        Neurons to check if they exist in Catmaid. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'

    Returns
    -------
    bool :
                        True if skeleton exists, False if not. If multiple
                        neurons are queried, returns a dict 
                        ``{ skid1 : True, skid2 : False, ... }``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if isinstance(x, (list, np.ndarray)):
        return {n: neuron_exists(n) for n in x}

    remote_get_neuron_name = remote_instance._get_single_neuronname_url(x)
    response = remote_instance.fetch(remote_get_neuron_name)

    if 'error' in response:
        return False
    else:
        return True


def get_treenode_info( treenode_ids, remote_instance =None ):
    """ This wrapper will retrieve info for a set of treenodes.

    Parameters
    ----------
    treenode_ids
                        single or list of treenode IDs
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    -------
    pandas DataFrame
                DataFrame in which rach row represents a queried treenode:

                >>> df
                  treenode_id neuron_name skeleton_id skeleton_name neuron_id 
                0
                1

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if not isinstance(treenode_ids, (list, np.ndarray)):
        treenode_ids = [ treenode_ids ]
    
    urls = [ remote_instance._get_treenode_info_url( tn ) for tn in treenode_ids ]
    post_data = [ {'None':0} for tn in treenode_ids ]

    data = _get_urls_threaded(urls, remote_instance,
                           post_data=post_data, desc='info')

    df = pd.DataFrame( [ [ treenode_ids[i] ] + list(n.values()) for i,n in enumerate(data) ],
                        columns = ['treenode_id'] + list( data[0].keys() )
                    )

    return df


def get_node_tags( node_ids, node_type, remote_instance =None ):
    """ This wrapper will retrieve labels (tags) for a set of treenodes.

    Parameters
    ----------
    node_ids
                        single or list of treenode or connector IDs
    node_type :         {'TREENODE','CONNECTOR'}
                        Set which node type of IDs you have provided as they use
                        different API endpoints!      
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    -------
    dict
                dictionary (d) containing tags for each node

    Examples
    --------
    >>> pymaid.get_node_tags( ['6626578', '6633237'] 
    ...                        'TREENODE',
    ...                        remote_instance )
    {'6633237': ['ends'], '6626578': ['ends'] } 

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if not isinstance(node_ids, (list, np.ndarray)):
        node_ids = [ node_ids ]

    #Make sure node_ids are strings
    node_ids = [ str(n) for n in node_ids ]
    
    url = remote_instance._get_node_labels_url()

    if node_type in ['TREENODE','TREENODES']:
        key = 'treenode_ids'
    elif node_type in ['CONNECTOR','CONNECTORS']:
        key = 'connector_ids'
    else:
        raise TypeError('Unknown node_type parameter: %s' % str( node_type) )
    
    post_data = { key : ','.join( [ str(tn) for tn in node_ids ] ) }

    return remote_instance.fetch( url, post = post_data )
    

def delete_tags(node_list, tags, node_type, remote_instance=None):
    """ Wrapper to remove tag(s) for a list of treenode(s) or connector(s).
    Works by getting existing tags, removing given tag(s) and then using 
    pymaid.add_tags() to push updated tags back to CATMAID.    

    Parameters
    ----------
    node_list :         list
                        Treenode or connector IDs to delete tags from
    tags :              list
                        Tags(s) to delete from provided treenodes/connectors.
                        Use ``tags=None`` and to remove all tags from a set of 
                        nodes.
    node_type :         {'TREENODE','CONNECTOR'}
                        Set which node type of IDs you have provided as they use
                        different API endpoints!
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    -------
    str :
                        Confirmation from Catmaid server

    See Also
    --------
    :func:`pymaid.pymaid.add_tags`
            Function to add tags to nodes.

    Examples
    --------
    Use this to clean up end-related tags from non-end treenodes
    >>> from pymaid import pymaid
    >>> # Load neuron
    >>> n = pymaid.get_neuron( 16 )
    >>> # Get non-end nodes 
    >>> non_leaf_nodes = n.nodes[ n.nodes.type != 'leaf' ]
    >>> # Define which tags to remove
    >>> tags_to_remove = ['ends','uncertain end','uncertain continuation','TODO']
    >>> # Remove tags
    >>> resp = pymaid.delete_tags(  non_leaf_nodes.treenode_id.tolist(),
    ...                             tags_to_remove,
    ...                             'TREENODE')
    2017-08-09 14:08:36,102 - pymaid.pymaid - WARNING - Skipping 8527 nodes without tags
    >>> # Above warning means that most nodes did not have ANY tags

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return   

    if not isinstance(node_list, (list, np.ndarray)):
        node_list = [node_list]

    #Make sure node list is strings
    node_list = [str(n) for n in node_list]

    if not isinstance(tags, (list, np.ndarray)):
        tags = [tags]

    if tags != [None]:
        # First, get existing tags for these nodes
        existing_tags = get_node_tags( node_list, node_type, remote_instance=remote_instance )

        # Check if our treenodes actually exist
        if [ n for n in node_list if n not in existing_tags ]:
            remote_instance.logger.warning('Skipping %i nodes without tags' % len([ n for n in node_list if n not in existing_tags ]))
            [ node_list.remove(n) for n in [ n for n in node_list if n not in existing_tags ] ]

        # Remove tags from that list that we want to have deleted
        existing_tags = { n : [ t for t in existing_tags[n] if t not in tags ] for n in node_list }
    else:
        existing_tags = ''    

    # Use the add_tags function to override existing tags
    return add_tags(node_list, existing_tags, node_type, remote_instance=remote_instance, override_existing=True)    


def add_tags(node_list, tags, node_type, remote_instance=None, override_existing=False):
    """ Wrapper to add or edit tag(s) for a list of treenode(s) or connector(s)    

    Parameters
    ----------
    node_list :         list
                        Treenode or connector IDs to edit
    tags :              {str, list, dict}
                        Tags(s) to add to provided treenode/connector ids. If 
                        a dictionary is provided `{node_id1: [tag1,tag2], ...}` 
                        each node gets individual tags. If string or list
                        are provided, all nodes will get the same tags. 
    node_type :         {'TREENODE','CONNECTOR'}
                        Set which node type of IDs you have provided as they use
                        different API endpoints!
    override_existing : bool, default=False
                        This needs to be set to True if you want to delete a tag.
                        Otherwise, your tags (even if empty) will not override
                        existing tags.
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    ------- 
    str :
                        Confirmation from Catmaid server

    Notes
    -----
    Use ``tags = ''`` and ``override_existing=True`` to delete all tags from
    nodes.

    See Also
    --------
    :func:`pymaid.pymaid.delete_tags`
            Function to delete given tags from nodes.

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if not isinstance(node_list, (list, np.ndarray)):
        node_list = [node_list]

    if not isinstance(tags, (list, np.ndarray, dict)):
        tags = [tags]

    if node_type in ['TREENODE', 'TREENODES']:
        add_tags_urls = [
            remote_instance._treenode_add_tag_url(n) for n in node_list]
    elif node_type in ['CONNECTOR','CONNECTORS']:
        add_tags_urls = [
            remote_instance._connector_add_tag_url(n) for n in node_list]    
    else:
        raise TypeError('Unknown node_type parameter: %s' % str( node_type) )

    if isinstance( tags, dict ):
        post_data = [
            {'tags': ','.join(tags[n]), 'delete_existing': override_existing} for n in node_list]
    else:        
        post_data = [
            {'tags': ','.join(tags), 'delete_existing': override_existing} for n in node_list]


    d = _get_urls_threaded(add_tags_urls, remote_instance,
                           post_data=post_data, desc='tags')

    return d


def get_review_details(x, remote_instance=None):
    """ Wrapper to retrieve review status (reviewer + timestamp) for each node 
    of a given skeleton -> uses the review API

    Parameters
    -----------
    x             
                        Neurons to get review-details for. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'

    Returns
    -------
    list
        list of reviewed nodes 

        >>> print(l)
        [   node_id, 
          [ 
          [ reviewer1, timestamp],
          [ reviewer2, timestamp] 
          ] 
        ]

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    node_list = []
    urls = []
    post_data = []

    for s in x:
        urls.append(remote_instance._get_review_details_url(s))
        # For some reason this needs to fetched as POST (even though actual
        # POST data is not necessary)
        post_data.append({'placeholder': 0})

    rdata = _get_urls_threaded(
        urls, remote_instance, post_data=post_data, desc='review')

    for neuron in rdata:
        # There is a small chance that nodes are counted twice but not tracking node_id speeds up this extraction a LOT
        #node_ids = []
        for arbor in neuron:
            node_list += [(n['id'], n['rids'])
                          for n in arbor['sequence'] if n['rids']]

    return node_list


def get_logs(remote_instance=None, operations=[], entries=50, display_start=0, search=''):
    """ Wrapper to retrieve logs (same data as in log widget)

    Parameters
    ----------    
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define  
                        globally as 'remote_instance'
    operations :        list of str, optional
                        If empty, all operations will be queried from server
                        possible operations: 'join_skeleton', 
                        'change_confidence', 'rename_neuron', 'create_neuron', 
                        'create_skeleton', 'remove_neuron', 'split_skeleton', 
                        'reroot_skeleton', 'reset_reviews', 'move_skeleton'
    entries :           int, optional
                        Number of entries to retrieve
    display_start :     int, optional
                        Sets range of entries: display_start -> display_start + entries
    search :            str, optional
                        Use to filter results for e.g. a specific skeleton ID
                        or neuron name

    Returns
    -------
    pandas.DataFrame   
        DataFrame in which each row represents a single operation

        >>> df
            user   operation   timestamp   x   y   z   explanation 
        0
        1

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if not operations:
        operations = [-1]

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
                               'skids', 'y', 'z', 'explanation']
                      )

    return df


def get_contributor_statistics(x, remote_instance=None, separate=False):
    """ Wrapper to retrieve contributor statistics for given skeleton ids.
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
                        Either pass directly to function or define  
                        globally as 'remote_instance'
    separate :          bool, optional
                        If true, stats are given per neuron

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron

        >>> df
            skeleton_id node_contributors multiuser_review_minutes  ..      
        1
        2
        3

           post_contributors construction_minutes  min_review_minutes  .. 
        1
        2
        3

           n_postsynapses  n_presynapses pre_contributors  n_nodes
        1
        2
        3

    """
    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x = eval_skids(x, remote_instance=remote_instance)

    if type(x) != type(list()):
        x = [x]

    columns = ['skeleton_id', 'n_nodes', 'node_contributors', 'n_presynapses',  'pre_contributors',
               'n_postsynapses', 'post_contributors', 'multiuser_review_minutes', 'construction_minutes', 'min_review_minutes']

    if not separate:
        get_statistics_postdata = {}

        for i in range(len(x)):
            key = 'skids[%i]' % i
            get_statistics_postdata[key] = x[i]

        remote_get_statistics_url = remote_instance._get_contributions_url()
        stats = remote_instance.fetch(
            remote_get_statistics_url, get_statistics_postdata)

        df = pd.DataFrame([[
            x,
            stats['n_nodes'],
            stats['node_contributors'],
            stats['n_pre'],
            stats['pre_contributors'],
            stats['n_post'],
            stats['post_contributors'],
            stats['multiuser_review_minutes'],
            stats['construction_minutes'],
            stats['min_review_minutes']
        ]],
            columns=columns,
            dtype=object
        )
    else:
        get_statistics_postdata = [{'skids[0]': s} for s in x]
        remote_get_statistics_url = [
            remote_instance._get_contributions_url() for s in x]

        stats = _get_urls_threaded(
            remote_get_statistics_url, remote_instance, post_data=get_statistics_postdata, desc='contributions')

        df = pd.DataFrame([[
            s,
            stats[i]['n_nodes'],
            stats[i]['node_contributors'],
            stats[i]['n_pre'],
            stats[i]['pre_contributors'],
            stats[i]['n_post'],
            stats[i]['post_contributors'],
            stats[i]['multiuser_review_minutes'],
            stats[i]['construction_minutes'],
            stats[i]['min_review_minutes']
        ] for i, s in enumerate(x)],
            columns=columns,
            dtype=object
        )
    return df


def get_neuron_list(remote_instance=None, user=None, node_count=1, start_date=[], end_date=[], reviewed_by=None):
    """ Wrapper to retrieves a list of all skeletons that fit given parameters 
    (see variables). If no parameters are provided, all existing skeletons are 
    returned!

    Parameters
    ----------
    remote_instance :   CatmaidInstance, optional                        
                        Either pass directly to function or define  
                        globally as 'remote_instance'.
    user :              int, optional
                        A single user_id.
    node_count :        int, optional
                        Minimum number of nodes.
    start_date :        list of integers, optional 
                        [year, month, day]
                        Only consider neurons created after.
    end_date :          list of integers , optional
                        [year, month, day]
                        Only consider neurons created before.

    Returns
    -------
    list              
                        ``[ skid, skid, skid, ... ]``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    get_skeleton_list_GET_data = {'nodecount_gt': node_count}

    if user:
        get_skeleton_list_GET_data['created_by'] = user

    if reviewed_by:
        get_skeleton_list_GET_data['reviewed_by'] = reviewed_by

    if start_date and end_date:
        get_skeleton_list_GET_data['from'] = ''.join(
            [str(d) for d in start_date])
        get_skeleton_list_GET_data['to'] = ''.join([str(d) for d in end_date])

    remote_get_list_url = remote_instance._get_list_skeletons_url()
    remote_get_list_url += '?%s' % urllib.parse.urlencode(
        get_skeleton_list_GET_data)
    skid_list = remote_instance.fetch(remote_get_list_url)

    return skid_list


def get_history(remote_instance=None, start_date=(datetime.date.today() - datetime.timedelta(days=7)).isoformat(), end_date=datetime.date.today().isoformat(), split=True):
    """ Wrapper to retrieves CATMAID project history

    Notes
    -----
    If the time window is too large, the connection might time out which will 
    result in an error! Make sure ``split = True`` to avoid that.

    Parameters
    ----------
    remote_instance :   CATMAID instance, optional  
                        Either pass directly to function or define globally as 
                        'remote_instance'                             
    start_date :        {datetime, str, tuple}, optional, default=today
                        dates can be either::
                            - datetime.date
                            - datetime.datetime
                            - str 'YYYY-MM-DD' format, e.g. '2016-03-09'
                            - tuple ( YYYY, MM, DD ), e.g. (2016,3,9)    
    end_date :          {datetime, str, tuple}, optional, default=last week
                        See start_date
    split :             bool, optional
                        If True, history will be requested in bouts of 6 months
                        Useful if you want to look at a very big time window 
                        as this can lead to gateway timeout

    Returns
    -------
    pandas.Series
            A pandas.Series with the following entries:

            {
            cable :             DataFrame containing cable created in nm. 
                                Rows = users, columns = dates
            connector_links :   DataFrame containing connector links created. 
                                Rows = users, columns = dates   
            reviewed :          DataFrame containing nodes reviewed. 
                                Rows = users, columns = dates
            user_details :      user-list (see pymaid.get_user_list())
            }

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pymaid import pymaid
    >>> rm = pymaid.CatmaidInstance(    'server_url', 
    ...                                 'http_user',
    ...                                 'http_pw',
    ...                                 'token')
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
            except:
                temp.append(0)
        return temp

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if isinstance(start_date, datetime.date):
        start_date = start_date.isoformat()
    elif isinstance(start_date, datetime.datetime):
        start_date = start_date.isoformat()[:10]
    elif isinstance(start_date, tuple):
        start_date = datetime.date(start_date[0], start_date[
                                   1], start_date[2]).isoformat()

    if isinstance(end_date, datetime.date):
        end_date = end_date.isoformat()
    elif isinstance(end_date, datetime.datetime):
        end_date = end_date.isoformat()[:10]
    elif isinstance(end_date, tuple):
        end_date = datetime.date(end_date[0], end_date[
                                 1], end_date[2]).isoformat()

    rounds = []
    if split:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        remote_instance.logger.info(
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
    for r in tqdm(rounds, desc='Retrieving history'):
        get_history_GET_data = {'self.project_id': remote_instance.project_id,
                                'start_date': r[0],
                                'end_date': r[1]
                                }

        remote_get_history_url = remote_instance._get_history_url()

        remote_get_history_url += '?%s' % urllib.parse.urlencode(
            get_history_GET_data)

        remote_instance.logger.debug(
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
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_treenodes', stats['days']) for u in stats['stats_table']],
                     index=[user_list.ix[u].login for u in stats[
                         'stats_table'].keys()],
                     columns=[datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']]),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_connectors', stats['days']) for u in stats['stats_table']],
                     index=[user_list.ix[u].login for u in stats[
                         'stats_table'].keys()],
                     columns=[datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']]),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_reviewed_nodes', stats['days']) for u in stats['stats_table']],
                     index=[user_list.ix[u].login for u in stats[
                         'stats_table'].keys()],
                     columns=[datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']]),
        user_list.reset_index(drop=True)
    ],
        index=['cable', 'connector_links', 'reviewed', 'user_details']
    )
    return df


def get_nodes_in_volume(left, right, top, bottom, z1, z2, remote_instance=None, coord_format='NM', resolution=(4, 4, 50)):
    """ Get nodes in provided volume. This is the same API enpoint that is 
    called when panning in the browser.

    Parameters
    ----------
    left :                  {int,float}
    right :                 {int,float}
    top :                   {int,float}
    bottom :                {int,float}
    z1 :                    {int,float}
    z2 :                    {int,float}    
                            Coordinates defining the volume 
                            Can be given in nm or pixels+slices.  
    remote_instance :       CATMAID instance, optional
                            Either pass directly to function or define 
                            globally as 'remote_instance'
    coord_format :          str, optional
                            Define whether provided coordinates are in 
                            nanometer ('NM') or in pixels/slices ('PIXEL')
    resolution :            tuple of floats, optional
                            x/y/z resolution in nm (default = ( 4, 4, 50 ) )
                            used to transform to nm if limits are given in 
                            pixels    

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

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

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

    data = {'treenodes': pd.DataFrame([[i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], datetime.datetime.fromtimestamp(int(i[8])), i[9], ]
                                       for i in node_list[0]],
                                      columns=['treenode_id', 'parent_id', 'skids', 'y', 'z', 'confidence',
                                               'radius', 'skeleton_id', 'edition_time', 'user_id'],
                                      dtype=object),
            'connectors':  pd.DataFrame([[i[0], i[1], i[2], i[3], i[4], datetime.datetime.fromtimestamp(int(i[5])), i[6], i[7]]
                                         for i in node_list[1]],
                                        columns=[
                                            'connector_id', 'skids', 'y', 'z', 'confidence', 'edition_time', 'user_id', 'partners'],
                                        dtype=object),
            'labels':      node_list[3],
            'node_limit_reached': node_list[4],
            }

    return data


def get_neurons_in_volume(volumes, remote_instance=None, intersect=False, min_size=1, only_soma=False):
    """ Retrieves neurons with processes within CATMAID volumes. This function
    uses the *BOUNDING BOX* around the volume as proxy and queries for neurons
    that are within that volume.

    Warning  
    -------
    Depending on the number of nodes in that volume, this can take quite a 
    while!

    Parameters
    ----------
    volumes :               list of str
                            Single or list of CATMAID volume names.
    remote_instance :       CATMAID instance 
                            Either pass directly to function or define 
                            globally as 'remote_instance'
    intersect :             bool, optional
                            if multiple volumes are provided, this parameter
                            determines if neurons have to be in all of the
                            neuropils or just a single 
    min_size :              int, optional
                            minimum size (in nodes) for neurons to be returned
    only_soma :             bool, optional
                            if True, only neurons with a soma will be returned

    Returns
    --------
    list                  
                            ``[ skeleton_id, skeleton_id, ... ]``

    See Also
    --------
    :func:`pymaid.pymaid.get_partners_in_volume`
                            Get only partners that make connections within a
                            given volume
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if type(volumes) != type(list()):
        volumes = [volumes]

    neurons = []

    for v in volumes:
        volume = get_volume(v, remote_instance)
        verts = volume['vertices']

        bbox = ((int(min([v[0] for v in verts])), int(max([v[0] for v in verts]))),
                (int(min([v[1] for v in verts])),
                 int(max([v[1] for v in verts]))),
                (int(min([v[2] for v in verts])),
                 int(max([v[2] for v in verts])))
                )

        remote_instance.logger.info('Retrieving nodes in volume: %s...' % v)
        temp = get_neurons_in_box(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], bbox[
                                  2][0], bbox[2][1], remote_instance=remote_instance)

        if not intersect:
            neurons += temp
        else:
            neurons += [temp]

    if intersect:
        # Filter for neurons that show up in all neuropils
        neurons = [n for l in neurons for n in l if False not in [
            n in v for v in neurons]]

    if min_size > 1:
        remote_instance.logger.info('Filtering neurons for size...')
        rev = get_review(list(set(neurons)), remote_instance)
        neurons = rev[rev.total_node_count > min_size].skeleton_id.tolist()

    if only_soma:
        remote_instance.logger.info('Filtering neurons for somas...')
        soma = has_soma(list(set(neurons)), remote_instance)
        neurons = [n for n in neurons if soma[n] is True]

    remote_instance.logger.info('Done. %i unique neurons found in volume(s): %s' % (
        len(set(neurons)), (',').join(volumes)))

    return list(set(neurons))


def get_neurons_in_box(left, right, top, bottom, z1, z2, remote_instance=None, unit='NM',  **kwargs):
    """ Retrieves neurons with processes within a defined volume. Because the 
    API returns only a limited number of neurons at a time, the defined volume 
    has to be chopped into smaller pieces for crowded areas - may thus take 
    some time! Unlike pymaid.get_neurons_in_volume(), this function will 
    retrieve ALL neurons within the box - not just the once entering/exiting.

    Parameters
    ----------
    left :                  {int,float}
    right :                 {int,float}
    top :                   {int,float}
    bottom :                {int,float}
    z1 :                    {int,float}
    z2 :                    {int,float}
    unit :                  {'NM','PIXEL'}
                            Unit of your coordinates. Attention:
                            'PIXEL' will also assume that Z1/Z2 is in slices.                                 
                            By default, a X/Y resolution of 3.8nm and a Z
                            resolution of 35nm is assumed. Pass 'xy_res' and
                            'z_res' as **kwargs to override this.

    remote_instance :       CATMAID instance 
                            Either pass directly to function or define 
                            globally as 'remote_instance'

    Returns
    --------
    list
                            ``[ skeleton_id, skeleton_id, ... ]``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    x_y_resolution = kwargs.get('xy_res', 3.8)
    z_resolution = kwargs.get('z_res', 35)

    if unit == 'PIXEL':
        left *= x_y_resolution
        right *= x_y_resolution
        top *= x_y_resolution
        bottom *= x_y_resolution
        z1 *= z_resolution
        z2 *= z_resolution

    def get_nodes(left, right, top, bottom, z1, z2, remote_instance, incursion):

        remote_instance.logger.debug('%i: Left %i, Right %i, Top %i, Bot %i, Z1 %i, Z2 %i' % (
            incursion, left, right, top, bottom, z1, z2))
        remote_instance.logger.debug('Incursion %i' % incursion)

        remote_nodes_list = remote_instance._get_node_list_url()

        node_list_postdata = {
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
            'z1': z1,
            'z2': z2,
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
        }

        node_list = remote_instance.fetch(
            remote_nodes_list, node_list_postdata)

        if node_list[3] is True:
            remote_instance.logger.debug('Incursing.')
            incursion += 8
            node_list = list()
            # Front left top
            temp, incursion = get_nodes(left,
                                        left + (right - left) / 2,
                                        top,
                                        top + (bottom - top) / 2,
                                        z1,
                                        z1 + (z2 - z1) / 2,
                                        remote_instance, incursion)
            node_list += temp

            # Front right top
            temp, incursion = get_nodes(left + (right - left) / 2,
                                        right,
                                        top,
                                        top + (bottom - top) / 2,
                                        z1,
                                        z1 + (z2 - z1) / 2,
                                        remote_instance, incursion)
            node_list += temp

            # Front left bottom
            temp, incursion = get_nodes(left,
                                        left + (right - left) / 2,
                                        top + (bottom - top) / 2,
                                        bottom,
                                        z1,
                                        z1 + (z2 - z1) / 2,
                                        remote_instance, incursion)
            node_list += temp

            # Front right bottom
            temp, incursion = get_nodes(left + (right - left) / 2,
                                        right,
                                        top + (bottom - top) / 2,
                                        bottom,
                                        z1,
                                        z1 + (z2 - z1) / 2,
                                        remote_instance, incursion)
            node_list += temp

            # Back left top
            temp, incursion = get_nodes(left,
                                        left + (right - left) / 2,
                                        top,
                                        top + (bottom - top) / 2,
                                        z1 + (z2 - z1) / 2,
                                        z2,
                                        remote_instance, incursion)
            node_list += temp

            # Back right top
            temp, incursion = get_nodes(left + (right - left) / 2,
                                        right,
                                        top,
                                        top + (bottom - top) / 2,
                                        z1 + (z2 - z1) / 2,
                                        z2,
                                        remote_instance, incursion)
            node_list += temp

            # Back left bottom
            temp, incursion = get_nodes(left,
                                        left + (right - left) / 2,
                                        top + (bottom - top) / 2,
                                        bottom,
                                        z1 + (z2 - z1) / 2,
                                        z2,
                                        remote_instance, incursion)
            node_list += temp

            # Back right bottom
            temp, incursion = get_nodes(left + (right - left) / 2,
                                        right,
                                        top + (bottom - top) / 2,
                                        bottom,
                                        z1 + (z2 - z1) / 2,
                                        z2,
                                        remote_instance, incursion)
            node_list += temp

        else:
            # If limit not reached, node list is still an array of 4
            return node_list[0], incursion - 1

        remote_instance.logger.info(
            "%i Incursion complete (%i nodes received)" % (incursion, len(node_list)))

        return node_list, incursion

    incursion = 1
    node_list, incursion = get_nodes(
        left, right, top, bottom, z1, z2, remote_instance, incursion)

    # Collapse list into unique skeleton ids
    skeletons = set()
    for node in node_list:
        skeletons.add(node[7])

    remote_instance.logger.info("Done: %i nodes from %i unique neurons retrieved." % (
        len(node_list), len(skeletons)))

    return list(skeletons)


def get_user_list(remote_instance=None):
    """ Get list of users for given CATMAID server (not project specific)

    Parameters
    ----------    
    remote_instance :   CATMAID instance 
                        Either pass directly to function or define globally 
                        as ``remote_instance``

    Returns
    ------
    pandas.DataFrame        
        DataFrame in which each row represents a user

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

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    user_list = remote_instance.fetch(remote_instance._get_user_list_url())

    columns = ['id', 'login', 'full_name', 'first_name', 'last_name', 'color']

    df = pd.DataFrame([[e[c] for c in columns] for e in user_list],
                      columns=columns
                      )

    df.sort_values(['login'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def get_paths(sources, targets, remote_instance=None, n_hops=2, min_synapses=2):
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
                        Either pass directly to function or define  
                        globally as 'remote_instance'.
    n_hops :            int, optional
                        Number of hops allowed between sources and targets.
    min_synapses :      int, optional
                        Minimum number of synpases between source and target.

    Returns
    -------
    igraph.Graph
                iGraph object containing the neurons that connect 
                sources and targets. Does only contain edges that 
                connect sources and targets!

    paths :     list
                List of skeleton IDs that constitute paths from
                sources to targets::

                    [ [ source1, skid1, target1 ], [source2, skid2, target2 ], ...  ]

    Attention
    ---------
    The returned iGraph graph does **only** contain the edges that connnect
    sources and targets. Other edges have been removed.

    Examples
    --------
    >>> # This assumes that you have already set up a Catmaid Instance    
    >>> from igraph import plot
    >>> g, paths = pymaid.get_paths( ['annotation:glomerulus DA1'],
    ...                              ['2333007'] )
    >>> g
    <igraph.Graph object at 0x11c857138>
    >>> paths
    [['57381', '4376732', '2333007'], ['57323', '630823', '2333007'], ...
    >>> plot( g, layout = g.layout("kk"), 
    ...       **{ 'edge_label' : g.es['weight'] } )

    """
    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    sources = eval_skids(sources, remote_instance=remote_instance)
    targets = eval_skids(targets, remote_instance=remote_instance)

    if not isinstance(targets, (list, np.ndarray)):
        targets = [targets]

    if not isinstance(sources, (list, np.ndarray)):
        sources = [sources]

    url = remote_instance._get_graph_dps_url()
    post_data = {
        'n_hops': n_hops,
        'min_synapses': min_synapses
    }

    for i, s in enumerate(sources):
        post_data['sources[%i]' % i] = s

    for i, t in enumerate(targets):
        post_data['targets[%i]' % i] = t

    # Response is just a set of skeleton IDs
    response = remote_instance.fetch(url, post=post_data)

    # Turn neurons into an iGraph graph
    g = igraph_catmaid.network2graph(
        response, remote_instance=remote_instance, threshold=min_synapses)

    # Get all paths between sources and targets
    all_paths = igraph_catmaid._find_all_paths(g,
                                               [i for i, v in enumerate(g.vs) if v[
                                                   'node_id'] in sources],
                                               [i for i, v in enumerate(g.vs) if v[
                                                   'node_id'] in targets],
                                               maxlen=n_hops
                                               )

    # Delete edges that don't go from sources to targets from graph
    edges_to_keep = set()
    for p in all_paths:
        for i in range(len(p) - 1):
            edges_to_keep.add((p[i], p[i + 1]))

    g.delete_edges([(e.source, e.target)
                    for e in g.es if (e.source, e.target) not in edges_to_keep])

    return g, [[g.vs[i]['node_id'] for i in p] for p in all_paths]


def get_volume(volume_name, remote_instance=None):
    """ Retrieves volume (mesh) from Catmaid server and converts to set of 
    vertices and faces.

    Parameters
    ----------
    volume_name :       str
                        Name of the volume to import - must be EXACT!
    remote_instance :   CATMAID instance, optional
                        Either pass directly to function or define  
                        globally as ``remote_instance``

    Returns
    -------
    dict 
        Dictionary containing vertices and faces::

            vertices :      list of tuples
                            [ (x,y,z), (x,y,z), .... ]
            faces :         list of tuples
                            [ ( vertex_ix, vertex_ix, vertex_ix ), ... ]


    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if not isinstance(volume_name, str):
        raise TypeError('Volume name must be str')

    remote_instance.logger.info('Retrieving volume <%s>' % volume_name)

    # First, get volume ID
    get_volumes_url = remote_instance._get_volumes()
    response = remote_instance.fetch(get_volumes_url)

    volume_id = [e['id'] for e in response if e['name'] == volume_name]

    if not volume_id:
        remote_instance.logger.error(
            'Did not find a matching volume for name %s' % volume_name)
        raise Exception(
            'Did not find a matching volume for name %s' % volume_name)
    else:
        volume_id = volume_id[0]

    # Now download volume
    url = remote_instance._get_volume_details(volume_id)
    response = remote_instance.fetch(url)

    mesh_str = response['mesh']
    mesh_name = response['name']

    mesh_type = re.search('<(.*?) ', mesh_str).group(1)

    # Now reverse engineer the mesh
    if mesh_type == 'IndexedTriangleSet':
        t = re.search("index='(.*?)'", mesh_str).group(1).split(' ')
        faces = [(int(t[i]), int(t[i + 1]), int(t[i + 2]))
                 for i in range(0, len(t) - 2, 3)]

        v = re.search("point='(.*?)'", mesh_str).group(1).split(' ')
        vertices = [(float(v[i]), float(v[i + 1]), float(v[i + 2]))
                    for i in range(0,  len(v) - 2, 3)]

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
                    for i in range(0,  len(v) - 2, 3)]

    else:
        remote_instance.logger.error("Unknown volume type: %s" % mesh_type)
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

    remote_instance.logger.debug('Volume type: %s' % mesh_type)
    remote_instance.logger.debug(
        '# of vertices after clean-up: %i' % len(final_vertices))
    remote_instance.logger.debug(
        '# of faces after clean-up: %i' % len(final_faces))

    return dict(vertices=final_vertices, faces=final_faces)


def eval_skids(x, remote_instance=None):
    """ Wrapper to evaluate parameters passed as skeleton IDs. Will turn
    annotations and neuron names into skeleton IDs.

    Parameters
    ----------
    x :             {int, str, CatmaidNeuron, CatmaidNeuronList, DataFrame}
                    Your options are either::
                    1. int or list of ints will be assumed to be skeleton IDs
                    2. str or list of str:
                        - if convertible to int, will be interpreted as x
                        - elif start with 'annotation:' will be assumed to be 
                          annotations
                        - else, will be assumed to be neuron names
                    3. For CatmaidNeuron/List or pandas.DataFrames will try
                       to extract skeleton_id parameter
    remote_instance : CatmaidInstance, optional

    Returns
    -------
    list of str
                    list containing skeleton IDs as strings
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if isinstance(x, (int, np.int64, np.int32, np.int)):
        return str(x)
    elif isinstance(x, (str, np.str)):
        try:
            int(x)
            return str(x)
        except:
            if x.startswith('annotation:'):
                return get_skids_by_annotation(x[11:], remote_instance=remote_instance)
            elif x.startswith('name:'):
                return get_skids_by_name(x[5:], remote_instance=remote_instance, allow_partial=False).skeleton_id.tolist()
            else:
                return get_skids_by_name(x, remote_instance=remote_instance, allow_partial=False).skeleton_id.tolist()
    elif isinstance(x, (list, np.ndarray)):
        skids = []
        for e in x:
            temp = eval_skids(e, remote_instance=remote_instance)
            if isinstance(temp, (list, np.ndarray)):
                skids += temp
            else:
                skids.append(temp)
        return list(set(skids))
    elif isinstance(x, core.CatmaidNeuron):
        return [x.skeleton_id]
    elif isinstance(x, core.CatmaidNeuronList):
        return list(x.skeleton_id)
    elif isinstance(x, pd.DataFrame):
        return x.skeleton_id.tolist()
    elif isinstance(x, pd.Series):
        return [x.skeleton_id]
    else:
        remote_instance.logger.error(
            'Unable to extract x from type %s' % str(type(x)))
        raise TypeError('Unable to extract x from type %s' % str(type(x)))
