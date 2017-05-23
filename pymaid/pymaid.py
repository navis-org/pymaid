""" 
A collection of tools to remotely access a CATMAID server via its API
    
    Copyright (C) 2017 Philipp Schlegel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along

Basic example:
------------
from pymaid.pymaid import CatmaidInstance, get_3D_skeleton

#HTTP_USER AND HTTP_PASSWORD are only necessary if your server requires a 
#http authentification
myInstance = CatmaidInstance(   'www.your.catmaid-server.org' , 
                                'HTTP_USER' , 
                                'HTTP_PASSWORD', 
                                'TOKEN' )

skeleton_data = get_3D_skeleton ( ['12345','67890'] , myInstance )

Additional examples:
-------------------
Also see https://github.com/schlegelp/PyMaid for Jupyter notebooks

Contents:
-------
CatmaidInstance :   Class holding credentials and server url 
                    Use to generate urls and fetch data

A long list of wrappers to retrieve various data from the CATMAID server.
Use dir(pymaid.pymaid) to get a list and help(pymaid.pymaid.wrapper) to learn 
more about it.

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

class CatmaidInstance:
    """ A class giving access to a CATMAID instance.

    Attributes
    ----------
    server :        string
                    The url for a CATMAID server.
    authname :      string
                    The http user. 
    authpassword :  string
                    The http password.
    authtoken :     string
                    User token - see CATMAID documentation on how to retrieve it.
    logger :        optional
                    provide name for logging.getLogger if you like the CATMAID 
                    instance to log to a specific logger by default (None), a 
                    dedicated logger __name__ is created
    debug :         boolean (optional)
                    if True, logging level is set to 'DEBUG' (default is 'INFO')

    Methods
    -------
    fetch ( url, post = None )
        retrieves information from CATMAID server
    get_XXX_url ( pid , sid , X )
        <CatmaidInstance> class contains a long list of function that generate URLs 
        to request data from the CATMAID server. Use dir(<CatmaidInstance>) to get 
        the full list. Most of these functions require a project id (pid) and a stack 
        id (sid) some require additional parameters, for example a skeleton id (skid). 

    Examples:
    --------
    # 1.) Fetch skeleton data for a single neuron

    from pymaid.pymaid import CatmaidInstance

    myInstance = CatmaidInstance(   'www.your.catmaid-server.org', 
                                    'user', 
                                    'password', 
                                    'token' 
                                )
    project_id = 1
    skeleton_id = 12345
    3d_skeleton_url = myInstance.get_compact_skeleton_url(  project_id , 
                                                            skeleton_id )
    skeleton_data = myInstance.fetch( 3d_skeleton_url )

    # 2.) Use wrapper functions to fetch 3D skeletons for multiple neurons

    from pymaid.pymaid import CatmaidInstance, get_3D_skeleton

    myInstance = CatmaidInstance(   'www.your.catmaid-server.org', 
                                    'user', 
                                    'password', 
                                    'token' )
    skeleton_data = get_3D_skeleton ( ['12345','67890'] , myInstance )
    
    """

    def __init__(self, server, authname, authpassword, authtoken, logger = None, debug=False ):
        self.server = server
        self.authname = authname
        self.authpassword = authpassword
        self.authtoken = authtoken
        self.opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())            

        #If pymaid is not run as module, make sure logger has a at least a StreamHandler
        if not logger:
            self.logger = logging.getLogger(__name__)             
        else:
            self.logger = logging.getLogger(logger)

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            #Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)

            self.logger.addHandler(sh)

            if not debug:
                self.logger.setLevel(logging.INFO)
            else:
                self.logger.setLevel(logging.DEBUG)

        self.logger.info('CATMAID instance created')

    def djangourl(self, path):
        """ Expects the path to lead with a slash '/'. """
        return self.server + path

    def auth(self, request):
        if self.authname:
            base64string = base64.encodestring(('%s:%s' % (self.authname, self.authpassword)).encode()).decode().replace('\n', '')
            request.add_header("Authorization", "Basic %s" % base64string)
        if self.authtoken:
            request.add_header("X-Authorization", "Token {}".format(self.authtoken))

    def fetch(self, url, post=None):
        """ Requires the url to connect to and the variables for POST, if any, in a dictionary. """
        if post:
            data = urllib.parse.urlencode(post)
            data = data.encode('utf-8')                        
            request = urllib.request.Request(url, data = data ) #headers = {'Content-Type': 'application/json', 'Accept':'application/json'}         
        else:
            request = urllib.request.Request(url)

        self.auth(request)        

        response = self.opener.open(request)

        return json.loads(response.read().decode("utf-8"))
    
    def get_stack_info_url(self, pid, sid):
        """ Use to parse url for retrieving stack infos. """
        return self.djangourl("/" + str(pid) + "/stack/" + str(sid) + "/info")
    
    def get_skeleton_nodes_url(self, pid, skid):
        """ Use to parse url for retrieving skeleton nodes (no info on parents or synapses, does need post data). """
        return self.djangourl("/" + str(pid) + "/treenode/table/" + str(skid) + "/content")
    
    def get_skeleton_for_3d_viewer_url(self, pid, skid):
        """ ATTENTION: this url doesn't work properly anymore as of 07/07/14
        use compact-skeleton instead
        Used to parse url for retrieving all info the 3D viewer gets (does NOT need post data)
        Format: name, nodes, tags, connectors, reviews
        """
        return self.djangourl("/" + str(pid) + "/skeleton/" + str(skid) + "/compact-json")
    
    def get_add_annotations_url(self,pid):
        """ Use to parse url to add annotations to skeleton IDs. """
        return self.djangourl("/" + str(pid) + "/annotations/add" )    
    
    def get_connectivity_url(self, pid):
        """ Use to parse url for retrieving connectivity (does need post data). """
        return self.djangourl("/" + str(pid) + "/skeletons/connectivity" )

    def get_connectors_url(self, pid):
        """ Use to retrieve list of connectors either pre- or postsynaptic a set of neurons - GET request
        Format: { 'links': [ skeleton_id, connector_id, x,y,z, S(?), confidence, creator, treenode_id, creation_date ] }
        """    
        return self.djangourl("/" + str(pid) + "/connectors/" )
    
    def get_connector_details_url(self, pid):
        """ Use to parse url for retrieving info connectors (does need post data). """
        return self.djangourl("/" + str(pid) + "/connector/skeletons" )
    
    def get_neuronnames(self, pid):
        """ Use to parse url for names for a list of skeleton ids (does need post data: pid, skid). """
        return self.djangourl("/" + str(pid) + "/skeleton/neuronnames" )
    
    def get_list_skeletons_url(self, pid):
        """ Use to parse url for names for a list of skeleton ids (does need post data: pid, skid). """
        return self.djangourl("/" + str(pid) + "/skeletons/")

    def get_completed_connector_links(self, pid):
        """ Use to parse url for retrieval of completed connector links by given user 
        GET request: 
        Returns list: [ connector_id, [x,z,y], node1_id, skeleton1_id, link1_confidence, creator_id, [x,y,z], node2_id, skeleton2_id, link2_confidence, creator_id ]
        """
        return self.djangourl("/" + str(pid) + "/connector/list/")

    def url_to_coordinates(self, pid, coords , stack_id = 0, tool = 'tracingtool' , active_skeleton_id = None, active_node_id = None ):
        """ Use to generate URL to a location

        Parameters:
        ----------
        coords :    list of integers
                    (x, y, z)

        """
        GET_data = {   'pid': pid,
                        'xp': coords[0],
                        'yp': coords[1],
                        'zp': coords[2],
                        'tool':tool,
                        'sid0':stack_id,
                        's0': 0
                    }

        if active_skeleton_id:
            GET_data['active_skeleton_id'] = active_skeleton_id
        if active_node_id:
            GET_data['active_node_id'] = active_node_id

        return( self.djangourl( '?%s' % urllib.parse.urlencode( GET_data )  ) )
    
    def get_user_list_url(self):
        """ Get user list for project. """
        return self.djangourl("/user-list" )
    
    def get_single_neuronname_url(self, pid, skid):
        """ Use to parse url for a SINGLE neuron (will also give you neuronID). """
        return self.djangourl("/" + str(pid) + "/skeleton/" + str(skid) + "/neuronname" )    
    
    def get_review_status(self, pid):
        """ Use to get skeletons review status. """
        return self.djangourl("/" + str(pid) + "/skeletons/review-status" )
    
    def get_neuron_annotations(self, pid):
        """ Use to get annotations for given neuron. DOES need skid as postdata. """
        return self.djangourl("/" + str(pid) + "/annotations/table-list" )
    
    def get_intersects(self, pid, vol_id, x, y, z):        
        """ Use to test if point intersects with volume. """
        return self.djangourl("/" + str(pid) + "/volumes/"+str(vol_id)+"/intersect" ) + '?%s' % urllib.parse.urlencode( {'x':x, 'y':y , 'z': z} )
    
    def get_volumes(self, pid):
        """ Get list of all volumes in project. """
        return self.djangourl("/" + str(pid) + "/volumes/")    

    #Get details on a given volume (mesh)
    def get_volume_details(self, pid, volume_id):
        return self.djangourl("/" + str(pid) + "/volumes/" + str(volume_id) )

    def get_annotations_for_skid_list(self, pid):
        """ ATTENTION: This does not seem to work anymore as of 20/10/2015 -> although it still exists in CATMAID code
            use get_annotations_for_skid_list2    
            Use to get annotations for given neuron. DOES need skid as postdata
        """
        return self.djangourl("/" + str(pid) + "/annotations/skeletons/list" )      

    def get_review_details_url(self, pid, skid):
        """ Use to retrieve review status for every single node of a skeleton.      
        For some reason this needs to fetched as POST (even though actual POST data is not necessary)
        Returns list of arbors, the nodes the contain and who has been reviewing them at what time  
        """ 
        return self.djangourl("/" + str(pid) + "/skeletons/" + str(skid) + "/review" )
    
    def get_annotations_for_skid_list2(self, pid):
        """ Use to get annotations for given neuron. DOES need skid as postdata. """
        return self.djangourl("/" + str(pid) + "/skeleton/annotationlist" )

    def get_logs_url(self, pid):
        """ Use to get logs. DOES need skid as postdata. """
        return self.djangourl("/" + str(pid) + "/logs/list" )
    
    def get_annotation_list(self, pid):        
        """ Use to parse url for retrieving list of all annotations (and their IDs!!!). Filters can be passed as POST optionally:
        
        Filter parameters that can be passed as POST (from CATMAID Github 20/10/2015):
      - name: annotations
        description: A list of (meta) annotations with which which resulting annotations should be annotated with.        
        type: array
        items:
            type: integer
            description: An annotation ID
      - name: annotates
        description: A list of entity IDs (like annotations and neurons) that should be annotated by the result set.        
        type: array
        items:
            type: integer
            description: An entity ID
      - name: parallel_annotations
        description: A list of annotation that have to be used alongside the result set.        
        type: array
        items:
            type: integer
            description: An annotation ID
      - name: user_id
        description: Result annotations have to be used by this user.        
        type: integer
      - name: neuron_id
        description: Result annotations will annotate this neuron.        
        type: integer
      - name: skeleton_id
        description: Result annotations will annotate the neuron modeled by this skeleton.        
        type: integer
      - name: ignored_annotations
        description: A list of annotation names that will be excluded from the result set.        
        type: array
        items:
            type: string
        """
        return self.djangourl("/" + str(pid) + "/annotations/" )
    
    def get_contributions_url(self, pid ):
        """ Use to parse url for retrieving contributor statistics for given skeleton (does need post data). """
        return self.djangourl("/" + str(pid) + "/skeleton/contributor_statistics_multiple" )     
    
    def get_annotated_url(self, pid):
        """ Use to parse url for retrieving annotated neurons (NEEDS post data). """        
        return self.djangourl("/" + str(pid) + "/annotations/query-targets" )

    def get_skid_from_tnid( self, pid, tnid):
        """ Use to parse url for retrieving the skeleton id to a single treenode id (does not need postdata) 
        API returns dict: {"count": integer, "skeleton_id": integer}
        """
        return self.djangourl( "/" + str(pid) + "/skeleton/node/" + str(tnid) + "/node_count" )
    
    def get_node_list_url(self, pid):
        """ Use to parse url for retrieving list of nodes (NEEDS post data). """
        return self.djangourl("/" + str(pid) + "/node/list" )  
    
    def get_node_info_url(self, pid):
        """ Use to parse url for retrieving user info on a single node (needs post data). """
        return self.djangourl("/" + str(pid) + "/node/user-info" )        

    def treenode_add_tag_url(self, pid, treenode_id):
        """ Use to parse url adding labels (tags) to a given treenode (needs post data)."""
        return self.djangourl("/" + str(pid) + "/label/treenode/" + str(treenode_id) + "/update" )

    def connector_add_tag_url(self, pid, treenode_id):
        """ Use to parse url adding labels (tags) to a given treenode (needs post data)."""
        return self.djangourl("/" + str(pid) + "/label/connector/" + str(treenode_id) + "/update" )
    
    def get_compact_skeleton_url(self, pid, skid, connector_flag = 1, tag_flag = 1):        
        """ Use to parse url for retrieving all info the 3D viewer gets (does NOT need post data).
        Returns, in JSON, [[nodes], [connectors], [tags]], with connectors and tags being empty when 0 == with_connectors and 0 == with_tags, respectively.
        Deprecated but kept for backwards compability!
        """
        return self.djangourl("/" + str(pid) + "/" + str(skid) + "/" + str(connector_flag) + "/" + str(tag_flag) + "/compact-skeleton")    

    def get_compact_details_url(self, pid, skid):        
        """ Similar to compact-skeleton but if 'with_history':True is passed as GET request, returned data will include all positions a nodes/connector has ever occupied plus the creation time and last modified.        
        """
        return self.djangourl("/" + str(pid) + "/skeletons/" + str(skid) + "/compact-detail")
    
    def get_compact_arbor_url(self, pid, skid, nodes_flag = 1, connector_flag = 1, tag_flag = 1):        
        """ The difference between this function and get_compact_skeleton is that the connectors contain the whole chain from the skeleton of interest to the
        partner skeleton: contains [treenode_id, confidence_to_connector, connector_id, confidence_from_connector, connected_treenode_id, connected_skeleton_id, relation1, relation2]
        relation1 = 1 means presynaptic (this neuron is upstream), 0 means postsynaptic (this neuron is downstream)
        """ 
        return self.djangourl("/" + str(pid) + "/" + str(skid) + "/" + str(nodes_flag) + "/" + str(connector_flag) + "/" + str(tag_flag) + "/compact-arbor")    
    
    def get_edges_url(self, pid):
        """ Use to parse url for retrieving edges between given skeleton ids (does need postdata).
        Returns list of edges: [source_skid, target_skid, weight]
        """
        return self.djangourl("/" + str(pid) + "/skeletons/confidence-compartment-subgraph" )
    
    def get_skeletons_from_neuron_id(self, neuron_id, pid):
        """ Use to get all skeletons of a given neuron (neuron_id). """
        return self.djangourl("/" + str(pid) + "/neuron/" + str(neuron_id) + '/get-all-skeletons' )
    
    def get_history_url(self, pid):
        """ Use to get user history. """
        return self.djangourl("/" + str(pid) + "/stats/user-history" )
  

def get_urls_threaded( urls , remote_instance, post_data = [], time_out = None ):
    """ Wrapper to retrieve a list of urls using threads

    Parameters:
    ----------
    urls :              list of strings
                        Urls to retrieve
    remote_instance :   CATMAID instance
                        Either pass directly to function or define globally 
                        as 'remote_instance'       
    post_data :         list of dicts
                        needs to be the same size as urls
    time_out :          integer or None
                        After this number of second, fetching data will time 
                        out (so as to not block the system)
                        If set to None, time out will be max( [ 20, len(urls) ] ) 
                        - e.g. 100s for 100 skeletons but at least 20s

    Returns:
    -------
    data :              data retrieved for each url -> order is kept!
    """

    data = [ None for u in urls ]
    threads = {}
    threads_closed = []   

    if time_out is None:
        time_out = max( [ len( urls ) , 20 ] ) 

    remote_instance.logger.debug('Creating %i threads to retrieve data' % len(urls) )
    for i, url in enumerate(urls):
        if post_data:
            t = retrieveUrlThreaded ( url, remote_instance, post_data = post_data[i] )
        else:
            t = retrieveUrlThreaded ( url, remote_instance )
        t.start()
        threads[ str(i) ] = t        
        remote_instance.logger.debug('Threads: %i' % len ( threads ) )  
    remote_instance.logger.debug('%i threads generated.' % len(threads) )

    remote_instance.logger.debug('Joining threads...') 

    start = cur_time = time.time()
    joined = 0
    while cur_time <= (start + time_out) and len( [ d for d in data if d != None ] ) != len(threads):
        for t in threads:
            if t in threads_closed:
                continue
            if not threads[t].is_alive():
                #Make sure we keep the order
                data[ int(t) ] = threads[t].join() 
                threads_closed.append(t)
        time.sleep(1)
        cur_time = time.time()

        remote_instance.logger.debug('Closing Threads: %i ( %is until time out )' % ( len( threads_closed ) , round( time_out - ( cur_time - start ) ) ) )             

    if cur_time > (start + time_out):        
        remote_instance.logger.warning('Timeout while joining threads. Retrieved only %i of %i urls' % ( len( [ d for d in data if d != None ] ), len(threads) ) )  
        for t in threads:
            if t not in threads_closed:
                remote_instance.logger.warning('Did not close thread for url: ' + urls[ int( t ) ] )
    else:
        remote_instance.logger.debug('Success! %i of %i urls retrieved.' % ( len(threads_closed) , len( urls ) ) )
    
    return data

class retrieveUrlThreaded(threading.Thread):
    """ Class to retrieve a URL by threading
    """
    def __init__(self,url,remote_instance,post_data=None):
        try:
            self.url = url
            self.post_data = post_data 
            threading.Thread.__init__(self)
            self.connector_flag = 1
            self.tag_flag = 1
            self.remote_instance = remote_instance
        except:
            remote_instance.logger.error('Failed to initiate thread for ' + self.url )

    def run(self):
        """
        Retrieve data from single url
        """          
        if self.post_data:
            self.data = self.remote_instance.fetch( self.url, self.post_data ) 
        else:
            self.data = self.remote_instance.fetch( self.url )         
        return 

    def join(self):
        try:
            threading.Thread.join(self)
            return self.data
        except:
            remote_instance.logger.error('Failed to join thread for ' + self.url )
            return None

def get_3D_skeleton ( skids, remote_instance = None , connector_flag = 1, tag_flag = 1, get_history = False, get_merge_history = False, time_out = None, get_abutting = False, project_id = 1):
    """ Wrapper to retrieve the skeleton data for a list of skeleton ids

    Parameters:
    ----------
    skids :             single or list of skeleton ids
    remote_instance :   CATMAID instance
                        Either pass directly to function or define globally 
                        as 'remote_instance'
    connector_flag :    set if connector data should be retrieved. 
                        Possible values = 0/False or 1/True
                        Note: the CATMAID API endpoint does currently not
                        support retrieving abutting connectors this way.
                        Please use <get_abutting = True> to set an additional 
                        flag.
    tag_flag :          set if tags should be retrieved. 
                        Possible values = 0/False or 1/True    
    time_out :          integer or None
                        After this number of second, fetching skeleton data 
                        will time out (so as to not block the system)
                        If set to None, time out will be max([ 20, len(skids) ]) 
                        -> e.g. 100s for 100 skeletons but at least 20s
    get_history:        boolean
                        if True, the returned skeleton data will contain 
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
    get_abutting:       if True, will retrieve abutting connectors
                        For some reason they are not part of /compact-json/, 
                        so we have to retrieve them via /connectors/ and add 
                        them to compact-json -> will give them connector type 3!


    Returns:
    --------
    Pandas dataframe (df): 
        neuron_name   skeleton_id   nodes   connectors  tags
    0   name1           skid1      node_df   conn_df    dict 
    1   name2           skid2      node_df   conn_df    dict
    2   name3           skid3      node_df   conn_df    dict
    ...

    neuron_name and skeleton_id are strings
    nodes and connectors are Pandas dataframes themselves
    tags is a dict: { 'tag' : [ treenode_id, treenode_id, ... ] }

    Dataframe column titles should be self explanatory with one exception:
    conn_df['relation']: 0 (presynaptic), 1 (postsynaptic), 2 (gap junction),
    3 (abutting)
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if type(skids) != type(list()):
        to_retrieve = [ skids ]
    else:
        to_retrieve = skids    

    #Convert tag_flag, connector_tag, get_history and get_merge_history to boolean if necessary
    if type(tag_flag) != type( bool() ):
        tag_flag = tag_flag == 1
    if type(connector_flag) != type( bool() ) :
        connector_flag = connector_flag == 1
    if type(get_history) != type( bool() ) :
        get_history = get_history == 1
    if type(get_merge_history) != type( bool() ) :
        get_merge_history = get_merge_history == 1

    #Generate URLs to retrieve
    urls = []    
    for i, skeleton_id in enumerate(to_retrieve):
        #Create URL for retrieving skeleton data from server with history details
        remote_compact_skeleton_url = remote_instance.get_compact_details_url( project_id , skeleton_id )
        #For compact-details, parameters have to passed as GET 
        remote_compact_skeleton_url += '?%s' % urllib.parse.urlencode( {'with_history': get_history , 'with_tags' : tag_flag , 'with_connectors' : connector_flag  , 'with_merge_history': get_merge_history } )
        #'True'/'False' needs to be lower case
        urls.append (  remote_compact_skeleton_url.lower() )

    skdata = get_urls_threaded( urls, remote_instance, time_out = time_out )   

    #Retrieve abutting
    if get_abutting: 
        remote_instance.logger.debug('Retrieving abutting connectors for %i neurons' % len(to_retrieve) )
        urls = []        

        for s in skids: 
            get_connectors_GET_data = { 'skeleton_ids[0]' : str( s ),
                                        'relation_type' : 'abutting' }                    
            urls.append ( remote_instance.get_connectors_url( project_id ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data) )           

        cn_data = get_urls_threaded( urls, remote_instance, time_out = time_out )

        #Add abutting to other connectors in skdata with type == 2
        for i,cn in enumerate(cn_data):
            if not get_history:
                skdata[i][1] += [ [ c[7], c[1], 3, c[2], c[3], c[4] ]  for c in cn['links'] ]
            else:
                skdata[i][1] += [ [ c[7], c[1], 3, c[2], c[3], c[4], c[8], None ]  for c in cn['links'] ]   

    names = get_names( to_retrieve, remote_instance )   

    if not get_history:        
        df = pd.DataFrame( [ [ 
                                names[ str(to_retrieve[i]) ],
                                str(to_retrieve[i]),                            
                                pd.DataFrame( n[0], columns = ['treenode_id','parent_id','creator_id','x','y','z','radius','confidence'], dtype = object ),
                                pd.DataFrame( n[1], columns = ['treenode_id','connector_id','relation','x','y','z'], dtype = object ),
                                n[2]  ]
                                 for i,n in enumerate( skdata ) 
                            ], 
                            columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                            dtype=object
                            )
    else:
        df = pd.DataFrame( [ [ 
                                names[ str(to_retrieve[i]) ],
                                str(to_retrieve[i]),                            
                                pd.DataFrame( n[0], columns = ['treenode_id','parent_id','creator_id','x','y','z','radius','confidence','creation_date','last_modified'], dtype = object ),
                                pd.DataFrame( n[1], columns = ['treenode_id','connector_id','relation','x','y','z','creation_date','last_modified'], dtype = object ),
                                n[2]  ]
                                 for i,n in enumerate( skdata ) 
                            ], 
                            columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                            dtype=object
                            )

    return df    

def get_arbor ( skids, remote_instance = None, node_flag = 1, connector_flag = 1, tag_flag = 1, project_id = 1 ):
    """ Wrapper to retrieve the skeleton data for a list of skeleton ids.
    Similar to get_3D_skeleton() but the connector data includes the whole
    chain: treenode1 -> (link_confidence) -> connector -> (link_confidence)
    -> treenode2. This means that connectors can shop up multiple times! 
    I.e. if they have multiple postsynaptic targets. 
    Does include connector x,y,z coordinates.

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'
    connector_flag :    set if connector data should be retrieved. 
                        Values = 0 or 1. (optional, default = 1)
    tag_flag :          set if tags should be retrieved. Values = 0 or 1. 
                        (optional, default = 1)

    Returns:
    --------
    Pandas dataframe (df): 
        neuron_name   skeleton_id   nodes   connectors  tags
    0   name1           skid1      node_df   conn_df    dict 
    1   name2           skid2      node_df   conn_df    dict
    2   name3           skid3      node_df   conn_df    dict
    ...

    neuron_name and skeleton_id are strings
    nodes and connectors are Pandas dataframes themselves
    tags is a dict: { 'tag' : [ treenode_id, treenode_id, ... ] }

    Dataframe column titles should be self explanatory with these exception:
    conn_df['relation_1'] describes treenode_1 to/from connector
    conn_df['relation_2'] describes treenode_2 to/from connector

    relations can be: 0 (presynaptic), 1 (postsynaptic), 2 (gap junction)
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    skdata = []

    for skeleton_id in skids:
        #Create URL for retrieving example skeleton from server
        remote_compact_arbor_url = remote_instance.get_compact_arbor_url( project_id , skeleton_id, node_flag, connector_flag, tag_flag )

        #Retrieve node_data for example skeleton
        arbor_data = remote_instance.fetch( remote_compact_arbor_url )

        skdata.append(arbor_data)

        remote_instance.logger.debug('%s retrieved' % str(skeleton_id) )    

    names = get_names( skids, remote_instance )

    df = pd.DataFrame( [ [ 
                            names[ str(skids[i]) ],
                            str(skids[i]),                            
                            pd.DataFrame( n[0], columns = ['treenode_id','parent_id','creator_id','x','y','z','radius','confidence'] ) , 
                            pd.DataFrame( n[1], columns=['treenode_1', 'link_confidence', 'connector_id', 'link_confidence','treenode_2', 'other_skeleton_id' , 'relation_1', 'relation_2' ]),
                            n[2]  ]
                             for i,n in enumerate( skdata ) 
                        ], 
                        columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                        dtype=object
                        )
    return df

def get_partners_in_volume(skids, volume, remote_instance = None , threshold = 1, project_id = 1, min_size = 2, approximate = False):
    """ Wrapper to retrieve the synaptic/gap junction partners of neurons 
    of interest WITHIN a given Catmaid Volume. Attention: total number of 
    connections returned is not restricted to that volume.

    Parameters:
    ----------
    skids :             list of skeleton ids
    volume :            string - name of volume in Catmaid
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'                        
    threshold :         does not seem to have any effect on CATMAID API and is 
                        therefore filtered afterwards. This threshold is 
                        applied to the TOTAL number of synapses across all
                        neurons!  
                        (optional, default = 1)
    min_size :          minimum node count of partner
                        (optional, default = 2 -> hide single-node partner)
    approximate :       boolean (default = False)
                        if True, bounding box around the volume is used. Will
                        speed up calculations a lot!

    Returns:
    ------- 
    Pandas dataframe (df): 
        neuron_name   skeleton_id   num_nodes   relation      skid1  skid2 ....
    0   name1           skid1      node_count1  upstream      n_syn  n_syn ...
    1   name2           skid2      node_count2  downstream    n_syn  n_syn ..   
    2   name3           skid3      node_count3  gapjunction   n_syn  n_syn .
    ...    

    Relation can be: upstream (incoming), downstream (outgoing) of the neurons
    of interest or gap junction

    NOTE: 
    (1) partners can show up multiple times if they are e.g. pre- AND 
    postsynaptic
    (2) the number of connections between two partners is not restricted to the
    volume
    """

    try:
       from morpho import in_volume
    except:
       from pymaid.morpho import in_volume

    #First, get list of connectors
    cn_data = get_connectors ( skids, remote_instance = remote_instance, 
                              incoming_synapses = True, outgoing_synapses = True, 
                              abutting = False, gap_junctions = True, project_id = project_id)

    remote_instance.logger.info('%i connectors retrieved - now checking for intersection with volume...' % cn_data.shape[0] )

    #Find out which connectors are in the volume of interest
    iv = in_volume(  cn_data[ ['x','y','z'] ], volume, remote_instance, approximate = approximate )

    #Get the subset of connectors within the volume
    cn_in_volume = cn_data[ iv ].copy()

    remote_instance.logger.info('%i connectors in volume - retrieving connected neurons...' % cn_in_volume.shape[0] )

    #Get details and extract connected skids
    cn_details = get_connector_details ( cn_in_volume.connector_id.unique().tolist() , remote_instance = remote_instance , project_id = project_id )    
    skids_in_volume = list( set( cn_details.presynaptic_to.tolist() + [ n for l in cn_details.postsynaptic_to.tolist() for n in l ] ))

    skids_in_volume = [ str(s) for s in skids_in_volume ]

    #Get all connectivity
    connectivity = get_partners(skids, remote_instance = remote_instance , 
                                threshold = threshold, project_id = project_id, 
                                min_size = min_size)    

    #Filter and return connectivity
    filtered_connectivity = connectivity[ connectivity.skeleton_id.isin( skids_in_volume ) ].copy().reset_index()

    remote_instance.logger.info('%i unique partners left after filtering (%i of %i connectors in given volume)' % ( len( filtered_connectivity.skeleton_id.unique() ) ,cn_in_volume.shape[0], cn_data.shape[0] ) )

    return filtered_connectivity

def get_partners (skids, remote_instance = None , threshold = 1, project_id = 1, min_size = 2, filt = [], directions = ['incoming','outgoing'] ):
    """ Wrapper to retrieve the synaptic/gap junction partners of neurons 
    of interest

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'
    threshold :         does not seem to have any effect on CATMAID API and is 
                        therefore filtered afterwards. This threshold is 
                        applied to the total number of synapses. 
                        (optional, default = 1)
    min_size :          minimum node count of partner
                        (optinal, default = 2 -> hide single-node partners!)
    filt :              list of strings (optional, default = [])
                        filters for neuron names (partial matches sufficient)
    directions :        list of strings - default = ['incoming', 'outgoing']
                        use to restrict to either up- or downstream partners

    Returns:
    ------- 
    Pandas dataframe (df): 
        neuron_name   skeleton_id   num_nodes   relation      skid1  skid2 ....
    0   name1           skid1      node_count1  upstream      n_syn  n_syn ...
    1   name2           skid2      node_count2  downstream    n_syn  n_syn ..   
    2   name3           skid3      node_count3  gapjunction   n_syn  n_syn .
    ...    

    Relation can be: upstream (incoming), downstream (outgoing) of the neurons
    of interest or gap junction

    NOTE: partners can show up multiple times if they are e.g. pre- AND 
    postsynaptic
    """

    def constructor_helper(entry,skid):
        """ Helper to extract connectivity from data returned by CATMAID server
        """        
        try:            
            return entry['skids'][ str(skid) ]
        except:            
            return 0

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_connectivity_url = remote_instance.get_connectivity_url( project_id )

    connectivity_post = {}    
    connectivity_post['boolean_op'] = 'OR'
    i = 0
    for skid in skids:
        tag = 'source_skeleton_ids[%i]' %i
        connectivity_post[tag] = skid
        i +=1    


    remote_instance.logger.info('Fetching connectivity')   
    connectivity_data = remote_instance.fetch( remote_connectivity_url , connectivity_post )

    #As of 08/2015, # of synapses is returned as list of nodes with 0-5 confidence: {'skid': [0,1,2,3,4,5]}
    #This is being collapsed into a single value before returning it:    
    
    for d in directions:
        pop = set()
        for entry in connectivity_data[ d ]:
            if sum( [ sum(connectivity_data[ d ][entry]['skids'][n]) for n in connectivity_data[ d ][entry]['skids'] ] ) >= threshold:
                for skid in connectivity_data[ d ][entry]['skids']:                
                    connectivity_data[ d ][entry]['skids'][skid] = sum(connectivity_data[ d ][entry]['skids'][skid])
            else:
                pop.add(entry)

            if min_size > 1:                
                if connectivity_data[ d ][entry]['num_nodes'] < min_size:
                    pop.add(entry)

        for n in pop:
            connectivity_data[ d ].pop(n)

    remote_instance.logger.info('Done. Found %i up- %i downstream neurons' % ( len(connectivity_data['incoming']) , len(connectivity_data['outgoing']) ) )

    names = get_names( list(connectivity_data['incoming']) + list(connectivity_data['outgoing']) + skids, remote_instance )    

    df = pd.DataFrame( columns = ['neuron_name','skeleton_id','num_nodes','relation'] + [ str(s) for s in skids ] )

    relations = { 
                    'incoming' : 'upstream' ,
                    'outgoing' : 'downstream',
                    'gapjunctions' : 'gapjunction'                    
                }

    for d in relations:
        df_temp = pd.DataFrame( [ [ 
                            names[ str( n ) ],
                            str( n ),
                            int(connectivity_data[ d ][ n ]['num_nodes']),
                            relations[d] ] 
                            + [ constructor_helper( connectivity_data[d][n], s ) for s in skids ]                            
                            for i,n in enumerate( connectivity_data[d] ) 
                        ], 
                        columns = ['neuron_name','skeleton_id','num_nodes','relation'] + [ str(s) for s in skids ],
                        dtype=object
                        )          

        df = pd.concat( [df , df_temp], axis = 0)

    if filt:
        f = [ True in [ f in name for f in filt ] for name in df.neuron_name.tolist() ]
        df = df [ f ]

    #Reindex concatenated dataframe
    df.index = range( df.shape[0] )
        
    return df
    

def get_names (skids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve neurons names for a list of skeleton ids

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'

    Returns:
    ------- 
    dict :              { skid1 : 'neuron_name', skid2 : 'neuron_name',  .. }
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    skids = list( set(skids) )

    remote_get_names_url = remote_instance.get_neuronnames( project_id )

    get_names_postdata = {}
    get_names_postdata['pid'] = 1
        
    for i in range(len(skids)):
        key = 'skids[%i]' % i
        get_names_postdata[key] = skids[i]

    names = remote_instance.fetch( remote_get_names_url , get_names_postdata )

    remote_instance.logger.debug( 'Names for %i of %i skeleton IDs retrieved' % ( len( names ), len ( skids ) ) )
        
    return(names)

def get_node_user_details(treenode_ids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve user info for a list of treenode and/or connectors

    Parameters:
    ----------
    treenode_ids :      list of treenode ids (can also be connector ids!)
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'

    Returns:
    ------- 
    Pandas dataframe :
      treenode_id creation_time user edition_time editor reviewers review_times
    1    int       timestamp   id    timestamp    id     id list timestamp list
    2
    ..
    """

    if type(treenode_ids) != type(list()):
        treenode_ids = [treenode_ids]

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_instance.logger.info('Retrieving details for %i nodes...' % len(treenode_ids))

    remote_nodes_details_url = remote_instance.get_node_info_url( project_id )

    get_node_details_postdata = {
                                }

    for i, tn in enumerate(treenode_ids):
        key = 'node_ids[%i]' % i
        get_node_details_postdata[key] = tn
   
    data = remote_instance.fetch( remote_nodes_details_url, get_node_details_postdata )

    data_columns = ['creation_time', 'user',  'edition_time', 'editor', 'reviewers', 'review_times' ]

    df = pd.DataFrame(
                        [ [ e ] + [ data[e][k] for k in data_columns ] for e in data.keys() ],
                        columns = ['treenode_id'] + data_columns,
                        dtype = object
                      )

    df['creation_time'] = [ datetime.datetime.strptime( d[:16] , '%Y-%m-%dT%H:%M' ) for d in df['creation_time'].tolist()  ]
    df['edition_time'] = [ datetime.datetime.strptime( d[:16] , '%Y-%m-%dT%H:%M' ) for d in df['edition_time'].tolist() ]
    df['review_times'] = [ [ datetime.datetime.strptime( d[:16] , '%Y-%m-%dT%H:%M' ) for d in lst ] for lst in df['review_times'].tolist() ]    

    return df

def get_node_lists (skids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve treenode table for a list of skids

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'

    Returns:
    ------- 
    dict :      
    { skid1 : Pandas dataframe =
     node_id parent_id confidence x y z radius creator last_edition reviewers tag
    0  123     124      5        ...    
    1  124     125      5        ..
    2  125     126      5        . 
    ...
    }
    """

    if type(skids) != type(list()):
        skids = [skids]

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_instance.logger.info('Retrieving %i node table(s)...' % len(skids))   

    user_list = get_user_list(remote_instance)

    user_dict = user_list.set_index('id').T.to_dict()

    nodes = {}    
    for run ,skid in enumerate(skids):

        remote_nodes_list_url = remote_instance.get_skeleton_nodes_url( project_id , skid )

        remote_instance.logger.debug('Retrieving node table of %s [%i of %i]...' % (str(skid),run,len(skids)))        

        try:
            node_list = remote_instance.fetch( remote_nodes_list_url )
        except:
            remote_instance.logger.warning('Time out while retrieving node table - retrying...')
            time.sleep(.5)
            try:
                node_list = remote_instance.fetch( remote_nodes_list_url )
                remote_instance.logger.warning('Success on second attempt!')
            except:
                remote_instance.logger.error('Unable to retrieve node table')        

        #Format of node_list: node_id, parent_node_id, confidence, x, y, z, radius, creator, last_edition_timestamp
        tag_dict = { n[0] : [] for n in node_list[0] }
        [ tag_dict[ n[0] ].append( n[1] ) for n in node_list[2] ]

        reviewer_dict = { n[0] : [] for n in node_list[0] }
        [ reviewer_dict[ n[0] ].append( user_list[ user_list.id == n[1] ]['login'].values[0] ) for n in node_list[1] ]

        nodes[skid] = pd.DataFrame( [ n + [ reviewer_dict[n[0]], tag_dict[n[0]] ] for n in node_list[0] ] ,
                                    columns = [ 'node_id', 'parent_node_id', 'confidence', 'x', 'y', 'z', 'radius', 'creator', 'last_edition_timestamp', 'reviewers', 'tags' ],
                                    dtype=object
                                    )

        #Replace creator_id with their login
        nodes[skid]['creator'] = [ user_dict[u]['login'] for u in nodes[skid]['creator'] ]

        remote_instance.logger.debug('Done')         

    return nodes

def get_edges (skids, remote_instance = None):
    """ Wrapper to retrieve edges (synaptic connections) between sets of neurons
    
    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'

    Returns:
    --------
    Pandas dataframe :
        source_skid     target_skid     weight 
    1   123             345                5
    2   345             890                4
    2   567             123                1

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_edges_url = remote_instance.get_edges_url( 1 )

    get_edges_postdata = {}
    get_edges_postdata['confidence_threshold'] = '0'
        
    for i in range(len(skids)):
        key = 'skeleton_ids[%i]' % i
        get_edges_postdata[key] = skids[i]

    edges = remote_instance.fetch( remote_get_edges_url , get_edges_postdata )

    df = pd.DataFrame( [ [ e[0], e[1], sum(e[2]) ] for e in edges['edges'] ],
                        columns = ['source_skid','target_skid','weight']
                        )
        
    return df

def get_connectors ( skids, remote_instance = None, incoming_synapses = True, outgoing_synapses = True, abutting = False, gap_junctions = False, project_id = 1):
    """ Wrapper to retrieve connectors for a set of neurons.    
    
    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance 
                        either pass directly to function or define 
                        globally as 'remote_instance'
    incoming_synapses : boolean (default = True)
                        if True, incoming synapses will be retrieved
    outgoing_synapses : boolean (default = True)
                        if True, outgoing synapses will be retrieved
    abutting :          boolean (default = False)
                        if True, abutting connectors will be retrieved
    gap_junctions :     boolean (default = False)
                        if True, gap junctions will be retrieved
    project_id :        int (default = 1)
                        ID of the CATMAID project

    Returns:
    ------- 
    Pandas dataframe (df) containing the following columns.     
    
      skeleton_id  connector_id  x  y  z  confidence  creator_id  treenode_id  \
    0
    1
    ...

      creation_time  edition_time type
    0
    1
    ...

    Please note: Each row represents a link (connector <-> treenode)! Connectors
    may thus show up in multiple rows. Use e.g. df.connector_id.unique() to get 
    a set of unique connector IDs.
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return    

    if type(skids) != type(list()):
        skids = [ skids ]

    get_connectors_GET_data = { 'with_tags': 'false' }

    cn_data = []

    #There seems to be some cap regarding how many skids you can send to the server, so we have to chop it into pieces
    for a in range( 0, len(skids), 50 ):
        for i,s in enumerate(skids[ a:a+50] ):
            tag = 'skeleton_ids[%i]' % i 
            get_connectors_GET_data[tag] = str( s )

        if incoming_synapses is True:
            get_connectors_GET_data['relation_type']='presynaptic_to'
            remote_get_connectors_url = remote_instance.get_connectors_url( project_id ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [ e + ['presynaptic_to'] for e in remote_instance.fetch( remote_get_connectors_url )['links'] ]

        if outgoing_synapses is True:
            get_connectors_GET_data['relation_type']='postsynaptic_to'
            remote_get_connectors_url = remote_instance.get_connectors_url( project_id ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [ e + ['postsynaptic_to'] for e in remote_instance.fetch( remote_get_connectors_url )['links'] ]

        if abutting is True:
            get_connectors_GET_data['relation_type']='abutting'
            remote_get_connectors_url = remote_instance.get_connectors_url( project_id ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [ e + ['abutting'] for e in remote_instance.fetch( remote_get_connectors_url )['links'] ]

        if gap_junctions is True:
            get_connectors_GET_data['relation_type']='gapjunction_with'
            remote_get_connectors_url = remote_instance.get_connectors_url( project_id ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
            cn_data += [ e + ['gap_junction'] for e in remote_instance.fetch( remote_get_connectors_url )['links'] ]              

    df = pd.DataFrame(  cn_data,
                        columns = [ 'skeleton_id', 'connector_id', 'x', 'y', 'z', 'confidence', 'creator_id', 'treenode_id', 'creation_time', 'edition_time' , 'type' ],
                        dtype=object
                         )

    remote_instance.logger.info('%i connectors for %i neurons retrieved' % ( df.shape[0], len(skids) ) )

    return df


def get_connector_details (connector_ids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve details on sets of connectors 
    
    Parameters:
    ----------
    connector_ids :     list of connector ids; can be found e.g. 
                        from compact skeletons (get_3D_skeleton)
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns:
    ------- 
    Pandas dataframe.
    Each row is a connector and contains the following data:

      connector_id  presynaptic_to  postsynaptic_to  presynaptic_to_node \
    0
    1
    2

      postsynaptic_to_node
    0
    1
    2    
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_connectors_url = remote_instance.get_connector_details_url( project_id )       
    
    #Depending on DATA_UPLOAD_MAX_NUMBER_FIELDS of your CATMAID server (default = 1000), we have to cut requests into batches < 1000
    connectors = []
    #for b in range( 0, len( connector_ids ), 999 ):
    get_connectors_postdata = {}         
    for i,s in enumerate( connector_ids ):    
        key = 'connector_ids[%i]' % i
        get_connectors_postdata[key] = s #connector_ids[i]

    connectors += remote_instance.fetch( remote_get_connectors_url , get_connectors_postdata )

    remote_instance.logger.info('Data for %i of %i unique connector IDs retrieved' %(len(connectors),len(set(connector_ids))))    

    columns = [ 'connector_id', 'presynaptic_to', 'postsynaptic_to', 'presynaptic_to_node', 'postsynaptic_to_node' ]

    df = pd.DataFrame( [ [ cn[0] ] + [ cn[1][e] for e in columns[1:]] for cn in connectors ],
                        columns = columns,
                        dtype=object
                         )

    return df

def get_review (skids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve review status for a set of neurons
    
    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns:
    ------- 
    Pandas dataframe:

       skeleton_id neuron_name  total_node_count nodes_reviewed percent_reviewed
    0    
    1
    2   

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_reviews_url = remote_instance.get_review_status( project_id )

    get_review_postdata = {}    
        
    for i in range(len(skids)):
        key = 'skeleton_ids[%i]' % i
        get_review_postdata[key] = str(skids[i])

    review_status = remote_instance.fetch( remote_get_reviews_url , get_review_postdata )  

    names = get_names(skids, remote_instance)

    df = pd.DataFrame ( [ [     s,
                                names[ str(s) ],  
                                review_status[s][0], 
                                review_status[s][1], 
                                int(review_status[s][1]/review_status[s][0]*100)  
                            ] for s in review_status ],
                        columns = ['skeleton_id', 'neuron_name', 'total_node_count', 'nodes_reviewed', 'percent_reviewed']
                         )
        
    return df
   
def get_neuron_annotation (skid, remote_instance = None, project_id = 1 ):
    """ Wrapper to retrieve annotations of a SINGLE neuron. 
    Contains timestamps and user_id
    
    Parameters:
    ----------
    skid :              string or int 
                        Single skeleton id.
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns:
    ------- 
    Pandas dataframe:

        annotation  time_annotated  unknown  user_id  annotation_id  user
    0
    1
    ..
    .
    
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    #This works with neuron_id NOT skeleton_id
    #neuron_id can be requested via neuron_names
    remote_get_neuron_name = remote_instance.get_single_neuronname_url( project_id , skid )
    neuronid = remote_instance.fetch( remote_get_neuron_name )['neuronid']

    remote_get_annotations_url = remote_instance.get_neuron_annotations( project_id )

    get_annotations_postdata = {}            
    get_annotations_postdata['neuron_id'] = int(neuronid)   

    annotations = remote_instance.fetch( remote_get_annotations_url , get_annotations_postdata )['aaData']    

    user_list = get_user_list(remote_instance)

    annotations = [ a + [ user_list[ user_list.id == a[3] ]['login'].values[0] ] for a in annotations]    

    df = pd.DataFrame( annotations,
                       columns = ['annotation', 'time_annotated', 'unknown', 'user_id', 'annotation_id', 'user'],
                       dtype = object
                        )

    df.sort_values('annotation', inplace = True)
        
    return df 

def has_soma ( skids , remote_instance = None, project_id = 1 ):
    """ Quick function to check if a neuron/a list of neurons have somas
    Searches for nodes that have a 'soma' tag AND a radius > 500nm

    Parameters:
    ----------
    skids :             single skeleton id or list of skids
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns
    -------
    dictionary :        { 'skid' : True, 'skid1' : False }
    """

    if type(skids) != type(list()):
        skids = [skids]

    skdata = get_3D_skeleton ( skids, remote_instance = remote_instance , 
                            connector_flag = 0, tag_flag = 1, 
                            get_history = False, time_out = None, 
                            project_id = project_id)

    d = {}
    for i, s in enumerate(skids):
        d[s] = False
        if 'soma' not in skdata[i][2]:
            continue        

        for tn in skdata[i][2]['soma']:
            if [ n for n in skdata[i][0] if n[0] == tn ][0][6] > 500:
                d[s] = True

    return d


def skid_exists( skid, remote_instance = None, project_id = 1 ):
    """ Quick function to check if skeleton id exists
    
    Parameters:
    ----------
    skid :              single skeleton id
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns:
    -------
    True if skeleton exists, False if not
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_neuron_name = remote_instance.get_single_neuronname_url( project_id , skid )
    response = remote_instance.fetch( remote_get_neuron_name )
    
    if 'error' in response:
        return False
    else:
        return True  

def get_annotation_id( annotations, remote_instance = None, project_id = 1, allow_partial = False ):
    """ Wrapper to retrieve the annotation ID for single or list of annotation(s)
    
    Parameters:
    ----------
    annotations :       single annotations or list of multiple annotations
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'
    allow_partial :     boolean
                        If True, will allow partial matches

    Returns:
    -------
    dict :              { 'annotation_name' : 'annotation_id', ....}
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_instance.logger.info('Retrieving list of annotations...')

    remote_annotation_list_url = remote_instance.get_annotation_list( project_id )
    annotation_list = remote_instance.fetch( remote_annotation_list_url )
         
    annotation_ids = {}
    annotations_matched = set()

    if type(annotations) == type(str()):
        for d in annotation_list['annotations']:            
            if d['name'] == annotations and allow_partial is False:                
                annotation_ids[ d['name'] ] = d['id']
                remote_instance.logger.debug('Found matching annotation: %s' % d['name'] )
                annotations_matched.add( d['name'] )
                break
            elif annotations in d['name'] and allow_partial is True:
                annotation_ids[ d['name'] ] = d['id']
                remote_instance.logger.debug('Found matching annotation: %s' % d['name'] )      

        if not annotation_ids:
            remote_instance.logger.warning('Could not retrieve annotation id for: ' + annotations )  

    elif type(annotations) == type(list()):        
        for d in annotation_list['annotations']:            
            if d['name'] in annotations and allow_partial is False:
                annotation_ids[ d['name'] ] = d['id'] 
                annotations_matched.add( d['name'] )
                remote_instance.logger.debug('Found matching annotation: %s' % d['name'] ) 
            elif True in [ a in d['name'] for a in annotations ] and allow_partial is True:
                annotation_ids[ d['name'] ] = d['id']
                annotations_matched |= set ( [ a for a in annotations if a in d['name'] ] )
                remote_instance.logger.debug('Found matching annotation: %s' % d['name'] ) 
    
        if len(annotations) != len(annotations_matched):
            remote_instance.logger.warning('Could not retrieve annotation id(s) for: ' + str( [ a for a in annotations if a not in annotations_matched ] ) )

    return annotation_ids

def get_skids_by_name(tag, remote_instance = None, allow_partial = True, project_id = 1):
    """ Wrapper to retrieve the all neurons with matching name
    
    Parameters:
    ----------
    tag :               name to search for
    allow_partial :     if True, partial matches are returned too    
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns:
    -------
    Pandas dataframe:

        name   skeleton_id
    0
    1
    2
    
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    search_url = remote_instance.get_annotated_url( project_id )
    annotation_post = { 'name': str(tag) , 'rangey_start': 0, 'range_length':500, 'with_annotations':False } 

    results = remote_instance.fetch( search_url, annotation_post )     

    match = []
    for e in results['entities']:
        if allow_partial and e['type'] == 'neuron' and tag.lower() in e['name'].lower():
            match.append( [ e['name'], e['skeleton_ids'][0] ] )
        if not allow_partial and e['type'] == 'neuron' and e['name'] == tag:
            match.append( [ e['name'], e['skeleton_ids'][0] ] )

    df = pd.DataFrame(  match,
                        columns = ['name', 'skeleton_id']
                         )

    df.sort_values( ['name'], inplace = True)

    return df

def get_skids_by_annotation( annotations, remote_instance = None, project_id = 1, allow_partial = False ):
    """ Wrapper to retrieve the all neurons annotated with given annotation(s)
    
    Parameters:
    ----------
    annotations :          single annotation or list of multiple annotations    
    remote_instance :      CATMAID instance; either pass directly to function 
                           or define globally as 'remote_instance'
    allow_partial :        allow partial match of annotation

    Returns:
    -------
    list :                  [ skid1, skid2, skid3 ]
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_instance.logger.info('Looking for Annotation(s): ' + str(annotations) )
    annotation_ids = get_annotation_id(annotations, remote_instance, project_id = 1, allow_partial = allow_partial)

    if not annotation_ids:
        remote_instance.logger.warning('No matching annotation found! Returning None')
        return []

    if allow_partial is True:
        remote_instance.logger.debug('Found id(s): %s (partial matches included)' % len(annotation_ids) )  
    elif type(annotations) == type( list() ):
        remote_instance.logger.debug('Found id(s): %s | Unable to retrieve: %i' % ( str(annotation_ids) , len(annotations)-len(annotation_ids) ))      
    elif type(annotations) == type( str() ):
        remote_instance.logger.debug('Found id: %s | Unable to retrieve: %i' % ( list(annotation_ids.keys())[0], 1 - len(annotation_ids) ))

    annotated_skids = []
    remote_instance.logger.info('Retrieving skids for annotationed neurons...')
    for an_id in annotation_ids.values():
        #annotation_post = {'neuron_query_by_annotation': annotation_id, 'display_start': 0, 'display_length':500}
        annotation_post = {'annotated_with0': an_id, 'rangey_start': 0, 'range_length':500, 'with_annotations':False}
        remote_annotated_url = remote_instance.get_annotated_url( project_id )
        neuron_list = remote_instance.fetch( remote_annotated_url, annotation_post )
        count = 0    
        for entry in neuron_list['entities']:
            if entry['type'] == 'neuron':
                annotated_skids.append(str(entry['skeleton_ids'][0]))

    remote_instance.logger.info('Found %i skeletons with matching annotation(s)' % len(annotated_skids) )
        
    return(annotated_skids)

def get_annotations_from_list(skid_list, remote_instance = None, project_id = 1 ):
    """ Wrapper to retrieve annotations for a list of skeleton ids
    If a neuron has no annotations, it will not show up in returned dict!
    Note: this API endpoint does not process more than 250 skids at a time!

    Parameters:
    ----------
    skid_list :         list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns:
    --------
    dict :              { skeleton_id : [annnotation, annotation ], ... }
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_annotations_url = remote_instance.get_annotations_for_skid_list2( project_id )

    get_annotations_postdata = {'metaannotations':0,'neuronnames':0}  

    for i in range(len(skid_list)):
        #key = 'skids[%i]' % i
        key = 'skeleton_ids[%i]' % i
        get_annotations_postdata[key] = str(skid_list[i])

    annotation_list_temp = remote_instance.fetch( remote_get_annotations_url , get_annotations_postdata )

    annotation_list = {}    
    
    for skid in annotation_list_temp['skeletons']:
        annotation_list[skid] = []        
        #for entry in annotation_list_temp['skeletons'][skid]:
        for entry in annotation_list_temp['skeletons'][skid]['annotations']:
            annotation_id = entry['id']
            annotation_list[skid].append(annotation_list_temp['annotations'][str(annotation_id)])      
   
    return(annotation_list) 

def add_tags ( node_list, tags, node_type, remote_instance = None, project_id = 1 ):
    """ Wrapper to add tag(s) to a list of treenode(s) or connector(s)

    Parameters:
    ----------
    node_list :         list of treenode or connector ids that will be tagged
    tags :              list of tags(s) to add to provided treenode/connector ids
    node_type :         string
                        set to 'TREENODE' or 'CONNECTOR' depending on 
                        what you want to tag
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'

    Returns confirmations from Catmaid server
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if type(node_list) != type(list()):
        node_list = [ node_list ]

    if type(tags) != type(list()):
        tags = [ tags ]


    if node_type == 'TREENODE':
        add_tags_urls = [ remote_instance.treenode_add_tag_url( project_id, n) for n in node_list ]
    elif node_type == 'CONNECTOR':
        add_tags_urls = [ remote_instance.connector_add_tag_url( project_id, n) for n in node_list ]

    post_data = [ {'tags': ','.join(tags) , 'delete_existing': False } for n in node_list ]

    d = get_urls_threaded( add_tags_urls , remote_instance, post_data = post_data, time_out = None )   
    
    return d

def add_annotations ( skid_list, annotations, remote_instance = None, project_id = 1 ):
    """ Wrapper to add annotation(s) to a list of neuron(s)

    Parameters:
    ----------
    skid_list :         list of skeleton ids that will be annotated
    annotations :       list of annotation(s) to add to provided skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns nothing
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if type(skid_list) != type(list()):
        skid_list = [skid_list]

    if type(annotations) != type(list()):
        annotations = [annotations]

    add_annotations_url = remote_instance.get_add_annotations_url ( project_id )

    add_annotations_postdata = {}

    for i in range(len(skid_list)):
        key = 'skeleton_ids[%i]' % i
        add_annotations_postdata[key] = str( skid_list[i] )

    for i in range(len(annotations)):
        key = 'annotations[%i]' % i
        add_annotations_postdata[key] = str( annotations[i] )

    remote_instance.logger.info( remote_instance.fetch( add_annotations_url , add_annotations_postdata ) )

    return 

def get_review_details ( skid_list, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve review status (reviewer + timestamp) for each node 
    of a given skeleton -> uses the review API

    Parameters:
    -----------
    skid_list :         list of skeleton ids to check
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'

    Returns:
    -------
    list :              list of reviewed nodes 
                        [   node_id, 
                            [ 
                            [reviewer1, timestamp],
                            [ reviewer2, timestamp] 
                            ] 
                        ]
    """

    node_list = []    
    urls = []
    post_data = []

    for skid in skid_list:
        urls.append ( remote_instance.get_review_details_url( project_id , skid ) )
        #For some reason this needs to fetched as POST (even though actual POST data is not necessary)
        post_data.append ( { 'placeholder' : 0 } )

    rdata = get_urls_threaded( urls , remote_instance, post_data = post_data, time_out = None )           

    for neuron in rdata:
        #There is a small chance that nodes are counted twice but not tracking node_id speeds up this extraction a LOT
        #node_ids = []
        for arbor in neuron:            
            node_list += [ ( n['id'] , n['rids'] ) for n in arbor['sequence'] if n['rids'] ]

    return node_list

def get_logs (remote_instance = None, operations = [] , entries = 50 , display_start = 0, project_id = 1, search = ''):
    """ Wrapper to retrieve log (like log widget)
    
    Parameters:
    ----------    
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'
    operations :        list of strings
                        if empty, all operations will be queried from server
                        possible operations: 'join_skeleton', 
                        'change_confidence', 'rename_neuron', 'create_neuron', 
                        'create_skeleton', 'remove_neuron', 'split_skeleton', 
                        'reroot_skeleton', 'reset_reviews', 'move_skeleton'
    entries :           integer (default = 50)
                        number of entries to retrieve
    project_id :        integer (default = 1)
                        Id of your CATMAID project

    Returns:
    -------
    Pandas DataFrame:
    
        user   operation   timestamp   x  y  z  explanation 
    0
    1
    ...      
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if not operations:
        operations = [-1]   

    logs = []
    for op in operations:
        get_logs_postdata =  {   'sEcho':6, 
                                'iColumns':7, 
                                'iDisplayStart':display_start, 
                                'iDisplayLength':entries,
                                'mDataProp_0':0,
                                'sSearch_0': '',
                                'bRegex_0':False,
                                'bSearchable_0':False,
                                'bSortable_0':True,
                                'mDataProp_1':1,
                                'sSearch_1': '',
                                'bRegex_1':False,
                                'bSearchable_1':False,
                                'bSortable_1':True,
                                'mDataProp_2':2,
                                'sSearch_2': '',
                                'bRegex_2':False,
                                'bSearchable_2':False,
                                'bSortable_2':True,
                                'mDataProp_3':3,
                                'sSearch_3':'',
                                'bRegex_3':False,
                                'bSearchable_3':False,
                                'bSortable_3':False,
                                'mDataProp_4':4,
                                'sSearch_4': '',
                                'bRegex_4':False,
                                'bSearchable_4':False,
                                'bSortable_4':False,
                                'mDataProp_5':5,
                                'sSearch_5': '',
                                'bRegex_5':False,
                                'bSearchable_5':False,
                                'bSortable_5':False,
                                'mDataProp_6':6,
                                'sSearch_6': '',
                                'bRegex_6':False,
                                'bSearchable_6':False,
                                'bSortable_6':False,
                                'sSearch': '',
                                'bRegex':False,
                                'iSortCol_0':2,
                                'sSortDir_0':'desc',
                                'iSortingCols':1,
                                'pid': project_id,
                                'operation_type': op,
                                'search_freetext': search}
        """
        {
                                'sEcho': 1,
                                'iColumns': 7,
                                'pid' : project_id,
                                'sSearch': '',
                                'bRegex':False,
                                'iDisplayStart' : display_start,                                
                                'iDisplayLength' : entries,
                                'operation_type' : op,
                                'iSortCol_0':2,
                                'sSortDir_0':'desc',
                                'iSortingCols':1,                                
                                'search_freetext' : search
                                } 
        """
    
        remote_get_logs_url = remote_instance.get_logs_url( project_id )
        logs += remote_instance.fetch( remote_get_logs_url, get_logs_postdata )['aaData']

    df = pd.DataFrame(  logs,
                        columns = ['user', 'operation', 'timestamp', 'x', 'y', 'z', 'explanation']
                        )

    return df


def get_contributor_statistics (skids, remote_instance = None, separate = False, project_id = 1):
    """ Wrapper to retrieve contributor statistics for given skeleton ids.
    By default, stats are given over all neurons.
    
    Parameters:
    ----------
    skids :             list of skeleton ids to check
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'
    separate :          boolean (default =False)
                        if true, stats are given per neuron
    project_id :        integer (default = 1)
                        Id of your CATMAID project

    Returns:
    -------
    Pandas dataframe
      skeleton_id node_contributors multiuser_review_minutes post_contributors ..      
    1
    2
    3

    ..  construction_minutes  min_review_minutes  n_postsynapses  n_presynapses .. 
    1
    2
    3

    .. pre_contributors  n_nodes
    1
    2
    3

    dictionary :  { 'node_contributors': {'user_id': nodes_contributed , ...}, 
                    'multiuser_review_minutes': XXX , 
                    'post_contributors': {'user_id': n_connectors_contributed}, 
                    'construction_minutes': XXX, 
                    'min_review_minutes': XXX, 
                    'n_post': XXX, 
                    'n_pre': XXX, 
                    'pre_contributors': {'user_id': n_connectors_contributed}, 
                    'n_nodes': XXX
                   }
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    columns = ['skeleton_id', 'n_nodes', 'node_contributors', 'n_presynapses',  'pre_contributors',  'n_postsynapses', 'post_contributors' ,'multiuser_review_minutes','construction_minutes', 'min_review_minutes' ]

    if not separate:
        get_statistics_postdata = {}    
            
        for i in range(len(skids)):
            key = 'skids[%i]' % i
            get_statistics_postdata[key] = skids[i]        
        
        remote_get_statistics_url = remote_instance.get_contributions_url( project_id )
        stats = remote_instance.fetch( remote_get_statistics_url, get_statistics_postdata )

        df = pd.DataFrame( [ [  
                                skids,
                                stats['n_nodes'],
                                stats['node_contributors'],                                
                                stats['n_pre'],
                                stats['pre_contributors'],
                                stats['n_post'],
                                stats['post_contributors'],
                                stats['multiuser_review_minutes'],
                                stats['construction_minutes'],
                                stats['min_review_minutes']                                
                                ] ],
                            columns = columns,
                            dtype = object
                            )
    else:
        get_statistics_postdata = [ { 'skids[0]' : s } for s in skids  ]
        remote_get_statistics_url = [ remote_instance.get_contributions_url( project_id ) for s in skids  ]

        stats = get_urls_threaded( remote_get_statistics_url , remote_instance, post_data = get_statistics_postdata, time_out = None )

        df = pd.DataFrame( [ [  
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
                                ] for i,s in enumerate(skids) ],
                            columns = columns,
                            dtype = object
                            )
    return df

def get_skeleton_list( remote_instance = None, user=None, node_count=1, start_date=[], end_date=[], reviewed_by = None, project_id = 1 ):
    """ Wrapper to retrieves a list of all skeletons that fit given parameters 
    (see variables). If no parameters are provided, all existing skeletons are 
    returned!

    Parameters:
    ----------
    remote_instance :   class
                        Your CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'.
    user :              integer
                        A single user_id.
    node_count :        integer
                        Minimum number of nodes.
    start_date :        list of integers [year, month, day]
                        Only consider neurons created after.
    end_date :          list of integers [year, month, day]
                        Only consider neurons created before.

    Returns:
    list :              [ skid, skid, skid ]
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    get_skeleton_list_GET_data = {'nodecount_gt':node_count}

    if user:
        get_skeleton_list_GET_data['created_by'] = user

    if reviewed_by:
        get_skeleton_list_GET_data['reviewed_by'] = reviewed_by
    
    if start_date and end_date:
        get_skeleton_list_GET_data['from'] = ''.join( [ str(d) for d in start_date ] )
        get_skeleton_list_GET_data['to'] = ''.join( [ str(d) for d in end_date ] )


    remote_get_list_url = remote_instance.get_list_skeletons_url( project_id )
    remote_get_list_url += '?%s' % urllib.parse.urlencode(get_skeleton_list_GET_data)    
    skid_list = remote_instance.fetch ( remote_get_list_url)

    return skid_list

def get_history( remote_instance = None, project_id = 1, start_date = '2016-10-29', end_date = '2016-11-08', split = True ):    
    """ Wrapper to retrieves CATMAID history 
    Attention: If the time window is too large, the connection might time out 
    which will result in an error!

    Parameters:
    ----------
    remote_instance :   CATMAID instance; either pass directly to function 
                        or define globally as 'remote_instance'
    user :              single user_id    
    start_date :        created after, date needs to be (year,month,day) 
                        format - e.g. '2016-10-29'
    end_date :          created before, date needs to be (year,month,day) 
                        format - e.g. '2017-10-29'
    split :             boolean
                        If True, history will be requested in bouts of 6 months
                        Useful if you want to look at a very big time window 
                        as this can lead to gateway timeout

    Returns:
    --------
    dict :        { 'days': [ date , date , ...], 
                    'stats_table': { user_ID: { date : { 'new_treenodes': int, 
                                                         'new_reviewed_nodes': int,
                                                         'new_connectorts': int },
                                                     },
                                                   } , 
                         'daysformatted': [ days_formated, ... ] }
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    rounds = []
    if split:
        start = datetime.datetime.strptime( start_date , "%Y-%m-%d").date()
        end = datetime.datetime.strptime( end_date , "%Y-%m-%d").date()

        remote_instance.logger.info('Retrieving %i days of history in bouts!' % (end-start).days )

        #First make big bouts of roughly 6 months each
        while start < (end - datetime.timedelta(days=6*30) ) :
            rounds.append( ( start.isoformat(), (start + datetime.timedelta(days=6*30)).isoformat() ) )
            start += datetime.timedelta(days=6*30)

        #Append the last bit
        if start < end:
            rounds.append( ( start.isoformat(), end.isoformat() ) )
    else:
        rounds = [ ( start_date, end_date ) ]

    data = []
    for r in rounds:
        get_history_GET_data = {    'pid': project_id ,
                                    'start_date': r[0],
                                    'end_date': r[1]
                                        }

        remote_get_history_url = remote_instance.get_history_url( project_id )

        remote_get_history_url += '?%s' % urllib.parse.urlencode(get_history_GET_data)

        remote_instance.logger.debug('Retrieving user history from %s to %s ' % ( r[0], r[1] ) )   

        data.append( remote_instance.fetch ( remote_get_history_url ) )
    
    #Now merge data into a single dict
    stats = dict( data[0] )
    for d in data:
        stats['days'] += [ e for e in d['days'] if e not in stats['days'] ]
        stats['daysformatted'] += [ e for e in d['daysformatted'] if e not in stats['daysformatted'] ]

        for u in d['stats_table']:
            stats['stats_table'][u].update( d['stats_table'][u] )

    return stats

def get_neurons_in_volume ( left, right, top, bottom, z1, z2, remote_instance = None, project_id = 1 ):
    """ Retrieves neurons with processes within a defined volume. Because the 
    API returns only a limited number of neurons at a time, the defined volume 
    has to be chopped into smaller pieces for crowded areas - may thus take 
    some time!

    Parameters:
    ----------
    left, right, top, z1, z2 :   Coordinates defining the volumes. Need to be 
                                 in nm, not pixels.
    remote_instance :            CATMAID instance; either pass directly to 
                                 function or define globally as 'remote_instance'
    
    Returns:
    --------
    list :                      [ skeleton_id, skeleton_id, ... ]
    """   

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return 

    def get_nodes( left, right, top, bottom, z1, z2, remote_instance, incursion, project_id ):  

        remote_instance.logger.info( '%i: %i, %i, %i, %i, %i ,%i' % (incursion, left, right, top, bottom, z1, z2) )

        remote_nodes_list = remote_instance.get_node_list_url ( project_id )

        x_y_resolution = 3.8

        #Atnid seems to be related to fetching the active node too (will be ignored if atnid = -1)
        node_list_postdata = {      'left':left * x_y_resolution,
                                    'right':right * x_y_resolution,
                                    'top': top * x_y_resolution,
                                    'bottom': bottom * x_y_resolution,
                                    'z1': z1,
                                    'z2': z2,
                                    'atnid':-1,
                                    'labels': False
                                }

        node_list = remote_instance.fetch( remote_nodes_list , node_list_postdata )

        

        if node_list[3] is True:
            remote_instance.logger.debug('Incursing')   
            incursion += 1         
            node_list = list()
            #Front left top
            node_list += get_nodes( left, 
                                        left + (right-left)/2, 
                                        top, 
                                        top + (bottom-top)/2, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Front right top
            node_list += get_nodes( left  + (right-left)/2, 
                                        right, 
                                        top,
                                        top + (bottom-top)/2, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Front left bottom            
            node_list += get_nodes( left, 
                                        left + (right-left)/2, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Front right bottom
            node_list += get_nodes( left  + (right-left)/2, 
                                        right, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Back left top
            node_list += get_nodes( left, 
                                        left + (right-left)/2, 
                                        top, 
                                        top + (bottom-top)/2, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
            #Back right top
            node_list += get_nodes( left  + (right-left)/2, 
                                        right, 
                                        top,
                                        top + (bottom-top)/2, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
            #Back left bottom            
            node_list += get_nodes( left, 
                                        left + (right-left)/2, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
            #Back right bottom
            node_list += get_nodes( left  + (right-left)/2, 
                                        right, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
        else:
            #If limit not reached, node list is still an array of 4
            return node_list[0]

            
        
        remote_instance.logger.info("Done (%i nodes)" % len(node_list) )

        return node_list

    incursion = 1
    node_list = get_nodes( left, right, top, bottom, z1, z2, remote_instance, incursion , project_id )

    #Collapse list into unique skeleton ids
    skeletons = set()
    for node in node_list:                       
        skeletons.add(node[7])           

    return list(skeletons)

def get_user_list( remote_instance = None ):
    """ Get list of users for given CATMAID server (not project specific)

    Parameters:
    ----------    
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'

    Returns:
    ------
    Pandas dataframe:
        
        id   login   full_name   first_name   last_name   color
    0
    1
    ..

    Use user_list.set_index('id').T.to_dict() to (1.) set user ID as index, 
    (2.) transpose the dataframe and (3.) turn it into a dict { user_id: { } }
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return 

    user_list = remote_instance.fetch ( remote_instance.get_user_list_url() )

    columns = [ 'id', 'login', 'full_name', 'first_name', 'last_name', 'color' ]

    df = pd.DataFrame(  [ [ e[c] for c in columns ] for e in user_list ],
                        columns = columns
                        )

    df.sort_values( ['login'], inplace=True)
    df.reset_index(inplace=True)

    return df

def get_volume( volume_name, remote_instance = None, project_id = 1 ):
    """ Retrieves volume (mesh) from Catmaid server and converts to set of 
    vertices and faces.

    Parameters:
    ----------
    volume_name :       string
                        name of the volume to import - must be EXACT!
    remote_instance :   CATMAID instance; either pass directly to function or 
                        define globally as 'remote_instance'

    Returns:
    -------
    vertices :          [ (x,y,z), (x,y,z), .... ]
    faces :             [ ( vertex_index, vertex_index, vertex_incex ), ... ]
    
    """   

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return 

    remote_instance.logger.info('Retrieving volume <%s>' % volume_name)

    #First, get volume ID
    get_volumes_url = remote_instance.get_volumes( project_id )
    response =  remote_instance.fetch ( get_volumes_url )      

    volume_id  = [ e['id'] for e in response if e['name'] == volume_name ]

    if not volume_id:
        remote_instance.logger.error('Did not find a matching volume for name %s' % volume_name)
        return
    else:
        volume_id = volume_id[0]

    #Now download volume
    url = remote_instance.get_volume_details( project_id, volume_id )
    response = remote_instance.fetch(url)

    mesh_string = response['mesh']
    mesh_name = response['name']

    mesh_type = re.search('<(.*?) ', mesh_string).group(1)

    #Now reverse engineer the mesh
    if mesh_type  == 'IndexedTriangleSet':            
        t = re.search("index='(.*?)'", mesh_string).group(1).split(' ')
        faces = [ ( int( t[i] ), int( t[i+1] ), int( t[i+2] ) ) for i in range( 0, len(t) - 2 , 3 ) ]

        v = re.search("point='(.*?)'", mesh_string).group(1).split(' ')
        vertices = [ ( float( v[i] ), float( v[i+1] ), float( v[i+2] ) ) for i in range( 0,  len(v) - 2 , 3 ) ]

    elif mesh_type  == 'IndexedFaceSet':
        #For this type, each face is indexed and an index of -1 indicates the end of this face set
        t = re.search("coordIndex='(.*?)'", mesh_string).group(1).split(' ')
        faces = []
        this_face = []
        for f in t:
            if int(f) != -1:
                this_face.append( int(f) )
            else:
                faces.append( this_face )
                this_face = []

        #Make sure the last face is also appended
        faces.append( this_face )

        v = re.search("point='(.*?)'", mesh_string).group(1).split(' ')
        vertices = [ ( float( v[i] ), float( v[i+1] ), float( v[i+2] ) ) for i in range( 0,  len(v) - 2 , 3 ) ]

    else:
        remote_instance.logger.error("Unknown volume type: %s" % mesh_type)        
        return

    #For some reason, in this format vertices occur multiple times - we have to collapse that to get a clean mesh
    final_faces = []
    final_vertices = []

    for t in faces:
        this_faces = []
        for v in t:
            if vertices[v] not in final_vertices:
                final_vertices.append( vertices[v] )
                
            this_faces.append( final_vertices.index( vertices[v] ) )

        final_faces.append( this_faces )

    remote_instance.logger.info('Volume type: %s' % mesh_type)
    remote_instance.logger.info('# of vertices after clean-up: %i' % len(final_vertices) )
    remote_instance.logger.info('# of faces after clean-up: %i' % len(final_faces) )    

    return final_vertices, final_faces

        
if __name__ == '__main__':
    """ Code below provides some examples for using this library and will only be executed if the file is executed directly instead of imported.
    """

    #First, create CATMAID instance. Here, a separate file (connect_catmaid) holding my credentials is called but you can easily do this yourself by using: 
    remote_instance = CatmaidInstance( 'server_url', 'http_user', 'http_pw', 'user_token' )    

    #Some example skids and annotation
    example_skids = [  '298953','1085816','1159799' ]
    example_annotation = 'glomerulus DA1'

    #Retrieve annotations for a list of neurons
    print(get_annotations_from_list (example_skids, remote_instance))

    #Retrieve names of neurons
    print(get_names (example_skids , remote_instance))    

    #Retrieve skeleton ids that have a given annotation
    print( get_skids_by_annotation( example_annotation , remote_instance ) )
    
    #Get CATMAID version running on your server
    print( remote_instance.fetch ( remote_instance.djangourl('/version') ) )

    #Retrieve user history
    print( get_history( remote_instance = remote_instance, project_id = 1, start_date = '2016-10-29', end_date = '2016-11-08', split = True ) )

    #Get review status
    print( get_review( example_skids ,remote_instance ) )

    #Get contribution stats for a list of neurons
    print( get_contributor_statistics( example_skids ,remote_instance ) )        

    #Get connections between neurons
    print(get_edges (example_skids, remote_instance))

    #Get neurons in a given volume (Warning: this will take a while!)
    print( get_neurons_in_volume ( 0, 28218, 21000, 28128, 6050, 39000, remote_instance ))

    #Retrieve synaptic partners for a set of neurons - ignore those connected by less than 3 synapses
    print ( get_partners (example_skids, remote_instance, threshold = 3))

    #Get 3D skeletons and print the first one
    print( get_3D_skeleton ( example_skids , remote_instance, 1 , 0 )[0] )
    
    #Get list of users
    print( remote_instance.fetch ( remote_instance.get_user_list_url() ) )

    #Get list of skeletons created by user 93 between 1/1/2016 and 1/10/2016. If you only provide the remote_instance, all neurons are returned
    print( get_skeleton_list(remote_instance, user=93 , node_count=1, start_date= [2016,1,1], end_date = [2016,10,1] ) )

    #Add annotations to neurons - be extremely careful with this!
    #add_annotations ( example_skids , ['test'], remote_instance )
