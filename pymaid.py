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

from pymaid import CatmaidInstance, get_3D_skeleton

myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )
skeleton_data = get_3D_skeleton ( ['12345','67890'] , myInstance )

Additional examples:
-------------------

Please open pymaid.py and check out if __name__ == '__main__' for additional examples

Contents:
-------

CatmaidInstance :   Class holding credentials and server url 
                    Use to generate urls and fetch data

A long list of wrappers to retrieve various data from the CATMAID server.
Use dir(pymaid) to get a list and help(wrapper) to learn about its function.

"""

import urllib
import json
import http.cookiejar as cj
import time
import base64
import threading
import time

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

    Methods
    -------
    fetch ( url, post = None )
        retrieves information from CATMAID server
    get_XXX_url ( pid , sid , X )
        <CatmaidInstance> class contains a long list of function that generate URLs to request data from the CATMAID server.
        Use dir(<CatmaidInstance>) to get the full list. Most of these functions require a project id (pid) and a stack id (sid)
        some require additional parameters, for example a skeleton id (skid). 

    Examples:
    --------
    # 1.) Fetch skeleton data for a single neuron

    from pymaid import CatmaidInstance

    myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )
    project_id = 1
    skeleton_id = 12345
    3d_skeleton_url = myInstance.get_compact_skeleton_url( project_id , skeleton_id )
    skeleton_data = myInstance.fetch( 3d_skeleton_url )

    # 2.) Use wrapper functions to fetch multiple skids

    from pymaid import CatmaidInstance, get_3D_skeleton

    myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )
    skeleton_data = get_3D_skeleton ( ['12345','67890'] , myInstance )
    
    """

    def __init__(self, server, authname, authpassword, authtoken):
        self.server = server
        self.authname = authname
        self.authpassword = authpassword
        self.authtoken = authtoken
        self.opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())    

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
            request = urllib.request.Request(url, data = data)        
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

    def get_completed_connector_links(self,pid):
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

    
    def get_single_neuronname(self, pid, skid):
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

    def get_annotations_for_skid_list(self, pid):
        """ ATTENTION: This does not seem to work anymore as of 20/10/2015 -> although it still exists in CATMAID code
            use get_annotations_for_skid_list2    
            Use to get annotations for given neuron. DOES need skid as postdata
        """
        return self.djangourl("/" + str(pid) + "/annotations/skeletons/list" )      

    def get_review_details_url ( self, pid,  skid):
        """ Use to retrieve review status for every single node of a skeleton.      
        For some reason this needs to fetched as POST (even though actual POST data is not necessary)
        Returns list of arbors, the nodes the contain and who has been reviewing them at what time  
        """ 
        return self.djangourl("/" + str(pid) + "/skeletons/" + str(skid) + "/review" )

    
    def get_annotations_for_skid_list2(self, pid):
        """ Use to get annotations for given neuron. DOES need skid as postdata. """
        return self.djangourl("/" + str(pid) + "/skeleton/annotationlist" )
    
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
        """ #Use to parse url for retrieving annotated neurons (does need post data). """        
        return self.djangourl("/" + str(pid) + "/annotations/query-targets" )
    
    def get_node_list(self, pid):
        """ Use to parse url for retrieving list of nodes (needs post data). """
        return self.djangourl("/" + str(pid) + "/node/list" )
    
    def get_node_info(self, pid):
        """ Use to parse url for retrieving user info on a single node (needs post data). """
        return self.djangourl("/" + str(pid) + "/node/user-info" )        
    
    def get_compact_skeleton_url(self, pid, skid, connector_flag = 1, tag_flag = 1):        
        """ Use to parse url for retrieving all info the 3D viewer gets (does NOT need post data).
        Returns, in JSON, [[nodes], [connectors], [tags]], with connectors and tags being empty when 0 == with_connectors and 0 == with_tags, respectively
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
    
    def get_skeletons_from_neuron_id(self,neuron_id,pid):
        """ Use to get all skeletons of a given neuron (neuron_id). """
        return self.djangourl("/" + str(pid) + "/neuron/" + str(neuron_id) + '/get-all-skeletons' )
    
    def get_history_url(self, pid):
        """ Use to get user history. """
        return self.djangourl("/" + str(pid) + "/stats/user-history" )
  

def retrieve_urls_threaded( urls , remote_instance, post_data = [], time_out = None , silent = False ):
    """ Wrapper to retrieve a list of urls using threads

    Parameters:
    ----------
    urls :              list of strings
                        Urls to retrieve
    remote_instance :   CATMAID instance
                        Either pass directly to function or define globally as 'remote_instance'       
    post_data :         list of dicts
                        needs to be the same size as urls
    time_out :          integer or None
                        After this number of second, fetching data will time out (so as to not block the system)
                        If set to None, time out will be max( [ 20, len(urls) ] ) - e.g. 100s for 100 skeletons but at least 20s

    Returns:
    -------
    data :              data retrieved for each url -> order is kept!
    """

    data = [ None for u in urls ]
    threads = {}
    threads_closed = []   

    if time_out is None:
        time_out = max( [ len( urls ) , 20 ] ) 

    print('Creating threads to retrieve data')
    for i, url in enumerate(urls):
        if post_data:
            t = retrieveUrlThreaded ( url, remote_instance, post_data = post_data[i] )
        else:
            t = retrieveUrlThreaded ( url, remote_instance )
        t.start()
        threads[ str(i) ] = t
        if not silent:
            print('\r Threads: '+str( len ( threads ) ) ,end='')  

    print('\n Joining threads...') 

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
        if not silent:
            print('\r Closing Threads: %i ( %is until time out )' % ( len( threads_closed ) , round( time_out - ( cur_time - start ) ) )  ,end='')             

    if cur_time > (start + time_out):        
        print('\n !WARNING: Timeout while joining threads. Retrieved only %i of %i urls' % (len( [ d for d in data if d != None ] ),len(threads)))  
        for t in threads:
            if t not in threads_closed:
                print('Did not close thread for url:', urls[ int( t ) ] )      
    else:
        print('\n Success! %i of %i urls retrieved.' % ( len(threads_closed) , len( urls ) ) )
    
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
            print('!Error initiating thread for',self.url)

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
            print('!ERROR joining thread for',self.url)
            return None

def get_3D_skeleton ( skids, remote_instance = None , connector_flag = 1, tag_flag = 1, get_history = False, time_out = None, silent = False, project_id = 1):
    """ Wrapper to retrieve the skeleton data for a list of skeleton ids

    Parameters:
    ----------
    skids :             single or list of skeleton ids
    remote_instance :   CATMAID instance
                        Either pass directly to function or define globally as 'remote_instance'
    connector_flag :    set if connector data should be retrieved. Possible values = 0/False or 1/True
    tag_flag :          set if tags should be retrieved. Possible values = 0/False or 1/True
    silent :            boolean
                        If True, details of the retrieval will not be printed to the consol. Useful to not crowd terminal if it does not allow cartridge return!         
    time_out :          integer or None
                        After this number of second, fetching skeleton data will time out (so as to not block the system)
                        If set to None, time out will be max( [ 20, len(skids) ] ) - e.g. 100s for 100 skeletons but at least 20s
    get_history:        boolean
                        if True, the returned skeleton data will contain creation date n[8] and last modified n[9] for each node -> compact-details url the 'with_history' option is used in this case
                        ATTENTION: if get_history = True, nodes/connectors that have been moved since their creation will have multiple entries reflecting their changes in position! 
                        Each state has the date it was modified as creation date and the next state's date as last modified. The most up to date state has the original creation date as last modified (full circle).
                        The creator_id is always the original creator though.

    Returns:
    -------
    list of 3D skeleton data in the same order as the list of skids passed as parameter: 
        [ [ neuron1_nodes, neuron1_connectors, neuron1_tags ], [ neuron2_nodes, ... ], [... ], ... ]
    

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    if type(skids) != type(list()):
        to_retrieve = [skids]
    else:
        to_retrieve = skids

    #Generate URLs to retrieve
    urls = []    
    for i, skeleton_id in enumerate(to_retrieve):
        if get_history is False:
            #Convert tag_flag and connector_tag to 0 or 1 if necessary
            if type(tag_flag) != type( int() ):
                tag_flag = int(tag_flag)
            if type(connector_flag) != type( int() ):
                connector_flag = int(connector_flag)

            #Create URL for retrieving skeleton data from server
            urls.append ( remote_instance.get_compact_skeleton_url( project_id , skeleton_id, connector_flag, tag_flag ) )
        else:
            #Convert tag_flag and connector_tag to boolean if necessary
            if type(tag_flag) != type( bool() ):
                tag_flag = tag_flag == 1
            if type(connector_flag) != type( bool() ) :
                connector_flag = connector_flag == 1

            #Create URL for retrieving skeleton data from server with history details
            remote_compact_skeleton_url = remote_instance.get_compact_details_url( project_id , skeleton_id )
            #For compact-details, parameters have to passed as GET 
            remote_compact_skeleton_url += '?%s' % urllib.parse.urlencode( {'with_history': True , 'with_tags' : tag_flag , 'with_connectors' : connector_flag  , 'with_merge_history': False } )
            #'True'/'False' needs to be lower case
            urls.append (  remote_compact_skeleton_url.lower() )

    skdata = retrieve_urls_threaded( urls, remote_instance, time_out = time_out, silent = silent )

    return skdata     

def get_arbor ( skids, remote_instance = None, node_flag = 1, connector_flag = 1, tag_flag = 1, project_id = 1 ):
    """ Wrapper to retrieve the skeleton data for a list of skeleton ids including detailed connector data. See get_compact_arbor_url.

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'
    connector_flag :    set if connector data should be retrieved. Values = 0 or 1. (optional, default = 1)
    tag_flag :          set if tags should be retrieved. Values = 0 or 1. (optional, default = 1)

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    sk_data = []

    for skeleton_id in skids:
        #Create URL for retrieving example skeleton from server
        remote_compact_arbor_url = remote_instance.get_compact_arbor_url( project_id , skeleton_id, node_flag, connector_flag, tag_flag )

        #Retrieve node_data for example skeleton
        arbor_data = remote_instance.fetch( remote_compact_arbor_url )

        sk_data.append(arbor_data)

        print('%s retrieved' % str(skeleton_id))

    return (sk_data)

def retrieve_partners (skids, remote_instance = None , threshold = 1, project_id = 1):
    """ Wrapper to retrieve the synaptic partners to neurons of interest

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'
    threshold :         does not seem to have any effect on CATMAID API and is therefore filtered afterwards. This threshold is applied to the total number of synapses. (optional, default = 1)

    Returns:
    ------- 
    filtered connectivity: {'incoming': { skid1: { 'num_nodes': XXXX, 'skids':{ 'skid3':n_snypases, 'skid4': n_synapses } } , skid2:{}, ... }, 'outgoing': { } }

    """

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

    connectivity_data = remote_instance.fetch( remote_connectivity_url , connectivity_post )

    #As of 08/2015, # of synapses is returned as list of nodes with 0-5 confidence: {'skid': [0,1,2,3,4,5]}
    #This is being collapsed into a single value before returning it:       

    for direction in ['incoming','outgoing']:
        pop = []
        for entry in connectivity_data[direction]:
            if sum( [ sum(connectivity_data[direction][entry]['skids'][n]) for n in connectivity_data[direction][entry]['skids'] ] ) >= threshold:
                for skid in connectivity_data[direction][entry]['skids']:                
                    connectivity_data[direction][entry]['skids'][skid] = sum(connectivity_data[direction][entry]['skids'][skid])
            else:
                pop.append(entry)

        for n in pop:
            connectivity_data[direction].pop(n)

        
    return(connectivity_data)
    

def retrieve_names (skids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve neurons names for a list of skeleton ids

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns:
    ------- 
    dictionary with names: { skid1 : neuron_name, skid2 : neuron_name,  .. }

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_names_url = remote_instance.get_neuronnames( project_id )

    get_names_postdata = {}
    get_names_postdata['pid'] = 1
        
    for i in range(len(skids)):
        key = 'skids[%i]' % i
        get_names_postdata[key] = skids[i]

    names = remote_instance.fetch( remote_get_names_url , get_names_postdata )
        
    return(names)

def retrieve_node_lists (skids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve treenode table for a list of skids

    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns:
    ------- 
    dictionary with list of nodes: { skid1 : [ [ node_id, parent_node_id, confidence, x, y, z, radius, creator, last_edition_timestamp ],[],... ], skid2 :  .. }    

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    print('Retrieving %i node tables...' % len(skids))   

    nodes = {}

    run = 1
    for skid in skids:

        remote_nodes_list_url = remote_instance.get_skeleton_nodes_url( project_id , skid )

        print('Retrieving node table of %s [%i of %i]...' % (str(skid),run,len(skids)), end = ' ')        

        try:
            node_list = remote_instance.fetch( remote_nodes_list_url )
        except:
            print('Time out on first try')
            time.sleep(.5)
            try:
                node_list = remote_instance.fetch( remote_nodes_list_url )
                print('Success on second attempt')
            except:
                print('Unable to retrieve')

        #Data format: node_id, parent_node_id, confidence, x, y, z, radius, creator, last_edition_timestamp

        nodes[skid] = node_list
        print('Done')

        run += 1    

    return nodes

def get_edges (skids, remote_instance = None):
    """ Wrapper to retrieve edges (synaptic connections) between sets of neurons
    
    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns list of edges: [source_skid, target_skid, weight]

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
        
    return(edges)

def get_connectors ( skids, remote_instance = None, incoming = True, outgoing = True, project_id = 1):
    """ Wrapper to retrieve connectors for a set of neurons
    
    Parameters:
    ----------
    connector_ids :     list of connector ids; can be found e.g. from compact skeletons
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'
    incoming :          boolean
                        if True, incoming connectors will be retrieved
    outgoing :          boolean
                        if True, outgoing connectors will be retrieved

    Returns:
    ------- 
    list of connectors:  [skeleton_id, connector_id, x,y,z, S(?), confidence, creator, treenode_id, creation_date ] 
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return    

    get_connectors_GET_data = { 'with_tags': 'false' }

    cn_data = []

    for i,s in enumerate(skids):
        tag = 'skeleton_ids[%i]' % i 
        get_connectors_GET_data[tag] = str( s )

    if incoming is True:
        get_connectors_GET_data['relation_type']='presynaptic_to'
        remote_get_connectors_url = remote_instance.get_connectors_url( project_id ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
        cn_data += remote_instance.fetch( remote_get_connectors_url )['links']

    if outgoing is True:
        get_connectors_GET_data['relation_type']='postsynaptic_to'
        remote_get_connectors_url = remote_instance.get_connectors_url( project_id ) + '?%s' % urllib.parse.urlencode(get_connectors_GET_data)
        cn_data += remote_instance.fetch( remote_get_connectors_url )['links']

    #Make sure we don't count the same connector twice
    clean_cn_data = []
    for cn in cn_data:
        if cn not in clean_cn_data:
            clean_cn_data.append(cn)

    return clean_cn_data


def get_connector_details (connector_ids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve details on sets of connectors 
    
    Parameters:
    ----------
    connector_ids :     list of connector ids; can be found e.g. from compact skeletons
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns:
    ------- 
    list of connectors: [ connector_id, {'presynaptic_to': skid, 'postsynaptic_to': [skid,skid,..]} ]

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_connectors_url = remote_instance.get_connector_details_url( project_id )

    get_connectors_postdata = {}    
        
    for i in range(len(connector_ids)):
        key = 'connector_ids[%i]' % i
        get_connectors_postdata[key] = connector_ids[i]

    connectors = remote_instance.fetch( remote_get_connectors_url , get_connectors_postdata )    

    print('Data for %i of %i connectors retrieved' %(len(connectors),len(connector_ids)))
        
    return(connectors)

def get_review (skids, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve review status for a set of neurons
    
    Parameters:
    ----------
    skids :             list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns:
    ------- 
    dict for skids: { skid : [node_count, nodes_reviewed], ... }

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
        
    return(review_status)
   
def get_neuron_annotation (skid, remote_instance = None, project_id = 1 ):
    """ Wrapper to retrieve annotations of a SINGLE neuron
    
    Parameters:
    ----------
    skid :              string or int 
                        Single skeleton id.
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns:
    ------- 
    dict of annotations: {'aaData': [['annotation', time_stamp, unknown_number , user_id , annotation_id], 'iTotalRecords': 15, 'iTotalDisplayRecords': 15}

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    #This works with neuron_id NOT skeleton_id
    #neuron_id can be requested via neuron_names
    remote_get_neuron_name = remote_instance.get_single_neuronname( project_id , skid )
    neuronid = remote_instance.fetch( remote_get_neuron_name )['neuronid']

    remote_get_annotations_url = remote_instance.get_neuron_annotations( project_id )

    get_annotations_postdata = {}            
    get_annotations_postdata['neuron_id'] = int(neuronid)
   

    annotations = remote_instance.fetch( remote_get_annotations_url , get_annotations_postdata )    
        
    return(annotations) 



def skid_exists( skid, remote_instance = None, project_id = 1 ):
    """ Quick function to check if skeleton id exists
    
    Parameters:
    skid - single skeleton id
    remote_instance - CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns True if skeleton exists, False if not

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    remote_get_neuron_name = remote_instance.get_single_neuronname( project_id , skid )
    response = remote_instance.fetch( remote_get_neuron_name )
    
    if 'error' in response:
        return False
    else:
        return True  

def retrieve_annotation_id( annotation, remote_instance = None, project_id = 1 ):
    """ Wrapper to retrieve the annotation ID for single or list of annotation(s)
    
    Parameters:
    ----------
    annotation :        single annotations or list of multiple annotations
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    print('Retrieving list of annotations...')

    remote_annotation_list_url = remote_instance.get_annotation_list( project_id )
    annotation_list = remote_instance.fetch( remote_annotation_list_url )
         
    
    if type(annotation) == type(str()):
        for dict in annotation_list['annotations']:
            #print(dict['name'])
            if dict['name'] == annotation:
                annotation_id = dict['id']
                print('Found matching annotation!')
                break
            else:
                annotation_id = 'not found'
    
        return([annotation_id])  

    elif type(annotation) == type(list()):
        annotation_ids = []
        for dict in annotation_list['annotations']:            
            if dict['name'] in annotation:
                annotation_ids.append( dict['id'] )
                annotation.remove( dict['name'] )  
    
        if len(annotation) != 0:
            print('Could not retrieve annotation id for:', annotation)

        return(annotation_ids)  

def retrieve_skids_by_name(tag, allow_partial = True, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve the all neurons with matching name
    
    Parameters:
    ----------
    tag :               name to search for
    allow_partial :     if True, partial matches are returned too    
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'
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
            match += e['skeleton_ids']
        if not allow_partial and e['type'] == 'neuron' and e['name'] == tag:
            match += e['skeleton_ids']

    return list( set(match) )

def retrieve_skids_by_annotation(annotation, remote_instance = None, project_id = 1 ):
    """ Wrapper to retrieve the all neurons annotated with given annotation(s)
    
    Parameters:
    ----------
    annotation :        single annotations or list of multiple annotations    
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    print('Looking for Annotation(s):',annotation) 
    annotation_ids = retrieve_annotation_id(annotation, remote_instance)

    if type(annotation) == type(list()):
        print('Found id(s): %s | Unable to retrieve: %i' % ( str(annotation_ids) , len(annotation)-len(annotation_ids) ))  
    elif type(annotation) == type( str() ):
        print('Found id: %s | Unable to retrieve: %i' % ( str(annotation_ids[0]) , 1 - len(annotation_ids) ))

    annotated_skids = []
    print('Retrieving skids of annotated neurons...')
    for an_id in annotation_ids:
        #annotation_post = {'neuron_query_by_annotation': annotation_id, 'display_start': 0, 'display_length':500}
        annotation_post = {'annotated_with0': an_id, 'rangey_start': 0, 'range_length':500, 'with_annotations':False}
        remote_annotated_url = remote_instance.get_annotated_url( project_id )
        neuron_list = remote_instance.fetch( remote_annotated_url, annotation_post )
        count = 0    
        for entry in neuron_list['entities']:
            if entry['type'] == 'neuron':
                annotated_skids.append(str(entry['skeleton_ids'][0]))    

        
    return(annotated_skids)

def get_annotations_from_list (skid_list, remote_instance = None, project_id = 1 ):
    """ Wrapper to retrieve annotations for a list of skeleton ids - if a neuron has no annotations, it will not show up in returned dict
    Attention! It seems like this URL does not process more than 250 skids at a time!

    Parameters:
    ----------
    skid_list :         list of skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'
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

def add_annotations ( skid_list, annotations, remote_instance = None, project_id = 1 ):
    """ Wrapper to add annotation(s) to a list of neuron(s)

    Parameters:
    ----------
    skid_list :         list of skeleton ids that will be annotated
    annotations :       list of annotation(s) to add to provided skeleton ids
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

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

    print( remote_instance.fetch( add_annotations_url , add_annotations_postdata ) )

    return 

def get_review_details ( skid_list, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve review status (reviewer + timestamp) for each node of a given skeleton -> uses the review API

    Parameters:
    -----------
    skid_list :         list of skeleton ids to check
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns:
    -------
    node_list :         list of reviewed nodes: [ node_id, [ [reviewer1, timestamp] ,[ reviewer2, timestamp] ] ]
    """

    node_list = []    
    urls = []
    post_data = []

    for skid in skid_list:
        urls.append ( remote_instance.get_review_details_url( project_id , skid ) )
        #For some reason this needs to fetched as POST (even though actual POST data is not necessary)
        post_data.append ( { 'placeholder' : 0 } )

    rdata = retrieve_urls_threaded( urls , remote_instance, post_data = post_data, time_out = None , silent = False )           

    for neuron in rdata:
        #There is a small chance that nodes are counted twice but not tracking node_id speeds up this extraction a LOT
        #node_ids = []
        for arbor in neuron:            
            node_list += [ ( n['id'] , n['rids'] ) for n in arbor['sequence'] if n['rids'] ]

    return node_list
            

def get_contributor_statistics (skid_list, remote_instance = None, project_id = 1):
    """ Wrapper to retrieve contributor statistics over ALL given skeleton ids
    
    Parameters:
    ----------
    skid_list :         list of skeleton ids to check
    remote_instance :   CATMAID instance; either pass directly to function or define globally as 'remote_instance'

    Returns:
    -------
    dictionary {'node_contributors': {'user_id': nodes_contributed , ...}, 'multiuser_review_minutes': XXX , 'post_contributors': {'user_id': postsynaptic_connectors_contributed}, 'construction_minutes': XXX, 'min_review_minutes': XXX, 'n_post': XXX, 'n_pre': XXX, 'pre_contributors': {'user_id': presynaptic_connectors_contributed, ...}, 'n_nodes': XXX}
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    get_statistics_postdata = {}    
        
    for i in range(len(skid_list)):
        key = 'skids[%i]' % i
        get_statistics_postdata[key] = skid_list[i]
    
    
    remote_get_statistics_url = remote_instance.get_contributions_url( project_id )
    statistics = remote_instance.fetch( remote_get_statistics_url, get_statistics_postdata )

    return(statistics)

def retrieve_skeleton_list( remote_instance = None , user=None, node_count=1, start_date=[], end_date=[], reviewed_by = None, project_id = 1 ):
    """ Wrapper to retrieves a list of all skeletons that fit given parameters (see variables). If no parameters are provided, all existing skeletons are returned.

    Parameters:
    ----------
    remote_instance :   class
                        Your CATMAID instance; either pass directly to function or define globally as 'remote_instance'.
    user :              integer
                        A single user_id.
    node_count :        integer
                        Minimum number of nodes.
    start_date :        list of integers [year, month, day]
                        Only consider neurons created after.
    end_date :          list of integers [year, month, day]
                        Only consider neurons created before.
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

def retrieve_history( remote_instance = None, project_id = 1, start_date = '2016-10-29', end_date = '2016-11-08'):    
    """ Wrapper to retrieves CATMAID history 

    Parameters:
    ----------
    remote_instance - CATMAID instance; either pass directly to function or define globally as 'remote_instance'
    user - single user_id
    node_count - minimum number of nodes
    start_date - created after, date needs to be (year,month,day) format
    end_date - created before, date needs to be (year,month,day) format
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

    get_history_GET_data = {    'pid': project_id ,
                                'start_date': start_date,
                                'end_date': end_date
                                    }

    remote_get_history_url = remote_instance.get_history_url( project_id )

    remote_get_history_url += '?%s' % urllib.parse.urlencode(get_history_GET_data)

    print('Retrieving user history: ',remote_get_history_url)   

    return remote_instance.fetch ( remote_get_history_url )


def get_neurons_in_volume ( left, right, top, bottom, z1, z2, remote_instance = None, project_id = 1 ):
    """ Retrieves neurons with processes within a defined volume. Because the API returns only a limited number of neurons at a time, the defined volume has to be chopped into smaller pieces for crowded areas - may thus take some time!

    Parameters:
    ----------
    left, right, top, z1, z2 :  Coordinates defining the volumes. Need to be in nm, not pixels.
    remote_instance :           CATMAID instance; either pass directly to function or define globally as 'remote_instance'
    
    """   

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return 

    def retrieve_nodes( left, right, top, bottom, z1, z2, remote_instance, incursion, project_id ):  

        print(incursion,':',left, right, top, bottom, z1, z2)      

        remote_nodes_list = remote_instance.get_node_list ( project_id )

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
            print('Incursing')   
            incursion += 1         
            node_list = list()
            #Front left top
            node_list += retrieve_nodes( left, 
                                        left + (right-left)/2, 
                                        top, 
                                        top + (bottom-top)/2, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Front right top
            node_list += retrieve_nodes( left  + (right-left)/2, 
                                        right, 
                                        top,
                                        top + (bottom-top)/2, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Front left bottom            
            node_list += retrieve_nodes( left, 
                                        left + (right-left)/2, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Front right bottom
            node_list += retrieve_nodes( left  + (right-left)/2, 
                                        right, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1, 
                                        z1 + (z2-z1)/2, 
                                        remote_instance, incursion )
            #Back left top
            node_list += retrieve_nodes( left, 
                                        left + (right-left)/2, 
                                        top, 
                                        top + (bottom-top)/2, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
            #Back right top
            node_list += retrieve_nodes( left  + (right-left)/2, 
                                        right, 
                                        top,
                                        top + (bottom-top)/2, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
            #Back left bottom            
            node_list += retrieve_nodes( left, 
                                        left + (right-left)/2, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
            #Back right bottom
            node_list += retrieve_nodes( left  + (right-left)/2, 
                                        right, 
                                        top + (bottom-top)/2, 
                                        bottom, 
                                        z1 + (z2-z1)/2, 
                                        z2, 
                                        remote_instance, incursion )
        else:
            #If limit not reached, node list is still an array of 4
            return node_list[0]

            
        
        print("Done.",len(node_list))

        return node_list

    incursion = 1
    node_list = retrieve_nodes( left, right, top, bottom, z1, z2, remote_instance, incursion , project_id )

    #Collapse list into unique skeleton ids
    skeletons = set()
    for node in node_list:                       
        skeletons.add(node[7])           

    return list(skeletons)
        
if __name__ == '__main__':
    """ Code below provides some examples for using this library and will only be executed if the file is executed directly instead of imported.
    """

    #First, create CATMAID instance. Here, a separate function that holds my credentials is called but you can easily do this yourself by using: remote_instance = CatmaidInstance( server_url, http_user, http_pw, user_token )    
    from connect_catmaid import connect_larval_em, connect_adult_em
    remote_instance = connect_adult_em()

    #Some example skids and annotation
    example_skids = [  '298953','1085816','1159799' ]
    example_annotation = 'glomerulus DA1'

    #Retrieve annotations for a list of neurons
    print(get_annotations_from_list (example_skids, remote_instance))

    #Retrieve names of neurons
    print(retrieve_names (example_skids , remote_instance))    

    #Retrieve skeleton ids that have a given annotation
    print( retrieve_skids_by_annotation( example_annotation , remote_instance ) )
    
    #Get CATMAID version running on your server
    print( remote_instance.fetch ( remote_instance.djangourl('/version') ) )

    #Retrieve user history
    print( retrieve_history( ) )

    #Get review status
    print( get_review( example_skids ,remote_instance ) )

    #Get contribution stats for a list of neurons
    print( get_contributor_statistics( example_skids ,remote_instance ) )        

    #Get connections between neurons
    print(get_edges (example_skids, remote_instance))

    #Get neurons in a given volume (Warning: this will take a while!)
    print( get_neurons_in_volume ( 0, 28218, 21000, 28128, 6050, 39000, remote_instance ))

    #Retrieve synaptic partners for a set of neurons - ignore those connected by less than 3 synapses
    print ( retrieve_partners (example_skids, remote_instance, threshold = 3))

    #Get 3D skeletons and print the first one
    print( get_3D_skeleton ( example_skids , remote_instance, 1 , 0 )[0] )
    
    #Get list of users
    print( remote_instance.fetch ( remote_instance.get_user_list_url() ) )

    #Get list of skeletons created by user 93 between 1/1/2016 and 1/10/2016. If you only provide the remote_instance, all neurons are returned
    print( retrieve_skeleton_list(remote_instance, user=93 , node_count=1, start_date= [2016,1,1], end_date = [2016,10,1] ) )

    #Add annotations to neurons - be extremely careful with this!
    #add_annotations ( example_skids , ['test'], remote_instance )
