""" 
A collection of tools to interace with CATMAID R libraries (e.g. nat, catnat, elmr, rcatmaid)
    
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
-------------
from pymaid import pymaid
from pymaid import rmaid

#Initialise Catmaid instance
rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')

#Fetch a neuron in CATMAID
skid = 123456
n = pymaid.get_3D_skeleton( skid, rm )

#Initialize R's rcatmaid 
rcatmaid = rmaid.init_rcatmaid( rm )

#Convert pymaid neuron to R neuron (works with neuron and neuronlist objects)
n_r = rmaid.neuron2r( n.ix[0] )

#Use some nat function
n_pruned = cat.prune_by_strahler( n_r )

#Convert back to pymaid object
n_py = rmaid.neuron2py( n_pruned, rm )

"""

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

cl = robjects.r('class')
names = robjects.r('names')

import pandas as pd

try:
   from pymaid import get_names
except:
   from pymaid.pymaid import get_names

import logging

#Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if not module_logger.handlers:
   #Generate stream handler
   sh = logging.StreamHandler()
   sh.setLevel(logging.INFO)
   #Create formatter and add it to the handlers
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   sh.setFormatter(formatter)
   module_logger.addHandler(sh)


def init_rcatmaid ( remote_instance ):
	""" This function initializes the R catmaid package from Jefferis 
	(https://github.com/jefferis/rcatmaid) and returns an instance of it

	Parameters:
	----------
	remote_instance :		CATMAID instance from pymaid.pymaid.CatmaidInstance()
								is used to extract credentials

	Returns:
	------- 
	catmaid :				robject containing Catmaid library

	"""

	#Import R Catmaid
	catmaid = importr('catmaid')

	#Use remote_instance's credentials
	catmaid.server = remote_instance.server
	catmaid.authname = remote_instance.authname
	catmaid.authpassword = remote_instance.authpassword
	catmaid.token = remote_instance.authtoken

	#Create the connection
	con = catmaid.catmaid_connection( server = catmaid.server, authname = catmaid.authname, authpassword = catmaid.authpassword, token = catmaid.token  )

	#Login
	catmaid.catmaid_login( con )

	module_logger.info('Rcatmaid successfully initiated.')

	return catmaid

def data2py ( data ):
	""" Takes data object from rcatmaid (e.g. 'catmaidneuron' from read.neuron.catmaid)
	and converts into Python Data. 

	Please note that:
	(1) Most R data comes as list (even if only 1 entry). This is preserved.
	(2) R lists with headers are converted to dictionaries
	(3) R DataFrames are converted to Pandas DataFrames


	Parameters:
	----------
	data :				any kind of R data 
						Can be nested (e.g. list of lists)!

	Returns:
	-------
	converted data
	"""	

	if cl(data)[0] == 'neuronlist':				
		neuron_list = pd.DataFrame( data = [ [ data2py(e) for e in neuron ] for neuron in data ] )		
		neuron_list.columns =  data.names #[ 'NumPoints', 'StartPoint','BranchPoints','EndPoints','nTrees', 'NumSeqs', 'SegList', 'd', 'skid', 'connectors', 'tags','url', 'headers'  ]
		neuron_list['name']	= data.slots['df'][2]
		return neuron_list

	elif cl(data)[0] == 'catmaidneuron' or cl(data)[0] == 'neuron':				
		neuron = pd.DataFrame( data = [ [ data2py(e) for e in data ] ]  )		
		neuron.columns = data.names  #[ 'NumPoints', 'StartPoint','BranchPoints','EndPoints','nTrees', 'NumSeqs', 'SegList', 'd', 'skid', 'connectors', 'tags','url', 'headers'  ]				
		return neuron

	elif cl(data)[0] == 'integer':
		return [ int( n ) for n in  data ]
	elif cl(data)[0] == 'character':
		return [ str( n ) for n in  data ]
	elif cl(data)[0] == 'numeric':
		return [ float( n ) for n in  data ]
	elif cl(data)[0] == 'data.frame':	
		df = pandas2ri.ri2py_dataframe( data ) 		
		return df
	elif 'list' in cl(data):
		#If this is just a list, return a list
		if not names( data ):
			return [ data2py(n) for n in data ]
		#If this list has headers, return as dictionary
		else:
			return { n : data2py(data[i]) for i,n in enumerate( names(data) ) }
	elif cl(data)[0] == 'NULL':
		return None
	else:
		module_logger.error('Dont know how to convert R datatype %s' % cl(data) )
		return data

def neuron2py ( neuron, remote_instance ):
	""" Converts an rcatmaid neuron or neuronlist object to a standard Python 
	PyMaid neuron.

	ATTENTION: node creator and confidence are not included in R's neuron/neuronlist
	and will be imported as <None>

	Parameters:
	----------
	neuron :				R neuron or neuronlist
	remote_instance : 		CATMAID instance	

	Returns:
	-------
	pandas DataFrame
	"""

	if 'rpy2' in str( type(neuron) ):
		neuron = data2py( neuron )

	#Nat function may return neuron objects that have ONLY nodes - no connectors, skeleton_id, name or tags!	
	if 'skid' in neuron:
		neuron_names = get_names( [ n[0] for n in neuron.skid.tolist() ], remote_instance )  	
	else:
		module_logger.warning('Neuron has only nodes (no name, skid, connectors or tags)')

	data = []
	for i in range( neuron.shape[0] ):
		n = neuron.ix[i]
		#Note that radius is divided by 2 -> this is because in rcatmaid the original radius is doubled for some reason
		nodes = pd.DataFrame( [ [ no.PointNo, no.Parent, None, no.X, no.Y, no.Z, no.W/2, None ] for no in n.d.itertuples() ], dtype = object )
		nodes.columns = ['treenode_id','parent_id','creator_id','x','y','z','radius','confidence']
		nodes.loc[ nodes.parent_id == -1, 'parent_id' ] = None

		if 'connectors' in n:
			connectors = pd.DataFrame( [ [ cn.treenode_id, cn.connector_id, cn.prepost, cn.x, cn.y, cn.z ] for cn in n.connectors.itertuples() ], dtype = object )
			connectors.columns = ['treenode_id','connector_id','relation','x','y','z']
		else:
			connectors = 'NA'

		if 'skid' in n:
			skid = n.skid[0]
			name = neuron_names[ n.skid[0] ]
		else:
			skid = 'NA'
			name = 'NA'

		if 'tags' in n:
			tags = n.tags
		else:
			tags = 'NA' 

		data.append( [   
							name,
							skid,
							nodes,
							connectors,
							tags
							]  )

	df = pd.DataFrame( 		data = data , 
                            columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                            dtype=object
                            )

	return df

def neuron2r ( neuron ):
	""" Converts a PyMaid neuron or list of neurons (DataFrames) to the 
	corresponding neuron/neuronlist object in R.

	The way this works is essentially converting the PyMaid object back
	into what rcatmaid expects as a response from a CATMAID server, then
	we are calling the same functions as in rcatmaid's read.neuron.catmaid().

	Attention: Currently, the project ID saved as part of neuronlist objects
	is ALWAYS 1.	
	"""
	
	try:
		nat = importr('nat')
	except:
		module_logger.error('R library "nat" not found!')
		return None
	

	if type( neuron ) == pd.DataFrame:
		"""
		The way neuronlist are constructed is a bit more complicated:
		They are essentially named lists { 'neuronA' : neuronobject, ... }
		BUT they also contain a dataframe that holds a DataFrame as attribute ( attr('df') = df )
		This dataframe looks like this

				pid 	skid  	name				
		skid1
		skid2

		In rpy2, these attributes are assigned using the .slots['df'] function
		"""

		nlist = {}
		for i in range( neuron.shape[0] ):
			nlist[ neuron.ix[i].skeleton_id ] =  to_rneuron( neuron.ix[i] )

		nlist = robjects.ListVector( nlist )
		nlist.rownames = neuron.skeleton_id.tolist()

		df = robjects.DataFrame( { 	'pid': robjects.IntVector( [1] * neuron.shape[0] ),
												 'skid': robjects.IntVector( neuron.skeleton_id.tolist() ),
												 'name': robjects.StrVector( neuron.neuron_name.tolist() )
												} )
		df.rownames = neuron.skeleton_id.tolist() 
		nlist.slots['df'] = df 

		nlist.rclass = robjects.r('c("neuronlist","list")')

		return nlist

	elif type ( neuron ) == pd.Series:		
		n = neuron
		#First convert into format that rcatmaid expects as server response

		"""
		node_data = robjects.DataFrame( {  'id': robjects.IntVector( n.nodes.tnodes.reenode_id.tolist() ),
													  'parent_id': robjects.IntVector( n.parent_id.tolist() ),
													  'user_id': robjects.IntVector( n.nodes.creator_id.tolist() ),
													  'x': robjects.IntVector( nnodes..x.tolist() ),
													  'y': robjects.IntVector( nnodes..y.tolist() ),
													  'z': robjects.IntVector( nnodes..z.tolist() ),
													  'radius': robjects.FloatVector( n.nodes.radius.tolist() ),
													  'confidence': robjects.IntVector( n.nodes.confidence.tolist() )
												  } )

		cn_data = robjects.DataFrame( {    'treenode_id': robjects.IntVector( n.connectors.treenode_id.tolist() ),
													  'connector_id': robjects.IntVector( n.connectors.connector_id.tolist() ),
													  'prepost': robjects.IntVector( n.connectors.relation.tolist() ),
													  'x': robjects.IntVector( n.connectors.x.tolist() ),
													  'y': robjects.IntVector( n.connectors.y.tolist() ),
													  'z': robjects.IntVector( n.connectors.z.tolist() )
												  } )

		res = robjects.ListVector({
											'nodes': node_data,
											'connectors': cn_data ,
											'tags' : robjects.ListVector ( n.tags )
									})
		"""

		#Prepare list of parents -> root node's parent "None" has to be replaced with -1
		parents = n.nodes.parent_id.tolist()
		parents[ parents.index(None) ] = -1  #should technically be robjects.r('-1L')

		swc = robjects.DataFrame( { 		'PointNo' : robjects.IntVector( n.nodes.treenode_id.tolist() ), 
											 'Label' : robjects.IntVector( [ 0 ] * n.nodes.shape[0]),
											 'X':	robjects.IntVector( n.nodes.x.tolist() ), 
											 'Y': robjects.IntVector( n.nodes.y.tolist() ), 
											 'Z': robjects.IntVector( n.nodes.z.tolist() ), 
											 'W': robjects.FloatVector( [ w * 2 for w in n.nodes.radius.tolist() ]  ) , 
											 'Parent': robjects.IntVector( parents )
						} )

		if n.nodes[ n.nodes.radius > 500 ].shape[0] == 1:
			soma_id = n.nodes[ n.nodes.radius > 500 ].treenode_id.tolist()[0]
		else:
			soma_id = robjects.r('NULL')

		#Generate nat neuron
		n_r = nat.as_neuron( swc, origin = soma_id, skid = n.skeleton_id )
			
		#Convert back to python dict so that we can add additional data
		n_py = { n_r.names[i] : n_r[i] for i in range(len(n_r)) }

		if n.connectors.shape[0] > 0:
			n_py['connectors'] = robjects.DataFrame( {    'treenode_id': robjects.IntVector( n.connectors.treenode_id.tolist() ),
														  'connector_id': robjects.IntVector( n.connectors.connector_id.tolist() ),
														  'prepost': robjects.IntVector( n.connectors.relation.tolist() ),
														  'x': robjects.IntVector( n.connectors.x.tolist() ),
														  'y': robjects.IntVector( n.connectors.y.tolist() ),
														  'z': robjects.IntVector( n.connectors.z.tolist() )
													  } )
		else:
			n_py['connectors'] = robjects.r('NULL')

		n_py['tags'] = robjects.ListVector( n.tags )

		#R neuron objects contain information about URL and response headers -> since we don't have that (yet), we will create the entries but leave them blank
		n_py['url'] = robjects.r('NULL')
		n_py['headers'] = robjects.ListVector( {
													'server ': 'NA',
													'date': 'NA',
													'content-type': 'NA',
													'transfer-encoding': 'NA',
													'connections': 'NA',
													'vary': 'NA',
													'expires': 'NA',
													'cache-control': 'NA',
													'content-encoding': 'NA'
												})


		#Convert back to R object
		n_r = robjects.ListVector( n_py )
		n_r.rclass = robjects.r('c("catmaidneuron","neuron")')

		return n_r
	else:
		module_logger.error('Unknown DataFrame format: %s' % str( type( neuron ) ) )













