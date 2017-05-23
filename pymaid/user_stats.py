"""
    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
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
"""

try:
   from pymaid import get_3D_skeleton, get_user_list, get_node_user_details
except:
   from pymaid.pymaid import get_3D_skeleton, get_user_list, get_node_user_details

import logging
import pandas as pd

#Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

if not module_logger.handlers:
  #Generate stream handler
  sh = logging.StreamHandler()
  sh.setLevel(logging.DEBUG)
  #Create formatter and add it to the handlers
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  sh.setFormatter(formatter)
  module_logger.addHandler(sh)

def get_time_invested( skids, remote_instance, interval = 1 ):
	"""
	Takes a list of skeleton IDs and calculates the time each user has spent working
	on this set of neurons.

	Parameters:
	-----------
	skids : 			single or list of skeleton IDs
	remote_instance :	Catmaid Instance
	interval :			integer (default = 1)
						size of the bins in minutes

	Returns:
	-------
	pandas DataFrame

		user   total   creation   edition   review
	0
	1
	2

	Values represent minutes. Creation/Edition/Review can overlap! This is why total 
	time spent is < creation + edition + review.
	"""

	bin_width = '%iMin' % interval

	user_list = get_user_list( remote_instance ).set_index('id')

	skdata = get_3D_skeleton( skids, remote_instance = remote_instance )

	node_ids = []
	connector_ids = []
	for n in skdata.itertuples():
		node_ids += n.nodes.treenode_id.tolist()
		connector_ids += n.connectors.connector_id.tolist()

	node_details = get_node_user_details( node_ids + connector_ids, remote_instance = remote_instance )		

	creation_timestamps = pd.DataFrame( node_details[ [ 'user' , 'creation_time' ]  ].values, columns = [ 'user' , 'timestamp' ] )
	edition_timestamps = pd.DataFrame( node_details[ [ 'editor' , 'edition_time' ]  ].values, columns = [ 'user' , 'timestamp' ] )
	#Generate dataframe for reviews	
	reviewers  = [ u for l in node_details.reviewers.tolist() for u in l ]
	timestamps  = [ ts for l in node_details.review_times.tolist() for ts in l ]
	review_timestamps = pd.DataFrame( [ [ u, ts ] for u, ts in zip (reviewers, timestamps ) ] , columns = [ 'user', 'timestamp' ] )

	all_timestamps = pd.concat( [creation_timestamps , edition_timestamps, review_timestamps ], axis = 0)

	stats = { 
				'total' : { u : 0 for u in all_timestamps.user.unique() },
				'creation' : { u : 0 for u in all_timestamps.user.unique() },
				'edition' : { u : 0 for u in all_timestamps.user.unique() },
				'review' : { u : 0 for u in all_timestamps.user.unique() }
				}

	for u in all_timestamps.user.unique():
		stats['total'][u] += sum( all_timestamps[ all_timestamps.user == u ].timestamp.to_frame().set_index('timestamp', drop = False ).groupby( pd.TimeGrouper( freq = bin_width ) ).count().values > 0 )[0] * interval
	for u in creation_timestamps.user.unique():		
		stats['creation'][u] += sum ( creation_timestamps[ creation_timestamps.user == u ].timestamp.to_frame().set_index('timestamp', drop = False ).groupby( pd.TimeGrouper( freq = bin_width ) ).count().values > 0 )[0] * interval
	for u in edition_timestamps.user.unique():
		stats['edition'][u] += sum ( edition_timestamps[ edition_timestamps.user == u ].timestamp.to_frame().set_index('timestamp', drop = False ).groupby( pd.TimeGrouper( freq = bin_width ) ).count().values > 0 )[0] * interval
	for u in review_timestamps.user.unique():
		stats['review'][u] += sum ( review_timestamps[ review_timestamps.user == u ].timestamp.to_frame().set_index('timestamp', drop = False ).groupby( pd.TimeGrouper( freq = bin_width ) ).count().values > 0 )[0] * interval	

	module_logger.info('Done! Use e.g. plotly to generate a plot: \n stats = get_time_invested( skids, remote_instance ) \n fig = { "data" : [ { "values" : stats.total.tolist(), "labels" : stats.user.tolist(), "type" : "pie" } ] } \n plotly.offline.plot(fig) ')
 
	return pd.DataFrame( [ [  user_list.ix[ u ].last_name, stats['total'][u] , stats['creation'][u], stats['edition'][u], stats['review'][u] ] for u in all_timestamps.user.unique() ] , columns = [ 'user', 'total' ,'creation', 'edition', 'review' ] ).sort_values('total', ascending = False)