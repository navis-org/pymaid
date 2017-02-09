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


import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import random, colorsys
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from pymaid import get_3D_skeleton

def plotneuron(skids, remote_instance, plot_outlines = True):
	""" 
	Retrieves 3D skeletons and generates matplotlib object.	
	Currently plots frontal view (x,y axes). X and y limits 
	have been set to fit the adult EM volume -> adjust if necessary.

	Parameters:
	----------
	skdis :				list
						list of CATMAID skeleton ids
	remote_instance :	CATMAID remote instance
	plot_outlines :		boolean
						If true brain outlines are plotted -> adult brain outlines are hard coded!

	Returns:
	--------
	fig, ax :			matplotlib figure and axe object


	"""

	brain = [(571757, -169305), (562337, -165537), (546490, -164097), (534964, -163653), (525877, -161215), (521001, -162878), (502161, -159775), (483654, -161658), (467586, -165426), (444978, -173295), (431347, -185596), (422592, -196124), (409072, -202551), (401204, -208425), (395774, -217734), (395331, -226710), (400872, -239011), (410735, -247212), (423479, -254858), (425474, -256742), (420487, -259291), (412508, -255413), (401093, -251534), (390676, -252531), (382143, -244109), (376159, -235022), (369842, -231697), (355546, -228594), (331166, -232473), (300025, -244109), (290273, -250315), (279856, -262394), (267998, -282009), (265782, -294311), (261127, -309493), (260462, -320132), (257470, -340855), (260241, -359362), (264895, -377869), (275756, -399479), (289386, -418429), (303682, -432504), (319419, -442699), (335044, -451011), (352887, -455665), (365298, -457438), (382697, -454224), (394998, -447797), (403531, -438709), (414613, -431285), (418825, -422973), (431237, -416213), (440545, -406904), (451184, -401031), (463485, -386956), (467696, -373658), (479776, -384075), (491744, -393051), (509919, -400144), (513909, -410561), (515017, -420646), (526985, -431395), (550701, -444029), (556685, -445802), (568875, -444694), (583171, -447686), (600238, -447021), (619853, -449237), (636698, -447686), (644012, -449348), (650772, -448905), (678920, -437269), (690667, -423970), (696541, -408788), (699311, -402250), (708398, -397373), (719702, -387621), (740758, -373880), (744637, -373325), (742642, -384186), (744526, -395046), (750953, -404577), (759487, -413997), (768352, -419538), (772896, -420092), (782648, -428957), (783867, -434277), (796722, -447021), (803815, -455887), (818221, -465417), (830855, -470182), (840386, -471180), (863325, -468742), (883716, -460763), (910646, -443807), (929596, -423305), (945665, -399812), (955860, -372993), (958631, -343958), (954198, -318358), (943227, -293978), (927158, -274695), (907432, -259513), (883938, -247433), (861109, -244663), (844264, -247766), (831963, -257961), (827530, -266162), (819108, -266495), (801598, -273698), (792733, -283893), (787413, -279128), (780321, -275249), (780210, -272922), (784754, -266162), (802042, -255412), (810464, -245882), (813345, -236905), (810575, -224715), (800379, -212525), (791514, -206541), (778880, -201332), (771677, -192799), (753059, -179500), (735217, -168973), (711280, -163875), (692329, -163764), (679807, -167532), (658197, -170635), (632486, -171078), (616196, -177727), (611098, -181606), (598021, -184155), (591150, -183047), (583171, -176398)]
	brain_p = Polygon(brain, closed = True, lw = 0, fill =True, fc = (0.9,0.9,0.9), alpha = 0.5 )
	hole = [(601774, -302587), (604265, -304709), (608786, -311906), (608786, -322239), (616536, -334602), (618565, -344197), (618381, -349548), (609524, -365048), (595592, -364218), (593378, -362003), (591994, -354622), (588765, -342998), (589503, -334694), (591348, -321501), (591348, -310891), (595961, -306739)]
	hole_p = Polygon(hole, closed = True, lw = 0, fill =True, fc = 'white', alpha = 1 )

	skdata = get_3D_skeleton ( skids, remote_instance, connector_flag = 1, tag_flag = 0 , get_history = False, time_out = None, silent = True)	

	fig, ax = plt.subplots(figsize = (15,7))	
	if plot_outlines:
		ax.add_patch( brain_p )
		ax.add_patch( hole_p )
	ax.set_ylim( ( -510000, -150000 ) )
	ax.set_xlim( ( 200000, 1000000 ) )
	plt.axis('off')

	colormap = random_colors ( len(skdata) , color_range='RGB') 

	for i, neuron in enumerate( skdata ):
		lines = []		

		#Get a list of all break points (leaf, branch, root):
		parent_nodes = [ n[1] for n in neuron[0] if n[1] != None ]
		list_of_parents = { n[0]:n[1] for n in neuron[0] }
		b_points = [ n for n in neuron[0] if parent_nodes.count( n[0] ) != 1 ]
		b_point_ids = [ n[0] for n in b_points ]
		node_dict = { n[0]: n for n in neuron[0] }		
		soma = [ n for n in neuron[0] if n[6] > 1000 ]
		
		for node in b_points:					
			if node[1] == None:
				continue

			parent = None
			this_node = node
			this_line = [ node[3:6] ]
			while parent not in b_point_ids:				
				#this_node = [ n for n in neuron[0] if n[0] == this_node[1] ][0]				
				this_node = node_dict[ this_node[1] ]
				this_line.append( this_node[3:6] )			
				parent = list_of_parents[ this_node[0] ]

				if parent is None:
					break				

			lines.append( this_line )				
		
		for k,l in enumerate( lines ):
			if k == 0:
				this_line = mlines.Line2D( [ x[0] for x in l ],[ -y[1] for y in l  ], lw = 1 , alpha = 0.7, color = colormap[i], label = '#%s' % skids[i] ) 			
			else:
				this_line = mlines.Line2D( [ x[0] for x in l ],[ -y[1] for y in l  ], lw = 1 , alpha = 0.7, color = colormap[i]) 
			ax.add_line(this_line)	

		if soma:			
			s = Circle( ( int( soma[0][3] ), int( -soma[0][4] ) ), radius = soma[0][6] , alpha = 0.7, fill = True, color = colormap[i] )
			ax.add_patch(s)


	return fig, ax


def random_colors (color_count,color_range='RGB'):
	""" Divides colorspace into N evenly distributed colors
	Returns
	-------
	colormap : 	list
				[ (r,g,b),(r,g,b),... ]
				
	"""
	### Make count_color an even number
	if color_count % 2 != 0:
		color_count += 1
	     
	colormap = []
	interval = 2/color_count  
	runs = int(color_count/2)   

	### Create first half with low brightness; second half with high brightness and slightly shifted hue
	if color_range == 'RGB':
		for i in range(runs):
			### High brightness
			h = interval * i
			s = 1
			v =  1
			hsv = colorsys.hsv_to_rgb(h,s,v)
			colormap.append( ( hsv[0], hsv[1], hsv[2] ) )             

			### Lower brightness, but shift hue by half an interval
			h = interval * (i+0.5)
			s = 1
			v =  0.5
			hsv = colorsys.hsv_to_rgb(h,s,v)
			colormap.append( ( hsv[0], hsv[1], hsv[2] ) )                       
	elif color_range == 'Grayscale':
		h = 0
		s = 0
		for i in range(color_count):
			v = 1/color_count * i
			hsv = colorsys.hsv_to_rgb(h,s,v)
			colormap.append( ( hsv[0], hsv[1], hsv[2] ) )    

	return(colormap)


if __name__ == '__main__':
	from connect_catmaid import connect_adult_em

	remote_instance = connect_adult_em()

	fix, ax = plotneuron([1420974], remote_instance)

	plt.legend()
	plt.show()
	#plt.savefig( 'renderings/neuron_plot.png', transparent = False )
