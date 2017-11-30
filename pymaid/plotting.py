#    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
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

""" Module contains functions to plot neurons in 2D and 3D.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import matplotlib.colors as mcl
import random
import colorsys
import logging
import png

import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import vispy
from vispy import scene
from vispy.geometry import create_sphere
from vispy.gloo.util import _screenshot

try:
    # Try setting vispy backend to PyQt5 
    vispy.use(app='PyQt5')
except:
    pass

import sys
import pandas as pd
import numpy as np
import random
import math
import igraph
from colorsys import hsv_to_rgb

from pymaid import morpho, igraph_catmaid, core, pymaid
from pymaid import cluster as clustmaid

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if len( module_logger.handlers ) == 0:
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
                '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)

__all__ = ['plot3d','plot2d','plot_network','clear3d','close3d','screenshot',]


def screenshot(file='screenshot.png', alpha=True):
    """ Saves a screenshot of active 3D canvas.

    Parameters
    ----------
    file :      str, optional
                Filename
    alpha :     bool, optional
                If True, alpha channel will be saved
    """
    if alpha:
        mode = 'RGBA'
    else:
        mode = 'RGB'

    im = png.from_array(_screenshot(alpha=alpha), mode=mode)
    im.save(file)

    return


def clear3d():
    """ Clear 3D canvas
    """
    try:
        canvas = globals()['canvas']
        canvas.central_widget.remove_widget(canvas.central_widget.children[0])
        canvas.update()
        globals().pop('vispy_scale_factor')
        del vispy_scale_factor
    except:
        pass


def close3d():
    """ Close existing 3D canvas (wipes memory)
    """
    try:
        canvas = globals()['canvas']
        canvas.close()
        globals().pop('canvas')
        globals().pop('vispy_scale_factor')        
        del canvas
        del vispy_scale_factor
    except:
        pass


def plot2d(x, *args, **kwargs):
    """ Generate 2D plots of neurons and neuropils.

    Parameters
    ----------
    x :               {skeleton IDs, core.CatmaidNeuron, core.CatmaidNeuronList, core.CatmaidVolume}
                      Objects to plot:: 

                        - int is intepreted as skeleton ID(s) 
                        - str is intepreted as volume name(s) 
                        - multiple objects can be passed as list (see examples)

    remote_instance : Catmaid Instance, optional
                      Need this too if you are passing only skids
    *args
                      See Notes for permissible arguments.
    **kwargs
                      See Notes for permissible keyword arguments.

    Examples
    --------
    >>> # 1. Plot two neurons and have plot2d download the skeleton data for you:    
    >>> fig, ax = pymaid.plot2d( [12345, 45567] )
    >>> matplotlib.pyplot.show()
    >>> # 2. Manually download a neuron, prune it and plot it:
    >>> neuron = pymaid.get_neuron( [12345], rm )
    >>> neuron.prune_distal_to( 4567 )
    >>> fig, ax = pymaid.plot2d( neuron )
    >>> matplotlib.pyplot.show()
    >>> # 3. Plots neuropil in grey, and mushroom body in red:
    >>> np = pymaid.get_volume('v14.neuropil')
    >>> np.color = (.8,.8,.8)
    >>> mb = pymaid.get_volume('v14.MB_whole')
    >>> mb.color = (.8,0,0)
    >>> fig, ax = pymaid.plot2d(  [ 12346, np, mb ] )    
    >>> matplotlib.pyplot.show()

    Returns
    --------
    fig, ax :      matplotlib figure and axis object

    Notes
    -----
    Currently plots only frontal view (x,y axes). X and y limits have been set
    to fit the adult EM volume -> adjust if necessary.

    Optional ``*args`` and ``**kwargs``:

    ``view`` (str, {'dorsal','lateral','frontal'})
       By default, frontal view is plotted.

    ``connectors`` (boolean, default = True )
       Plot connectors (synapses, gap junctions, abutting)

    ``connectors_only`` (boolean, default = False)
       Plot only connectors, not the neuron

    ``zoom`` (boolean, default = False)
       Zoom in on higher brain centers

    ``auto_limits`` (boolean, default = True)
       By default, limits are being calculated such that they fit the neurons
       plotted.

    ``limits`` (dict, default = None)
       Manually override limits for plot. Dict needs to define min and max
       values for each axis: ``{ 'x' : [ int, int ], 'y' : [ int, int ] }``
       If ``auto_limits = False`` and ``limits = None``, a hard-coded fallback is used
       - this may not suit your needs!

    ``ax`` (matplotlib ax, default=None)
       Pass an ax object if you want to plot on an existing canvas

    ``color`` (tuple, dict)

    """

    #Dotprops are currently ignored!
    skids, skdata, _dotprops, volumes = _parse_objects(x)         

    remote_instance = kwargs.get('remote_instance', None)
    connectors = kwargs.get('connectors', True)
    connectors_only = kwargs.get('connectors_only', False)
    zoom = kwargs.get('zoom', False)
    limits = kwargs.get('limits', None)
    auto_limits = kwargs.get('auto_limits', False)
    ax = kwargs.get('ax', None)
    color = kwargs.get('color', None)
    view = kwargs.get('view', 'frontal')

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if skids:
        skdata += pymaid.get_neuron(skids, remote_instance, connector_flag=1,
                                   tag_flag=0, get_history=False, get_abutting=True)   

    if not color and (skdata.shape[0] + _dotprops.shape[0])>0:
        cm = _random_colors(
            skdata.shape[0] + _dotprops.shape[0], color_space='RGB', color_range=1)
        colormap = {}

        if not skdata.empty:
            colormap.update(
                {str(n): cm[i] for i, n in enumerate(skdata.skeleton_id.tolist())})            
        if not _dotprops.empty:
            colormap.update({str(n): cm[i + skdata.shape[0]]
                             for i, n in enumerate(_dotprops.gene_name.tolist())})            
    elif isinstance(color, dict):
        colormap = {n: tuple(color[n]) for n in color}
    elif isinstance(color,(list,tuple)):        
        colormap = {n: tuple(color) for n in skdata.skeleton_id.tolist()}        
    elif isinstance(color,str):
        color = tuple( [ int(c *255) for c in mcl.to_rgb(color) ] )
        colormap = {n: color for n in skdata.skeleton_id.tolist()}
    else:
        colormap = {} 

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
    else:
        fig = None #we don't really need this    

    if volumes:
        for v in volumes:            
            c = v.get('color', (0.9, 0.9, 0.9))

            if sum(c) > 3:
                c = np.array(c) / 255

            vpatch = mpatches.Polygon(
            v.to_2d(invert_y=True), closed=True, lw=0, fill=True, fc=c, alpha=1)
            ax.add_patch(vpatch)   

    if limits:
        catmaid_limits = limits
    elif auto_limits:
        min_x = min([n.nodes.x.min() for n in skdata.itertuples()] +
                    [n.connectors.x.min() for n in skdata.itertuples()])
        max_x = max([n.nodes.x.max() for n in skdata.itertuples()] +
                    [n.connectors.x.max() for n in skdata.itertuples()])

        min_y = min([n.nodes.y.min() for n in skdata.itertuples()] +
                    [n.connectors.y.min() for n in skdata.itertuples()])
        max_y = max([n.nodes.y.max() for n in skdata.itertuples()] +
                    [n.connectors.y.max() for n in skdata.itertuples()])

        min_z = min([n.nodes.z.min() for n in skdata.itertuples()] +
                    [n.connectors.z.min() for n in skdata.itertuples()])
        max_z = max([n.nodes.z.max() for n in skdata.itertuples()] +
                    [n.connectors.z.max() for n in skdata.itertuples()])

        max_dim = max([max_x - min_x, max_y - min_y, max_z - min_z]) * 1.1

        # Also make sure that dimensions along all axes are the same -
        # otherwise plot will be skewed
        catmaid_limits = {  # These limits refer to x/y/z in CATMAID -> will later on be inverted to make 2d plot
            'x': [int((min_x + (max_x - min_x) / 2) - max_dim / 2), int((min_x + (max_x - min_x) / 2) + max_dim / 2)],
            'z': [int((min_z + (max_z - min_z) / 2) - max_dim / 2), int((min_z + (max_z - min_z) / 2) + max_dim / 2)],
            # This needs to be inverted
            'y': [int((min_y + (max_y - min_y) / 2) + max_dim / 2), int((min_y + (max_y - min_y) / 2) - max_dim / 2)]
        }  # z and y need to be inverted here!
    elif zoom:
        catmaid_limits = {  # These limits refer to x/y in CATMAID -> need to invert y (CATMAID coordindates start in upper left, matplotlib bottom left)
            'x': [380000, 820000],
            'y': [355333, 150000]
        }
    else:
        catmaid_limits = {  # These limits refer to x/y in CATMAID -> need to invert y (CATMAID coordindates start in upper left, matplotlib bottom left)
            #'x': [200000, 1000000],
            #'y': [510000, 150000]
            'x': [200000, 1000000],
            'y': [730000, -70000]
        }

    ax.set_ylim((-catmaid_limits['y'][0], -catmaid_limits['y'][1]))
    ax.set_xlim((catmaid_limits['x'][0], catmaid_limits['x'][1]))

    plt.axis('off')

    module_logger.debug('Plot limits set to: x= %i -> %i; y = %i -> %i' % (catmaid_limits[
                        'x'][0], catmaid_limits['x'][1], -catmaid_limits['y'][0], -catmaid_limits['y'][1]))    

    # Create slabs (lines)
    for i, neuron in enumerate(skdata.itertuples()):
        module_logger.debug('Working on neuron %s...' % neuron.neuron_name)
        lines = []

        if 'type' not in neuron.nodes:
            morpho.classify_nodes(neuron)

        soma = neuron.nodes[neuron.nodes.radius > 1]

        if not connectors_only:
            # Now make traces
            try:
                neuron._generate_segments
            except:
                neuron._generate_segments = morpho._generate_segments(neuron)

            lines = _slabs_to_coords(neuron, neuron.slabs, invert=False)

            module_logger.debug('Creating %i lines' % len(lines))
            module_logger.debug([len(l) for l in lines])
            for k, l in enumerate(lines):
                # User first line to assign a legend
                if k == 0:
                    this_line = mlines.Line2D([int(x[0]) for x in l], [-int(y[1]) for y in l], lw=1, alpha=.9, color=colormap[ neuron.skeleton_id ], 
                                label='%s - #%s' % (neuron.neuron_name, neuron.skeleton_id))
                else:
                    this_line = mlines.Line2D(
                        [int(x[0]) for x in l], [- int(y[1]) for y in l], lw=1, alpha=.9, color=colormap[ neuron.skeleton_id ])
                ax.add_line(this_line)

            for n in soma.itertuples():
                s = mpatches.Circle((int(n.x), int(-n.y)), radius=n.radius, alpha=.9,
                                    fill=True, color=colormap[ neuron.skeleton_id ], zorder=4, edgecolor='none')
                ax.add_patch(s)

        if connectors or connectors_only:
            module_logger.debug('Plotting %i pre- and %i postsynapses' % (neuron.connectors[
                                neuron.connectors.relation == 0].shape[0], neuron.connectors[neuron.connectors.relation == 1].shape[0]))
            # postsynapses
            ax.scatter(neuron.connectors[neuron.connectors.relation == 1].x.tolist(), (-neuron.connectors[
                       neuron.connectors.relation == 1].y).tolist(), c='blue', alpha=1, zorder=4, edgecolor='none', s=2)
            # presynapses
            ax.scatter(neuron.connectors[neuron.connectors.relation == 0].x.tolist(), (-neuron.connectors[
                       neuron.connectors.relation == 0].y).tolist(), c='red', alpha=1, zorder=4, edgecolor='none', s=2)
            # gap junctions
            ax.scatter(neuron.connectors[neuron.connectors.relation == 2].x.tolist(), (-neuron.connectors[
                       neuron.connectors.relation == 2].y).tolist(), c='green', alpha=1, zorder=4, edgecolor='none', s=2)
            # abutting
            ax.scatter(neuron.connectors[neuron.connectors.relation == 3].x.tolist(), (-neuron.connectors[
                       neuron.connectors.relation == 3].y).tolist(), c='magenta', alpha=1, zorder=4, edgecolor='none', s=1)

    module_logger.info('Done. Use matplotlib.pyplot.show() to show plot.')

    return fig, ax


def _slabs_to_coords(x, slabs, invert=False):
    """Turns lists of treenode_ids into coordinates

    Parameters
    ----------
    x :         {pandas DataFrame, CatmaidNeuron}
                Must contain the nodes
    slabs :     list of treeenode IDs
    invert :    boolean, optional
                If True, coordinates will be inverted

    Returns
    -------
    coords :    list of tuples
                [ (x,y,z), (x,y,z ) ]
    """

    coords = []
    nodes = x.nodes.set_index('treenode_id')

    for l in slabs:
        if not invert:
            coords.append(nodes.ix[l][['x', 'y', 'z']].as_matrix().tolist())
        else:
            coords.append((nodes.ix[l][['x', 'y', 'z']]
                           * -1).as_matrix().tolist())

    return coords


def _random_colors(color_count, color_space='RGB', color_range=1):
    """ Divides colorspace into N evenly distributed colors

    Returns
    -------
    colormap :  list
             [ (r,g,b),(r,g,b),... ]

    """
    if color_count == 1:
        return [(0, 0, 0)]

    # Make count_color an even number
    if color_count % 2 != 0:
        color_count += 1

    colormap = []
    interval = 2 / color_count
    runs = int(color_count / 2)

    # Create first half with low brightness; second half with high brightness
    # and slightly shifted hue
    if color_space == 'RGB':
        for i in range(runs):
            # High brightness
            h = interval * i
            s = 1
            v = 1
            hsv = colorsys.hsv_to_rgb(h, s, v)
            colormap.append(tuple(v * color_range for v in hsv))

            # Lower brightness, but shift hue by half an interval
            h = interval * (i + 0.5)
            s = 1
            v = 0.5
            hsv = colorsys.hsv_to_rgb(h, s, v)
            colormap.append(tuple(v * color_range for v in hsv))
    elif color_space == 'Grayscale':
        h = 0
        s = 0
        for i in range(color_count):
            v = 1 / color_count * i
            hsv = colorsys.hsv_to_rgb(h, s, v)
            colormap.append(tuple(v * color_range for v in hsv))

    module_logger.debug('%i random colors created: %s' %
                        (color_count, str(colormap)))

    return(colormap)


def _fibonacci_sphere(samples=1, randomize=True):
    """ Calculates points on a sphere
    """
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return points


def plot3d(x, *args, **kwargs):
    """ Generates 3D plot using either vispy (default, http://vispy.org) or
    plotly (http://plot.ly)

    Parameters
    ----------

    x :               {skeleton IDs, core.CatmaidNeuron, core.CatmaidNeuronList,
                       core.Dotprops, core.Volumes}
                      Objects to plot:: 

                        - int is intepreted as skeleton ID(s) 
                        - str is intepreted as volume name(s) 
                        - multiple objects can be passed as list (see examples)

    remote_instance : CATMAID Instance, optional
                      Need to pass this too if you are providing only skids
                      also necessary if you want to include volumes! If
                      possible, will try to get remote instance from neuron
                      object.
    backend :         {'vispy','plotly'}, default = 'vispy'
       | ``vispy`` uses OpenGL to generate high-performance 3D plots but is less pretty.
       | ``plotly`` generates 3D plots in .html which are shareable but take longer to generate.

    connectors :      bool, default=False
                      Plot synapses and gap junctions.
    by_strahler :     bool, default=False
                      Will shade neuron(s) by strahler index.
    by_confidence :   bool, default=False
                      Will shade neuron(s) by arbor confidence                      
    cn_mesh_colors :  bool, default=False
                      Plot connectors using mesh colors.
    limits :          dict, optional
                      Manually override plot limits.
                      Format: ``{'x' :[min,max], 'y':[min,max], 'z':[min,max]}``
    auto_limits :     bool, default=True
                      Autoscales plot to fit the neurons.
    downsampling :    int, default=None
                      Set downsampling of neurons before plotting.
    clear3d :         bool, default=False
                      If True, canvas is cleared before plotting (only for
                      vispy).
    color :           {tuple, dict}, default=random
                      Use single tuple (r,g,b) to give all neurons the same
                      color. Use dict to give individual colors to neurons:
                      ``{ skid : (r,g,b), ... }``. R/G/B must be 0-255
    use_neuron_color : bool, default=False
                      If True, will try using the ``.color`` attribute of 
                      CatmaidNeurons.
    width :           int, default=600
    height :          int, default=600
                      Use to define figure/window size.
    title :           str, default='Neuron plot'
                      Plot title (for plotly only!)
    fig_autosize :    bool, default=False
                      For plotly only! Autoscale figure size.
                      Attention: autoscale overrides width and height

    Returns
    --------
    If ``backend='vispy'``

       Opens a 3D window and returns:

            - ``canvas`` - Vispy canvas object 
            - ``view`` - Vispy view object -> use to manipulate camera, add object, etc.

    If ``backend='plotly'``

       ``fig`` - dictionary to generate plotly 3D figure:

            Use for example: ``plotly.offline.plot(fig, filename='3d_plot.html')``
            to generate html file and open it webbrowser

    Examples
    --------
    This assumes that you have alread initialised a remote instance as ``rm``

    >>> # Plot single neuron
    >>> nl = pymaid.get_neuron(16, remote_instance=rm)
    >>> pymaid.plot3d(nl)
    >>> # Clear canvas
    >>> pymaid.clear3d()
    >>> # Plot3D can deal with combinations of objects
    >>> nl2 = pymaid.get_neuron('annotation:glomerulus DA1', remote_instance=rm)
    >>> vol = pymaid.get_volume('v13.LH_R')
    >>> vol['color'] = (255,0,0,.5)
    >>> # This plots two neuronlists, two volumes and a single neuron
    >>> pymaid.plot3d( [ nl1, nl2, vol, 'v13.AL_R', 233007 ] )    
    >>> # Pass kwargs
    >>> pymaid.plot3d(nl1, connectors=True, clear3d=True, )     
    """

    def _plot3d_vispy():
        """
        Plot3d() helper function to generate vispy 3D plots. This is just to
        improve readability.
        """
        if kwargs.get('clear3d', False):
            clear3d()

        if 'vispy_scale_factor' not in globals():
            # Calculate a scale factor: if the scene is too big, we run into issues with line width, etc.
            # Should keep it between -1000 and +1000
            global vispy_scale_factor
            max_dim = max([math.fabs(n)
                           for n in [max_x, min_x, max_y, min_y, max_z, min_z]])
            vispy_scale_factor = 1000 / max_dim
        else:
            vispy_scale_factor = globals()['vispy_scale_factor']           

        # If does not exists yet, initialise a canvas object and make global
        if 'canvas' not in globals():
            global canvas
            canvas = scene.SceneCanvas(keys='interactive', size=(
                width, height), bgcolor='white')
            view = canvas.central_widget.add_view()

            # Add camera
            view.camera = scene.TurntableCamera()

            # Set camera range
            view.camera.set_range((min_x * vispy_scale_factor, max_x * vispy_scale_factor),
                                  (min_y * vispy_scale_factor, max_y * vispy_scale_factor),
                                  (min_z * vispy_scale_factor, max_z * vispy_scale_factor)
                                  )
        else:
            canvas = globals()['canvas']

            # Check if we already have a view, if not (e.g. if plot.clear3d()
            # has been used) add new
            if canvas.central_widget.children:
                view = canvas.central_widget.children[0]
            else:
                view = canvas.central_widget.add_view()

                
                # Add camera
                view.camera = scene.TurntableCamera()

                # Set camera range
                view.camera.set_range((min_x * vispy_scale_factor, max_x * vispy_scale_factor),
                                      (min_y * vispy_scale_factor, max_y * vispy_scale_factor),
                                      (min_z * vispy_scale_factor, max_z * vispy_scale_factor)
                                      )                

        for i, neuron in enumerate(skdata.itertuples()):
            module_logger.debug('Working on neuron %s' %
                                str(neuron.skeleton_id))
            try:
                neuron_color = colormap[str(neuron.skeleton_id)]
            except:
                neuron_color = (0, 0, 0)

            if max(neuron_color) > 1:
                neuron_color = np.array(neuron_color) / 255            

            # Get root node indices (may be more than one if neuron has
            # been cut weirdly)
            root_ix = neuron.nodes[
                neuron.nodes.parent_id.isnull()].index.tolist()            

            if not connectors_only:

                # Extract treenode_coordinates and their parent's coordinates                
                tn_coords = neuron.nodes[['x', 'y', 'z']].apply(
                    pd.to_numeric).as_matrix()
                parent_coords = neuron.nodes.set_index('treenode_id').loc[neuron.nodes.parent_id.tolist(
                )][['x', 'y', 'z']].apply(pd.to_numeric).as_matrix()

                # Pop root from coordinate lists
                tn_coords = np.delete(tn_coords, root_ix, axis=0)
                parent_coords = np.delete(parent_coords, root_ix, axis=0)

                # Turn coordinates into segments
                segments = [item for sublist in zip(
                    tn_coords, parent_coords) for item in sublist]

                # Add alpha to color based on strahler
                if by_strahler or by_confidence:                    
                    if by_strahler:                            
                        if 'strahler_index' not in neuron.nodes:
                            morpho.calc_strahler_index(neuron)                        

                        # Generate list of alpha values
                        alpha = neuron.nodes['strahler_index'].as_matrix()

                    if by_confidence:
                        if 'arbor_confidence' not in neuron.nodes:
                            morpho.arbor_confidence(neuron)

                        # Generate list of alpha values
                        alpha = neuron.nodes['arbor_confidence'].as_matrix()

                    # Pop root from coordinate lists
                    alpha = np.delete(alpha, root_ix, axis=0)

                    alpha = alpha / (max(alpha)+1)                    
                    # Duplicate values (start and end of each segment!)
                    alpha = np.array([ v for l in zip(alpha,alpha) for v in l ])

                    # Turn color into array (need 2 colors per segment for beginnng and end)
                    neuron_color = np.array( [ neuron_color ] * (tn_coords.shape[0] * 2), dtype=float )

                    neuron_color = np.insert(neuron_color, 3, alpha, axis=1)

                if segments:
                    # Create line plot from segments. Note that we divide coords by
                    # a scale factor
                    t = scene.visuals.Line(pos=np.array(segments) * vispy_scale_factor,
                                           color=list(neuron_color),
                                           width=2,
                                           connect='segments',
                                           antialias=False,
                                           method='gl') #method can also be 'agg'
                    view.add(t)

                if by_strahler or by_confidence:
                    #Convert array back to a single color without alpha
                    neuron_color=neuron_color[0][:3]

                # Extract and plot soma
                soma = neuron.nodes[neuron.nodes.radius > 1]
                if soma.shape[0] >= 1:
                    radius = min(
                        soma.ix[soma.index[0]].radius * vispy_scale_factor, 10)
                    sp = create_sphere(5, 5, radius=radius)
                    s = scene.visuals.Mesh(vertices=sp.get_vertices() + soma.ix[soma.index[0]][
                                           ['x', 'y', 'z']].as_matrix() * vispy_scale_factor, 
                                           faces=sp.get_faces(), 
                                           color=neuron_color)
                    view.add(s)

            if connectors or connectors_only:             
                for j in [0, 1, 2]:
                    if cn_mesh_colors:
                        color = neuron_color
                    else:
                        color = syn_lay[j]['color']

                    if max(color) > 1:
                        color = np.array(color) / 255

                    this_cn = neuron.connectors[
                        neuron.connectors.relation == j]                    

                    if this_cn.empty:
                        continue

                    pos = this_cn[['x', 'y', 'z']].apply(
                        pd.to_numeric).as_matrix()

                    if syn_lay['display'] == 'mpatches.Circles':
                        con = scene.visuals.Markers()

                        con.set_data(pos=np.array(pos) * vispy_scale_factor,
                                     face_color=color, edge_color=color, size=1)

                        view.add(con)

                    elif syn_lay['display'] == 'lines':                        
                        tn_coords = neuron.nodes.set_index('treenode_id').ix[this_cn.treenode_id.tolist(
                        )][['x', 'y', 'z']].apply(pd.to_numeric).as_matrix()

                        segments = [item for sublist in zip(
                            pos, tn_coords) for item in sublist]                        

                        t = scene.visuals.Line(pos=np.array(segments) * vispy_scale_factor,
                                               color=color,
                                               width=2,
                                               connect='segments',
                                               antialias=False,
                                               method='gl') #method can also be 'agg'
                        view.add(t)

        for neuron in dotprops.itertuples():
            try:
                neuron_color = colormap[str(neuron.gene_name)]
            except:
                neuron_color = (10, 10, 10)

            if max(neuron_color) > 1:
                neuron_color = np.array(neuron_color) / 255

            # Prepare lines - this is based on nat:::plot3d.dotprops
            halfvect = neuron.points[
                ['x_vec', 'y_vec', 'z_vec']] / 2 * scale_vect

            starts = neuron.points[['x', 'y', 'z']
                                   ].as_matrix() - halfvect.as_matrix()
            ends = neuron.points[['x', 'y', 'z']
                                 ].as_matrix() + halfvect.as_matrix()

            segments = [item for sublist in zip(
                starts, ends) for item in sublist]

            t = scene.visuals.Line(pos=np.array(segments) * vispy_scale_factor,
                                   color=neuron_color,
                                   width=2,
                                   connect='segments',
                                   antialias=False,
                                   method='gl') #method can also be 'agg'
            view.add(t)

            # Add soma
            sp = create_sphere(5, 5, radius=4)
            s = scene.visuals.Mesh(vertices=sp.get_vertices(
            ) + np.array([neuron.X, neuron.Y, neuron.Z]) * vispy_scale_factor, faces=sp.get_faces(), color=neuron_color)
            view.add(s)

        # Now add neuropils:
        for v in volumes_data:
            color = np.array(volumes_data[v]['color'], dtype=float)

            # Add alpha
            if len(color) < 4:
                color = np.append(color, [.6])

            if max(color) > 1:
                color[:3] = color[:3] / 255

            s = scene.visuals.Mesh(vertices=np.array(volumes_data[v][
                                   'verts']) * vispy_scale_factor, faces=volumes_data[v]['faces'], color=color)
            view.add(s)

        # Add a 3D axis to keep us oriented
        # ax = scene.visuals.XYZAxis( )
        # view.add(ax)

        # And finally: show canvas
        canvas.show()

        module_logger.info(
            'Use pymaid.clear3d() to clear canvas and pymaid.close3d() to close canvas.')

        return canvas, view

    def _plot3d_plotly():
        """
        Plot3d() helper function to generate plotly 3D plots. This is just to
        improve readability and structure of the code.
        """

        if limits:
            catmaid_limits = limits
        elif auto_limits:
            # Set limits based on data but make sure that dimensions along all
            # axes are the same - otherwise plot will be skewed
            max_dim = max([max_x - min_x, max_y - min_y, max_z - min_z]) * 1.1
            catmaid_limits = {  # These limits refer to x/y/z in CATMAID -> will later on be inverted and switched to make 3d plot
                'x': [int((min_x + (max_x - min_x) / 2) - max_dim / 2), int((min_x + (max_x - min_x) / 2) + max_dim / 2)],
                'y': [int((min_z + (max_z - min_z) / 2) - max_dim / 2), int((min_z + (max_z - min_z) / 2) + max_dim / 2)],
                'z': [int((min_y + (max_y - min_y) / 2) - max_dim / 2), int((min_y + (max_y - min_y) / 2) + max_dim / 2)]
            }  # z and y need to be inverted here!
        elif not skdata.empty:
            catmaid_limits = {  # These limits refer to x/y/z in CATMAID -> will later on be inverted and switched to make 3d plot
                'x': [200000, 1000000],  # Make sure [0] < [1]!
                # Also make sure that dimensions along all axes are the same -
                # otherwise plot will be skewed
                'z': [-70000, 730000],
                'y': [-150000, 650000]
            }
        elif not dotprops.empty:
            catmaid_limits = {  # These limits refer to x/y/z in CATMAID -> will later on be inverted and switched to make 3d plot
                'x': [-500, 500],  # Make sure [0] < [1]!
                # Also make sure that dimensions along all axes are the same -
                # otherwise plot will be skewed
                'z': [-500, 500],
                'y': [-500, 500]
            }

        # Use catmaid project's limits to scale axis -> we basically have to invert
        # everything to give the plot the right orientation
        ax_limits = {
            'x': [-catmaid_limits['x'][1], -catmaid_limits['x'][0]],
            'z': [-catmaid_limits['z'][1], -catmaid_limits['z'][0]],
            'y': [-catmaid_limits['y'][1], -catmaid_limits['y'][0]]
        }

        trace_data = []

        # Generate sphere for somas
        fib_points = _fibonacci_sphere(samples=30)

        module_logger.info('Generating traces...')

        #Generate slabs for all neurons at once -> uses multi-cores!
        skdata._generate_segments()

        for i, neuron in enumerate(skdata.itertuples()):
            module_logger.debug('Working on neuron %s' %
                                str(neuron.skeleton_id))

            neuron_name = neuron.neuron_name
            skid = neuron.skeleton_id

            if not connectors_only:
                if by_strahler:
                    s_index = morpho.calc_strahler_index(
                        skdata.ix[i], return_dict=True)

                # First, we have to generate slabs from the neurons
                if 'type' not in neuron.nodes:
                    morpho.classify_nodes(neuron)

                soma = neuron.nodes[neuron.nodes.radius > 1]                

                coords = _slabs_to_coords(neuron, neuron.slabs, invert=True)

                # We have to add (None,None,None) to the end of each slab to
                # make that line discontinuous there
                coords = [t + [[None] * 3] for t in coords]

                x_coords = [co[0] for l in coords for co in l]
                y_coords = [co[1] for l in coords for co in l]
                z_coords = [co[2] for l in coords for co in l]
                c = []

                if by_strahler:
                    for k, s in enumerate(slabs):
                        this_c = 'rgba(%i,%i,%i,%f)' % (colormap[str(skid)][0],
                                                        colormap[str(skid)][1],
                                                        colormap[str(skid)][2],
                                                        s_index[s[0]] / max(s_index.values()
                                                                            ))
                        # Slabs are separated by a <None> coordinate -> this is
                        # why we need one more color entry
                        c += [this_c] * (len(s) + 1)
                else:
                    try:
                        c = 'rgb%s' % str(colormap[str(skid)])
                    except:
                        c = 'rgb(10,10,10)'

                trace_data.append(go.Scatter3d(x=x_coords,
                                               y=z_coords,  # y and z are switched
                                               z=y_coords,
                                               mode='lines',
                                               line=dict(
                                                   color=c,
                                                   width=5
                                               ),
                                               name=neuron_name,
                                               legendgroup=neuron_name,
                                               showlegend=True,
                                               hoverinfo='none'

                                               ))

                # Add soma(s):
                for n in soma.itertuples():
                    try:
                        color = 'rgb%s' % str(colormap[str(skid)])
                    except:
                        color = 'rgb(10,10,10)'
                    trace_data.append(go.Mesh3d(
                        x=[(v[0] * n.radius / 2) - n.x for v in fib_points],
                        # y and z are switched
                        y=[(v[1] * n.radius / 2) - n.z for v in fib_points],
                        z=[(v[2] * n.radius / 2) - n.y for v in fib_points],

                        alphahull=.5,
                        color=color,
                        name=neuron_name,
                        legendgroup=neuron_name,
                        showlegend=False,
                        hoverinfo='name'
                    )
                    )

            if connectors or connectors_only:
                # Set dataframe indices to treenode IDs - will facilitate
                # distributing nodes
                if neuron.nodes.index.name != 'treenode_id':
                    neuron.nodes.set_index('treenode_id', inplace=True)

                for j in [0, 1, 2]:
                    if cn_mesh_colors:
                        try:
                            color = colormap[str(skid)]
                        except:
                            color = (10,10,10)
                    else:
                        color = syn_lay[j]['color']

                    this_cn = neuron.connectors[
                        neuron.connectors.relation == j]

                    if syn_lay['display'] == 'mpatches.Circles':
                        trace_data.append(go.Scatter3d(
                            x=this_cn.x.as_matrix() * -1,
                            y=this_cn.z.as_matrix() * -1,  # y and z are switched
                            z=this_cn.y.as_matrix() * -1,
                            mode='markers',
                            marker=dict(
                                color='rgb%s' % str(color),
                                size=2
                            ),
                            name=syn_lay[j]['name'] + ' of ' + neuron_name,
                            showlegend=True,
                            hoverinfo='none'
                        ))
                    elif syn_lay['display'] == 'lines':
                        # Find associated treenode
                        tn = neuron.nodes.ix[this_cn.treenode_id.tolist()]
                        x_coords = [n for sublist in zip(this_cn.x.as_matrix(
                        ) * -1, tn.x.as_matrix() * -1, [None] * this_cn.shape[0]) for n in sublist]
                        y_coords = [n for sublist in zip(this_cn.y.as_matrix(
                        ) * -1, tn.y.as_matrix() * -1, [None] * this_cn.shape[0]) for n in sublist]
                        z_coords = [n for sublist in zip(this_cn.z.as_matrix(
                        ) * -1, tn.z.as_matrix() * -1, [None] * this_cn.shape[0]) for n in sublist]

                        trace_data.append(go.Scatter3d(
                            x=x_coords,
                            y=z_coords,  # y and z are switched
                            z=y_coords,
                            mode='lines',
                            line=dict(
                                color='rgb%s' % str(color),
                                width=5
                            ),
                            name=syn_lay[j]['name'] + ' of ' + neuron_name,
                            showlegend=True,
                            hoverinfo='none'
                        ))

            neuron.nodes.reset_index(inplace=True)

        for neuron in dotprops.itertuples():
            # Prepare lines - this is based on nat:::plot3d.dotprops
            halfvect = neuron.points[
                ['x_vec', 'y_vec', 'z_vec']] / 2 * scale_vect

            starts = neuron.points[['x', 'y', 'z']
                                   ].as_matrix() - halfvect.as_matrix()
            ends = neuron.points[['x', 'y', 'z']
                                 ].as_matrix() + halfvect.as_matrix()

            x_coords = [n for sublist in zip(
                starts[:, 0] * -1, ends[:, 0] * -1, [None] * starts.shape[0]) for n in sublist]
            y_coords = [n for sublist in zip(
                starts[:, 1] * -1, ends[:, 1] * -1, [None] * starts.shape[0]) for n in sublist]
            z_coords = [n for sublist in zip(
                starts[:, 2] * -1, ends[:, 2] * -1, [None] * starts.shape[0]) for n in sublist]

            try:
                c = 'rgb%s' % str(colormap[neuron.gene_name])
            except:
                c = 'rgb(10,10,10)'

            trace_data.append(go.Scatter3d(x=x_coords,  # (-neuron.nodes.ix[ s ].x).tolist(),
                                           # (-neuron.nodes.ix[ s ].z).tolist(), #y and z are switched
                                           y=z_coords,
                                           # (-neuron.nodes.ix[ s ].y).tolist(),
                                           z=y_coords,
                                           mode='lines',
                                           line=dict(
                                               color=c,
                                               width=5
                                           ),
                                           name=neuron.gene_name,
                                           legendgroup=neuron.gene_name,
                                           showlegend=True,
                                           hoverinfo='none'
                                           ))

            # Add soma
            rad = 4
            trace_data.append(go.Mesh3d(
                x=[(v[0] * rad / 2) - neuron.X for v in fib_points],
                # y and z are switched
                y=[(v[1] * rad / 2) - neuron.Z for v in fib_points],
                z=[(v[2] * rad / 2) - neuron.Y for v in fib_points],

                alphahull=.5,

                color=c,
                name=neuron.gene_name,
                legendgroup=neuron.gene_name,
                showlegend=False,
                hoverinfo='name'
            )
            )

        module_logger.info('Tracing done.')

        # Now add neuropils:
        for v in volumes_data:
            if volumes_data[v]['verts']:
                trace_data.append(go.Mesh3d(
                    x=[-v[0] for v in volumes_data[v]['verts']],
                    # y and z are switched
                    y=[-v[2] for v in volumes_data[v]['verts']],
                    z=[-v[1] for v in volumes_data[v]['verts']],

                    i=[f[0] for f in volumes_data[v]['faces']],
                    j=[f[1] for f in volumes_data[v]['faces']],
                    k=[f[2] for f in volumes_data[v]['faces']],

                    opacity=.5,
                    color='rgb' + str(volumes_data[v]['color']),
                    name=v,
                    showlegend=True,
                    hoverinfo='none'
                )
                )

        layout = dict(
            width=width,
            height=height,
            autosize=fig_autosize,
            title=pl_title,
            scene=dict(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(240, 240, 240)',
                    range=ax_limits['x']

                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(240, 240, 240)',
                    range=ax_limits['y']
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(240, 240, 240)',
                    range=ax_limits['z']
                ),
                camera=dict(
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    eye=dict(
                        x=-1.7428,
                        y=1.0707,
                        z=0.7100,
                    )
                ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )

        # Need to remove width and height to make autosize actually matter
        if fig_autosize:
            layout.pop('width')
            layout.pop('height')

        fig = dict(data=trace_data, layout=layout)

        module_logger.info('Done. Plotted %i nodes and %i connectors' % (sum([n.nodes.shape[0] for n in skdata.itertuples() if not connectors_only] + [
                           n.points.shape[0] for n in dotprops.itertuples()]), sum([n.connectors.shape[0] for n in skdata.itertuples() if connectors or connectors_only])))
        module_logger.info(
            'Use plotly.offline.plot(fig, filename="3d_plot.html") to plot. Optimised for Google Chrome.')

        return fig    

    skids, skdata, dotprops, volumes = _parse_objects(x)    

    # Backend
    backend = kwargs.get('backend', 'vispy')

    # CatmaidInstance    
    remote_instance = kwargs.get('remote_instance', None)    

    # Parameters for neurons    
    color = kwargs.get('color', None)
    names = kwargs.get('names', [])
    downsampling = kwargs.get('downsampling', 1)
    connectors = kwargs.get('connectors', False)
    by_strahler = kwargs.get('by_strahler', False)
    by_confidence = kwargs.get('by_confidence', False)
    cn_mesh_colors = kwargs.get('cn_mesh_colors', False)
    connectors_only = kwargs.get('connectors_only', False)
    use_neuron_color = kwargs.get('use_neuron_color', False)

    syn_lay_new = kwargs.get('synapse_layout',  {})
    syn_lay = {0: {
        'name': 'Presynapses',
        'color': (255, 0, 0)
    },
        1: {
        'name': 'Postsynapses',
        'color': (0, 0, 255)
    },
        2: {
        'name': 'Gap junctions',
        'color': (0, 255, 0)
    },
        'display': 'lines' #'mpatches.Circles' 
    }
    syn_lay.update(syn_lay_new)

    # Parameters for dotprops
    scale_vect = kwargs.get('scale_vect', 1)
    alpha_range = kwargs.get('alpha_range', False)

    # Parameters for figure
    pl_title = kwargs.get('title', 'Neuron Plot')
    width = kwargs.get('width', 600)
    height = kwargs.get('height', 600)
    fig_autosize = kwargs.get('fig_autosize', False)
    limits = kwargs.get('limits', [])
    auto_limits = kwargs.get('auto_limits', True)
    auto_limits = kwargs.get('autolimits', auto_limits)

    if backend not in ['plotly','vispy']:
        module_logger.error(
            'Unknown backend: %s. See help(plot.plot3d).' % str(backend))
        return

    if not remote_instance and isinstance(skdata, core.CatmaidNeuronList):
        try:
            remote_instance = skdata._remote_instance
        except:
            pass

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        
    if skids and remote_instance:
        skdata += pymaid.get_neuron(skids, remote_instance,
                                   connector_flag=1,
                                   tag_flag=0,
                                   get_history=False,                                   
                                   get_abutting=True)
    elif skids and not remote_instance:
        module_logger.error(
            'You need to provide a CATMAID remote instance.')

    if not color and (skdata.shape[0] + dotprops.shape[0])>0:
        cm = _random_colors(
            skdata.shape[0] + dotprops.shape[0], color_space='RGB', color_range=255)
        colormap = {}

        if not skdata.empty:
            colormap.update(
                {str(n): cm[i] for i, n in enumerate(skdata.skeleton_id.tolist())})            
        if not dotprops.empty:
            colormap.update({str(n): cm[i + skdata.shape[0]]
                             for i, n in enumerate(dotprops.gene_name.tolist())})       
        if use_neuron_color:
            colormap.update( { n.skeleton_id : n.color for n in skdata } )     
    elif isinstance(color, dict):
        colormap = {n: tuple(color[n]) for n in color}
    elif isinstance(color,(list,tuple)):
        colormap = {n: tuple(color) for n in skdata.skeleton_id.tolist()}
    elif isinstance(color,str):
        color = tuple( [ int(c *255) for c in mcl.to_rgb(color) ] )
        colormap = {n: color for n in skdata.skeleton_id.tolist()}
    else:
        colormap = {}

    # Make sure colors are 0-255
    if colormap:
        if max([ v for n in colormap for v in colormap[n] ]) <= 1:
            module_logger.warning('Looks like RGB values are 0-1. Converting to 0-255.')
            colormap = { n : tuple( [ int(v * 255) for v in colormap[n] ] ) for n in colormap }   

    # Get and prepare volumes
    volumes_data = {}
    for v in volumes:
        if isinstance(v, str):
            if not remote_instance:
                module_logger.error(
                    'Unable to add volumes - please also pass a Catmaid Instance using <remote_instance = ... >')
                return
            else:
                v = pymaid.get_volume(v, remote_instance)
        
        volumes_data[ v['name'] ] = {'verts': v['vertices'],
                           'faces': v['faces'], 'color': v['color']}

    # Get boundaries of what to plot
    min_x = min([n.nodes.x.min() for n in skdata.itertuples()] + 
                [n.connectors.x.min() for n in skdata.itertuples()] + 
                [n.points.x.min() for n in dotprops.itertuples()] +
                [ np.array(volumes_data[v]['verts']).min(axis=0)[0] for v in volumes_data ] )
    max_x = max([n.nodes.x.max() for n in skdata.itertuples()] + 
                [n.connectors.x.max() for n in skdata.itertuples()] + 
                [n.points.x.max() for n in dotprops.itertuples()] +
                [ np.array(volumes_data[v]['verts']).max(axis=0)[0] for v in volumes_data ] )

    min_y = min([n.nodes.y.min() for n in skdata.itertuples()] + 
                [n.connectors.y.min() for n in skdata.itertuples()] + 
                [n.points.y.min() for n in dotprops.itertuples()] +
                [ np.array(volumes_data[v]['verts']).min(axis=0)[1] for v in volumes_data ] )

    max_y = max([n.nodes.y.max() for n in skdata.itertuples()] + 
                [n.connectors.y.max() for n in skdata.itertuples()] + 
                [n.points.y.max() for n in dotprops.itertuples()] +
                [ np.array(volumes_data[v]['verts']).max(axis=0)[1] for v in volumes_data ] )

    min_z = min([n.nodes.z.min() for n in skdata.itertuples()] + 
                [n.connectors.z.min() for n in skdata.itertuples()] + 
                [n.points.z.min() for n in dotprops.itertuples()] +
                [ np.array(volumes_data[v]['verts']).min(axis=0)[2] for v in volumes_data ] )
    max_z = max([n.nodes.z.max() for n in skdata.itertuples()] + 
                [n.connectors.z.max() for n in skdata.itertuples()] + 
                [n.points.z.max() for n in dotprops.itertuples()] + 
                [ np.array(volumes_data[v]['verts']).max(axis=0)[2] for v in volumes_data ])

    module_logger.info('Preparing neurons for plotting...')
    # First downsample neurons
    if downsampling > 1 and not connectors_only and not skdata.empty:
        module_logger.info('Downsampling neurons...')
        morpho.module_logger.setLevel('ERROR')
        skdata.downsample( downsampling )
        morpho.module_logger.setLevel('INFO')
        module_logger.info('Downsampling finished.')
    elif skdata.shape[0] > 100:
        module_logger.info(
            'Large dataset detected. Consider using the <downsampling> parameter if you encounter bad performance.')

    if backend == 'plotly':
        return _plot3d_plotly()
    else:
        return _plot3d_vispy()


def plot_network(x, *args, **kwargs):
    """ Uses python-igraph and plotly to generate a network plot    

    Parameters
    ----------
    x
                      Neurons as single or list of either:

                      1. skeleton IDs (int or str)
                      2. neuron name (str, exact match)
                      3. annotation: e.g. ``'annotation:PN right'``
                      4. CatmaidNeuron or CatmaidNeuronList object
                      5. pandas.DataFrame containing an adjacency matrix., 
                         e.g. from :funct:`~pymaid.create_adjacency_matrix`
                      6. iGraph representation of the network    
    remote_instance : CATMAID Instance, optional
                      Need to pass this too if you are providing only skids.
    layout :          string, default = 'fr' -> Fruchterman-Reingold
                      See http://igraph.org/python/doc/tutorial/tutorial.html
                      for available layouts.
    syn_cutoff :      int, default=False
                      If provided, connections will be maxed at this value.
    syn_threshold :   int, default=0
                      Edges with less connections are ignored.
    groups :          dict
                      Use to group neurons. Format:
                      ``{ 'Group A' : [skid1, skid2, ..], }``
    colormap :        {str, tuple, dict }
                | Set to 'random' (default) to assign random colors to neurons
                | Use single tuple to assign the same color to all neurons:
                | e.g. ``( (220,10,50) )``
                | Use dict to assign rgb colors to individual neurons:
                | e.g. ``{ neuron1 : (200,200,0), .. }``
    label_nodes :     bool, default=True
                      Plot neuron labels.
    label_edges :     bool, default=True
                      Plot edge labels.
    width :           int, default=800
    height :          int, default=800
                      Figure width and height.
    node_hover_text : dict
                      Provide custom hover text for neurons:
                      ``{ neuron1 : 'hover text', .. }``
    node_size :       {int, dict}
                      | Use int to set node size once.
                      | Use dict to set size for individual nodes:
                      | ``{ neuron1 : 20, neuron2 : 5,  .. }``

    Returns
    -------
    fig : plotly dict
       Use for example ``plotly.offline.plot(fig, filename='plot.html')`` to
       generate html file and open it webbrowser

    """

    remote_instance = kwargs.get('remote_instance', None)
    layout = kwargs.get('layout', 'fr')

    syn_cutoff = kwargs.get('syn_cutoff', None)
    syn_threshold = kwargs.get('syn_threshold', 1)
    groups = kwargs.get('groups', [])
    colormap = kwargs.get('colormap', 'random')

    label_nodes = kwargs.get('label_nodes', True)
    label_edges = kwargs.get('label_edges', True)

    node_labels = kwargs.get('node_labels', [])
    node_hover_text = kwargs.get('node_hover_text', [])
    node_size = kwargs.get('node_size', 20)

    width = kwargs.get('width', 800)
    height = kwargs.get('height', 800)    

    if remote_instance is None:
        if isinstance(x, core.CatmaidNeuronList) or isinstance(x, core.CatmaidNeuron):
            remote_instance = x._remote_instance
        elif 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if not isinstance(x, (igraph.Graph,pd.DataFrame) ):
        x = pymaid.eval_skids(x, remote_instance=remote_instance)
        adj_mat = clustmaid.create_adjacency_matrix(x,
                                                    x,
                                                    remote_instance,
                                                    syn_cutoff=syn_cutoff,
                                                    syn_threshold=syn_threshold,
                                                    row_groups=groups,  # This is where the magic happens
                                                    col_groups=groups  # This is where the magic happens
                                                    )
    elif isinstance(x, pd.DataFrame ):
        adj_mat = x

    if not isinstance(x, igraph.Graph):
        # Generate igraph object and apply layout
        g = igraph_catmaid.matrix2graph(
            adj_mat, syn_threshold=syn_threshold, syn_cutoff=syn_cutoff)
    else:
        g = x

    try:
        layout = g.layout(layout, weights=g.es['weight'])
    except:
        layout = g.layout(layout)
    pos = layout.coords

    # Prepare colors
    if type(colormap) == type(dict()):
        colors = colormap
        # Give grey color to neurons that are not in colormap
        colors.update({v['label']: (.5, .5, .5)
                       for i, v in enumerate(g.vs) if v['label'] not in colormap})
    elif colormap == 'random':
        c = _random_colors(len(g.vs), color_space='RGB', color_range=255)
        colors = {v['label']: c[i] for i, v in enumerate(g.vs)}
    elif type(colormap) == type(tuple()):
        colors = {v['label']: colormap for i, v in enumerate(g.vs)}
    else:
        module_logger.error(
            'I dont understand the colors you have provided. Please, see help(plot.plot_network).')
        return None

    edges = []
    annotations = []
    for e in g.es:
        e_width = 2 + 5 * round(e['weight']) / max(g.es['weight'])        

        edges.append(
            go.Scatter(dict(
                x=[pos[e.source][0], pos[e.target][0], None],
                y=[pos[e.source][1], pos[e.target][1], None],
                mode='lines',
                hoverinfo='text',
                text=str(e['weight']),
                line=dict(
                    width=e_width,
                    color='rgb(255,0,0)'
                )
            ))
        )

        annotations.append(dict(
            x=pos[e.target][0],
            y=pos[e.target][1],
            xref='x',
            yref='y',
            showarrow=True,
            align='center',
            arrowhead=2,
            arrowsize=.5,
            arrowwidth=e_width,
            arrowcolor='#636363',
            ax=pos[e.source][0],
            ay=pos[e.source][1],
            axref='x',
            ayref='y',
            standoff=10
        ))

        if label_edges:
            center_x = (pos[e.target][0] - pos[e.source]
                        [0]) / 2 + pos[e.source][0]
            center_y = (pos[e.target][1] - pos[e.source]
                        [1]) / 2 + pos[e.source][1]

            if e['weight'] == syn_cutoff:
                t = '%i +' % int(e['weight'])
            else:
                t = str(int(e['weight']))

            annotations.append(dict(
                x=center_x,
                y=center_y,
                xref='x',
                yref='y',
                showarrow=False,
                text=t,
                font=dict(color='rgb(0,0,0)', size=10)
            )

            )

    # Prepare hover text
    if not node_hover_text:
        node_hover_text = {n['label']: n['label'] for n in g.vs}
    else:
        # Make sure all nodes are represented
        node_hover_text.update({n['label']: n['label']
                                for n in g.vs if n['label'] not in node_hover_text})

    # Prepare node sizes
    if type(node_size) == type(dict()):
        n_size = [node_size[n['label']] for n in g.vs]
    else:
        n_size = node_size

    nodes = go.Scatter(dict(
        x=[e[0] for e in pos],
        y=[e[1] for e in pos],
        text=[node_hover_text[n['label']] for n in g.vs],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=n_size,
            color=['rgb' + str(tuple(colors[n['label']])) for n in g.vs]
        )
    ))

    if label_nodes:
        annotations += [dict(
            x=e[0],
            y=e[1],
            xref='x',
            yref='y',
            text=g.vs[i]['label'],
            showarrow=False,
            font=dict(color='rgb(0,0,0)', size=12)
        )
            for i, e in enumerate(pos)]

    layout = dict(
        width=width,
        height=height,
        showlegend=False,
        annotations=annotations,
        xaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
        ),
        hovermode='closest'
    )

    data = go.Data([nodes])

    fig = go.Figure(data=data, layout=layout)

    module_logger.info(
        'Done! Use e.g. plotly.offline.plot(fig, filename="network_plot.html") to plot.')

    return fig

def _parse_objects(x,remote_instance=None):
    """ Helper class to extract objects for plotting
    """

    if not isinstance(x, list):
        x = [x]

    # Check for skeleton IDs
    skids = []
    for ob in x:
        try:
            skids.append(int(ob))
        except:            
            pass

    # Collect neuron objects and collate to single Neuronlist
    neuron_obj = [ob for ob in x if isinstance(
        ob, (pd.DataFrame, pd.Series, core.CatmaidNeuron, core.CatmaidNeuronList)) 
        and not isinstance(ob, (core.Dotprops, core.Volume))] # dotprops and volumes are instances of pd.DataFrames
    
    skdata = core.CatmaidNeuronList( neuron_obj, make_copy=False)
    

    # Collect dotprops
    dotprops = [ob for ob in x if isinstance(ob,core.Dotprops)]

    if len(dotprops) == 1:
        dotprops = dotprops[0]
    elif len(dotprops) == 0:
        dotprops = pd.DataFrame()
    elif len(dotprops) > 1:
        dotprops = pd.concat(dotprops)

    # Collect and parse volumes
    volumes = [ ob for ob in x if isinstance(ob, (core.Volume, str) ) ]

    return skids, skdata, dotprops, volumes
