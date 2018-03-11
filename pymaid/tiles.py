#    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
#    Copyright (C) 2017 Philipp Schlegel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along

import io
import imageio
import urllib.request

import sys
import os
import gc

from pymaid import fetch, core, utils

from tqdm import trange, tqdm
if utils.is_jupyter():
    from tqdm import tqdm_notebook, tnrange
    tqdm = tqdm_notebook
    trange = tnrange

import pandas as pd
import numpy as np
import scipy
import math

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import time
import urllib

import threading

import logging

# Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if len( module_logger.handlers ) == 0:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
                '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)

def crop_neuron(x, output, dimensions=(1000,1000), interpolate_z_res=40, remote_instance=None):
    """ Crops EM tiles following a neuron's segments.

    Parameters
    ----------
    x :                  {core.CatmaidNeuron}
    output :             str
                         File or folder.
    dimensions :         tuple of int, optional
                         Dimensions of square to cut out in nanometers.
    interpolate_z_res :  {int, None}, optional
                         If not none, will interpolate in Z direction to given
                         resolution. Use this to interpolate virtual nodes.
    remote_instance :    pymaid.CatmaidInstance, optional
    """
    if isinstance(x, core.CatmaidNeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.CatmaidNeuron ):
        raise TypeError('Need a single CatmaidNeuron, got "{0}".'.type(x))

    if len(dimensions) != 2:
        raise ValueError('Need two dimensions, got {0}'.format(len(dimensions)))

    # Evalutate remote instance
    remote_instance = fetch._eval_remote_instance(remote_instance)

    # Prepare treenode table to be indexed by treenode_id
    this_tn = x.nodes.set_index('treenode_id')

    # Iterate over neuron's segments
    bboxes = []
    for seg in x.segments:
        # Get treenode coordinates
        center_coords = this_tn.loc[ seg, ['x','y','z'] ].values

        # If a z resolution for interpolation is given, interpolate virtual nodes
        if interpolate_z_res:
            interp_coords = center_coords[0:1]
            # Go over all treenode -> parent pairs
            for i, (co, next_co) in enumerate( zip(center_coords[:-1], center_coords[1:]) ):
                # If nodes are more than interpolate_z_res nm away from another
                if math.fabs(co[2] - next_co[2]) >= (2*interpolate_z_res):
                    # Get steps we would expect to be there
                    steps = int( math.fabs(co[2] - next_co[2]) / interpolate_z_res )

                    # If we're going anterior, we need to inverse step size
                    if co[2] < next_co[2]:
                        step_size = interpolate_z_res
                    else:
                        step_size = -interpolate_z_res

                    # Interpolate coordinates
                    new_co = [ ( co[0] + int((next_co[0]-co[0])/steps * (i+1)),
                                 co[1] + int((next_co[1]-co[1])/steps * (i+1)),
                                 z  )
                                 for i,z in enumerate(range( co[2] + step_size, next_co[2], step_size )) ]

                    # Track new coordinates
                    interp_coords = np.append( interp_coords, new_co, axis=0 )
                # Add next coordinate
                interp_coords = np.append( interp_coords, [ next_co ], axis=0 )
            # Use interpolated coords
            center_coords = interp_coords

        # Turn into bounding boxes: left, right, top, bottom, z
        bbox = np.array( [[ co[0]-dimensions[0]/2,
                              co[0]+dimensions[0]/2,
                              co[1]-dimensions[1]/2,
                              co[1]+dimensions[1]/2,
                              co[2] ]
                              for co in center_coords]
                        ).astype(int)
        bboxes += list( bbox )

    # Generate tile job
    job = LoadTiles( bboxes,
                     zoom_level=0,
                     coords='NM',
                     remote_instance=remote_instance )

    #job.generate_img()

    return job

class LoadTiles:
    """ Loads tiles from CATMAID and returns stitched image.

    Important
    ---------
    Loading lots of tiles is memory intensive. A single 100x100 pixel image
    already requires 80Kb, 1000x1000 8Mb and so on.

    Parameters
    ----------
    bbox :        {list, np.ndarray}
                  Window to crop: [left, right, top, bottom, z1, z2]
                  If z2 is ommitted, z2 = z1.
    stack_id :    str, optional
                  Image stack to use. If not provided, will try finding the
                  fastest server on the project.
    zoom_level :  int, optional
                  Zoom level
    coords :      {'NM','PIXEL'}, optional
                  Format of t/l/w/h/z. Can be pixel or nanometers.
    mem_lim :     int, optional
                  Memory limit in megabytes for loading tiles. This restricts
                  the number of tiles that can be simultaneously loaded into
                  memory.
    """

    # TODOs
    # -----
    # 1. Check for available image mirror automatically (make stack_mirror and stack_id superfluous) - DONE
    # 2. Test using matplotlib instead (would allow storing nodes and scalebar as SVG) - DONE
    # 3. Code clean up
    # 4. Add second mode that loads sections sequentially, saves them and discards tiles: slower but memory efficient

    def __init__(self, bbox, zoom_level=0, coords='NM', mem_lim=4000, remote_instance=None):
        """ Initialise class.
        """
        if coords not in ['PIXEL','NM']:
            raise ValueError('Coordinates need to be "PIXEL" or "NM".')

        # Convert single bbox to multiple bounding boxes
        if isinstance( bbox, np.ndarray ):
            if bbox.ndim == 1:
                self.bboxes = [ bbox ]
            elif bbox.ndim == 2:
                self.bboxes = bbox
            else:
                raise ValueError('Unable to interpret bounding box with {0} dimensions'.format( bbox.ndim) )
        elif isinstance(bbox, list):
            if any( isinstance( el, (list, np.ndarray) ) for el in bbox ):
                self.bboxes = bbox
            else:
                self.bboxes = [ bbox ]
        else:
            raise TypeError('Bounding box must be list or array, not {0}'.format( type(bbox)) )

        self.remote_instance = fetch._eval_remote_instance(remote_instance)
        self.zoom_level = zoom_level
        self.coords = coords
        self.mem_lim = mem_lim

        self.get_stack_info()

        self.bboxes2imgcoords()

        memory_est = self.estimate_memory()

        module_logger.info('Estimated memory usage for loading all required at once tiles: {0:.2f} Mb'.format(memory_est))

    def estimate_memory(self):
        """ Estimates memory [Mb] consumption of loading all tiles."""

        all_tiles = [  ]
        for j, im in enumerate( self.image_coords ):
            for i, ix_x in enumerate( range( im['tile_left'], im['tile_right'] ) ):
                for k, ix_y in enumerate( range( im['tile_top'], im['tile_bot'] ) ):
                    all_tiles.append( (ix_x, ix_y, im['tile_z'] ) )

        n_tiles = len( set(all_tiles) )

        return (n_tiles * self.bytes_per_tile) / 10 ** 6


    def get_stack_info(self):
        """ Retrieves basic info about image stack and mirrors.
        """

        # Get available stacks for the project
        available_stacks = self.remote_instance.fetch(
                                            self.remote_instance._get_stacks_url()
                                                    )
        stack_id =available_stacks[0]['id']

        # Fetch and store stack info
        info = self.remote_instance.fetch( self.remote_instance._get_stack_info_url(stack_id) )

        self.stack_dimension_x = info['dimension']['x']
        self.stack_dimension_y = info['dimension']['y']
        self.stack_dimension_z = info['dimension']['z']

        self.resolution_x = info['resolution']['x']
        self.resolution_y = info['resolution']['y']
        self.resolution_z = info['resolution']['z']

        # Get fastest img mirror
        self.img_mirror = sorted( info['mirrors'], key = lambda x : test_response_time(x['image_base'], calls=2 ) )[0]
        self.tile_width = self.img_mirror['tile_width']
        self.mirror_url = self.img_mirror['image_base']
        self.file_ext = self.img_mirror['file_extension']

        # Memory size per tile in byte
        self.bytes_per_tile = self.tile_width ** 2 * 8

        module_logger.info('Fastest image mirror: {0}'.format( self.mirror_url) )

    def bboxes2imgcoords(self):
        """ Converts bounding box(es) to coordinates for individual images.
        """

        # This keeps track of images (one per z slice)
        self.image_coords = []

        for bbox in self.bboxes:
            if len(bbox) not in [5,6]:
                raise ValueError('Need 5 or 6 coordinates (top, left, bottom, right, z1, [z2]), got {0}'.format(len(bbox)))

            # If z2 not given, add z2 = z1
            if len(bbox) == 5:
                np.append( bbox, bbox[-1] )

            # Make sure we have left/right, top/bot, z1, z2 in correct order
            left = min ( bbox[0:2] )
            right = max ( bbox[0:2] )
            top = min( bbox[2:4] )
            bottom = max( bbox[2:4] )
            z1 = int( min( bbox[4:6] ) )
            z2 = int( max( bbox[4:6] ) )

            if self.coords == 'NM':
                # Map coordinates to pixels (this already accounts for zoom)
                px_left = self._to_x_index( left )
                px_right = self._to_x_index( right )
                px_top = self._to_y_index( top )
                px_bot = self._to_y_index( bottom )
                px_z1 = self._to_z_index( z1 )
                px_z2 = self._to_z_index( z2 )

                nm_left, nm_right, nm_top, nm_bot, nm_z1, nm_z2 = left, right, top, bottom, z1, z2
            else:
                # Adjust pixel coordinates to zoom level
                px_left = int( left / (2**self.zoom_level) + 0.5)
                px_right = int( right / (2**self.zoom_level) + 0.5)
                px_top = int( top / (2**self.zoom_level) + 0.5)
                px_bot = int( bottom / (2**self.zoom_level) + 0.5)
                px_z1 = int( z1 )
                px_z2 = int( z2 )

                # Turn pixel coordinates into real world coords
                nm_left = int( left * self.resolution_x)
                nm_right = int( right * self.resolution_x)
                nm_top = int( top * self.resolution_y)
                nm_bot = int( bottom * self.resolution_y)
                nm_z1 = int( z1 * self.resolution_z)
                nm_z2 = int( z2 * self.resolution_z)

            # Map to tiles
            tile_left = int( px_left / self.tile_width )
            tile_right = int( px_right / self.tile_width ) + 1
            tile_top = int( px_top / self.tile_width )
            tile_bot = int( px_bot / self.tile_width ) + 1

            # Get borders to crop
            border_left = px_left - ( tile_left * self.tile_width )
            border_right = ( tile_right * self.tile_width ) - px_right
            border_top = px_top - ( tile_top * self.tile_width )
            border_bot = ( tile_bot * self.tile_width ) - px_bot

            # Generate a single entry for each z slice in this bbox
            for px_z, nm_z in zip( range( px_z1, px_z2 + 1 ), np.arange( nm_z1, nm_z2 + self.resolution_z, self.resolution_z)  ):
                # Tile we will have to load for this image
                this_tiles = []
                for i, ix_x in enumerate( range( tile_left, tile_right ) ):
                    for k, ix_y in enumerate( range( tile_top, tile_bot ) ):
                        this_tiles.append( (ix_x, ix_y, px_z ) )

                this_im = dict(
                                # Nanometer coords
                                nm_bot = nm_bot,
                                nm_top = nm_top,
                                nm_left = nm_left,
                                nm_right = nm_right,
                                nm_z = int(nm_z),

                                # Tile indices
                                tile_bot = tile_bot,
                                tile_top = tile_top,
                                tile_left = tile_left,
                                tile_right = tile_right,
                                tile_z = px_z,

                                px_bot = px_bot,
                                px_top = px_top,
                                px_left = px_left,
                                px_right = px_right,
                                px_z = px_z,

                                px_border_top = border_top,
                                px_border_left = border_left,
                                px_border_bot = border_bot,
                                px_border_right = border_right,

                                tiles_to_load = this_tiles,

                                )

                self.image_coords.append( this_im )


    def _get_tiles(self, tiles):
        """ Retrieves all tiles in parallel.

        Parameters
        ----------
        tiles :     {list, np.ndarray}
                    Triplets of x/y/z tile indices. E.g. [ (20,10,400 ), (...) ]
        """
        tiles = list( set(tiles) )

        data = { co : None for co in tiles }
        threads = {}
        threads_closed = []
        for i, coords in enumerate(tiles):
            url = self._get_tile_url( *coords )
            t = _retrieveTileThreaded(url)
            t.start()
            threads[ coords ] = t

        # Initialise progress bar
        pbar = tqdm(total=len(threads),
                    desc='Loading tiles',
                    disable=module_logger.getEffectiveLevel()>=40,
                    leave=False)

        # Save start value of pbar (in case we have an external pbar)
        pbar_start = pbar.n

        while len(threads_closed) != len(threads):
            for t in threads:
                if t in threads_closed:
                    continue
                if not threads[t].is_alive():
                    # Make sure we keep the order
                    data[t] = threads[t].join()
                    threads_closed.append(t)
            time.sleep(1)

            #Update progress bar
            p_delta = len(threads_closed) - (pbar.n - pbar_start)
            pbar.update( p_delta  )

        # Close and clear pbar
        pbar.close()

        # Check if we got all tiles
        for t in threads:
            if t not in threads_closed:
                module_logger.warning(
                    'Did not close thread for tile {0}'.format(t) )

        return data

    def generate_img(self):
        """ Download and stitch tiles."""
        tiles = {}
        max_safe_tiles = int( ( self.mem_lim * 10**6 ) / self.bytes_per_tile )

        # Assemble tiles into the requested images
        images = []
        for l, im in enumerate( tqdm(self.image_coords, 'Stitching') ):
            # Get a list of all tiles that remain to be used and are not currently part of the tiles
            remaining_tiles = [ t for img in self.image_coords[l:] for t in img['tiles_to_load'] ]

            # Clear tiles that we don't need anymore and force garbage collection
            to_delete = [ t for t in tiles if t not in remaining_tiles ]
            for t in to_delete:
                del tiles[t]
            gc.collect()

            # Check if we're still missing tiles
            missing_tiles = [ t for t in im['tiles_to_load'] if t not in tiles ]

            if len(tiles) == 0 or len(missing_tiles) > 0:
                tiles_to_get = [ t for t in remaining_tiles if t not in tiles ][ : max_safe_tiles ] + missing_tiles
            else:
                tiles_to_get = []

            if tiles_to_get:
                # Get missing tiles
                tiles.update( self._get_tiles( tiles_to_get ) )

            # Generate empty array
            im_dim = np.array([
                       math.fabs( (im['tile_bot'] - im['tile_top']) * self.tile_width),
                       math.fabs( (im['tile_right'] - im['tile_left']) * self.tile_width)]).astype(int)
            img = np.zeros(im_dim)

            # Fill array
            for i, ix_x in enumerate( range( im['tile_left'], im['tile_right'] ) ):
                for k, ix_y in enumerate( range( im['tile_top'], im['tile_bot'] ) ):
                    # Paste this tile onto our canvas
                    img[k*self.tile_width : (k+1)*self.tile_width, i*self.tile_width : (i+1)*self.tile_width] = tiles[ ( ix_x, ix_y, im['tile_z'] ) ]

            # Remove borders and create a copy (otherwise we will not be able to clear the original tile)
            cropped_img = np.array( img[ im['px_border_top'] : -im['px_border_bot'],
                                    im['px_border_left'] : -im['px_border_right']] )

            # Add slice
            images.append(cropped_img)

            # Delete image to free memory (not sure if this does much though)
            del img

        # Clear tile data
        del tiles
        gc.collect()

        # Before we assemble all images into a large stack make sure that
        # all individual images have the same dimensions
        dims = np.vstack( [ im.shape for im in images ])

        min_dims = np.min( dims, axis=0 )

        # Get standard deviation to check if they are all the same
        if sum( np.std( dims, axis=0 ) ) != 0:
            module_logger.warning('Varying image dimensions detected. Cropping everything to the smallest image size: {0}'.format(min_dims))

            # Crop images to the smallest common dimension
            for im in images:
                if im.shape != min_dims:
                    center = im.shape // 2
                    im = im[ center[0] - min_dims[0] : center[0] + min_dims[0],
                             center[1] - min_dims[1] : center[1] + min_dims[1] ]

        self.img = np.dstack(images).astype(int)

    def scalebar(self, ax, size=1000, pos='lower left', label=True, line_kws={}, label_kws={}):
        """ Adds scalebar to image.

        Parameters
        ----------
        size :  int
                Size of scalebar. In NM!
        ax :    matplotlib.axes
        font :  PIL font, optional
                If provided, will write the size below scalebar.
        pos :   {'lowerleft','upperleft','lowerright','upperright'}, optional
                Position of scalebar.
        label : bool, optional
                If True will label scalebar.
        line_kws
                Keyword arguments passed to plt.plot().
        label_kws
                Keyword arguments passed to plt.text()
        """

        positions = { 'lower left': ( self.img.shape[1] * .05, self.img.shape[0] * .95 ),
                      'upper left': ( self.img.shape[1] * .05, self.img.shape[0] * .05 ),
                      'lower right': ( self.img.shape[1] * .95, self.img.shape[0] * .95 ),
                      'upper right': ( self.img.shape[1] * .95, self.img.shape[0] * .05 )}

        if pos not in positions:
            raise ValueError('Wrong position. Please use either {0}'.format(','.join(positions)))

        co = positions[pos]

        ax.plot ( [ co[0], co[0] + size / self.resolution_x ], [co[1],co[1]], **line_kws )

        if label:
            ax.text( co[0] + ((co[0] + size / self.resolution_x) - co[0])/2,
                     co[1] + 10,
                     '{0} nm'.format(size),
                     horizontalalignment='center',
                     verticalalignment='center',
                     **label_kws )

    def render_nodes(self, ax, treenodes=True, connectors=True, slice_ix=None, tn_color='yellow', cn_color='none', tn_ec=None, cn_ec='orange', skid_include=[], cn_include=[], tn_kws={}, cn_kws={}):
        """ Renders nodes onto image.

        Parameters
        ----------
        ax :            matplotlib ax
        slice_x :       int, optional
                        If multi-slice, provide index of slice to label.
        tn_color :      {str, tuple, dict}
                        Map skeleton ID to color.
        cn_color :      {str, tuple, dict}
                        Map connector ID to color.
        ec :            {str, tuple, dict}
                        Edge color.
        skid_include :  list, optional
                        List of skeleton IDs to include.
        cn_include :    list, optional
                        List of connector IDs to include.
        tn_kws :        dict, optional
                        Keywords passed on to matplotlib.pyplot.scatter for
                        treenodes.
        cn_kws :        dict, optional
                        Keywords passed on to matplotlib.pyplot.scatter for
                        connectors.
        """

        if slice_ix == None and len(self.image_coords) == 1:
            slice_ix = 0
        elif slice_ix == None:
            raise ValueError('Please provide index of the slice you want nodes to be rendered for.')

        slice_info = self.image_coords[slice_ix]

        #Get node list
        nodes = fetch.get_nodes_in_volume(
                                           slice_info['nm_left'], slice_info['nm_right'],
                                           slice_info['nm_top'], slice_info['nm_bot'],
                                           slice_info['nm_z'] - self.resolution_z, # get one slice up
                                           slice_info['nm_z'] + self.resolution_z, # and one slice down
                                           remote_instance=self.remote_instance,
                                           coord_format='NM' )

        # Interpolate virtual
        self.nodes = self._make_virtual_nodes(nodes['treenodes'])
        self.connectors = nodes['connectors']

        #Filter to only this Z
        self.nodes = self.nodes[ self.nodes.z == slice_info['nm_z'] ]
        self.connectors = self.connectors[ self.connectors.z == slice_info['nm_z'] ]

        #Filter to fit bounding box
        self.nodes = self.nodes[
                                (self.nodes.x <= slice_info['nm_right']) &
                                 (self.nodes.x >= slice_info['nm_left']) &
                                 (self.nodes.y >= slice_info['nm_top']) &
                                 (self.nodes.y <= slice_info['nm_bot'])
                                 ]
        self.connectors = self.connectors[
                                        (self.connectors.x <= slice_info['nm_right']) &
                                        (self.connectors.x >= slice_info['nm_left']) &
                                        (self.connectors.y >=  slice_info['nm_top']) &
                                        (self.connectors.y <= slice_info['nm_bot'])
                                            ]


        # Filter if provided
        if len(skid_include) > 0:
            self.nodes = self.nodes[ self.nodes.skeleton_id.isin( skid_include ) ]
        if len(cn_include) > 0:
            self.connectors = self.connectors[ self.connectors.skeleton_id.isin( node_include ) ]

        # Calculate pixel coords:
        # 1. Offset
        self.nodes.loc[:,'x'] -= slice_info['nm_left']
        self.nodes.loc[:,'y'] -= slice_info['nm_top']
        self.connectors.loc[:,'x'] -= slice_info['nm_left']
        self.connectors.loc[:,'y'] -= slice_info['nm_top']
        # 2. Turn into pixel coordinates
        self.nodes.loc[:,'x'] /= self.resolution_x
        self.nodes.loc[:,'y'] /= self.resolution_y
        self.connectors.loc[:,'x'] /= self.resolution_x
        self.connectors.loc[:,'y'] /= self.resolution_y
        # 3. Round positions
        self.nodes.loc[:,'x'] = self.nodes.x.astype(int)
        self.nodes.loc[:,'y'] = self.nodes.y.astype(int)
        self.connectors.loc[:,'x'] = self.connectors.x.astype(int)
        self.connectors.loc[:,'y'] = self.connectors.y.astype(int)
        # 4. Set colours
        if isinstance(tn_color, dict):
            default_colour = tn_color.get('default', 'black')
            self.nodes['color'] = [ tn_color.get( str(s), default_colour ) for s in self.nodes.skeleton_id.values ]
        else:
            self.nodes['color'] = [ tn_color for x in range(self.nodes.shape[0])]

        if isinstance(cn_color, dict):
            default_colour = cn_color.get('default', 'black')
            self.connectors['color'] = [ cn_color.get( str(s), default_colour ) for s in self.connectors.connector_id.values ]
        else:
            self.connectors['color'] = [ cn_color for x in range(self.connectors.shape[0])]

        if isinstance(tn_ec, dict):
            self.nodes['ec'] = [ tn_ec.get( str(s), None ) for s in self.nodes.skeleton_id.values ]
        else:
            self.nodes['ec'] = [ tn_ec for x in range(self.nodes.shape[0])]

        if isinstance(cn_ec, dict):
            self.connectors['ec'] = [ cn_ec.get( str(s), None ) for s in self.connectors.connector_id.values ]
        else:
            self.connectors['ec'] = [ cn_ec for x in range(self.connectors.shape[0])]

        if treenodes:
            if tn_ec:
                tn_ec = self.nodes.ec.values
            ax.scatter( self.nodes.x.values, self.nodes.y.values,
                        facecolors=self.nodes.color.values,
                        edgecolors=tn_ec,
                        **tn_kws )

        if connectors:
            if cn_ec:
                cn_ec = self.connectors.ec.values
            ax.scatter( self.connectors.x.values, self.connectors.y.values,
                        facecolors=self.connectors.color.values,
                        edgecolors=cn_ec,
                        **cn_kws )

    def _get_tile_url(self, x, y, z ):
        """ Returns tile url."""
        return '{0}{1}/{2}/{3}/{4}.{5}'.format( self.mirror_url,
                                                self.zoom_level,
                                                z,
                                                y,
                                                x,
                                                self.file_ext )

    def _to_x_index(self, x, enforce_bounds=True ):
        """ Converts a real world position to a x pixel position.
        Also, makes sure the value is in bounds.
        """
        zero_zoom = x / self.resolution_x
        if enforce_bounds:
            zero_zoom = min(max(zero_zoom, 0.0), self.stack_dimension_x - 1.0)
        return int( zero_zoom / (2**self.zoom_level) + 0.5 )

    def _to_y_index(self, y, enforce_bounds=True ):
        """ Converts a real world position to a y pixel position.
        Also, makes sure the value is in bounds.
        """
        zero_zoom = y / self.resolution_y
        if enforce_bounds:
            zero_zoom = min(max(zero_zoom, 0.0), self.stack_dimension_y - 1.0)
        return int( zero_zoom / (2**self.zoom_level) + 0.5 )

    def _to_z_index(self, z, enforce_bounds=True ):
        """ Converts a real world position to a slice/section number.
        Also, makes sure the value is in bounds.
        """
        section = z / self.resolution_z + 0.5
        if enforce_bounds:
            section = min(max(section, 0.0), self.stack_dimension_z - 1.0)
        return int( section )

    def _make_virtual_nodes(self, nodes):
        """ Generates virtual nodes. Currently, this simply adds new nodes
        to the table but does NOT rewire the neurons accordingly!
        """
        # Get nodes that have a parent in our list
        has_parent = nodes[ nodes.parent_id.isin(nodes.treenode_id) ]

        # Get treenode and parent section
        tn_section = has_parent.z.values / self.resolution_z
        pn_section = nodes.set_index('treenode_id').loc[ has_parent.parent_id.values, 'z' ].values / self.resolution_z

        # Get distance in sections
        sec_dist = np.absolute( tn_section - pn_section )

        # Get those that have more than one section in between them
        to_interpolate = has_parent[ sec_dist > 1 ]
        tn_locs = to_interpolate[['x','y','z']].values
        pn_locs = nodes.set_index('treenode_id').loc[ to_interpolate.parent_id, ['x','y','z'] ].values
        distances = sec_dist[ sec_dist > 1 ].astype(int)
        skids = to_interpolate.skeleton_id.values

        virtual_nodes = []
        for i in range( to_interpolate.shape[0] ):
            x_interp = np.round( np.linspace( tn_locs[i][0], pn_locs[i][0], distances[i] + 1 ).astype(int) )
            y_interp = np.round( np.linspace( tn_locs[i][1], pn_locs[i][1], distances[i] + 1 ).astype(int) )
            z_interp = np.round( np.linspace( tn_locs[i][2], pn_locs[i][2], distances[i] + 1 ) )

            virtual_nodes += [ [ x_interp[k], y_interp[k], z_interp[k], skids[i] ] for k in range( 1, len(x_interp)-1 ) ]

        virtual_nodes = pd.DataFrame(virtual_nodes,
                                     columns=['x','y','z','skeleton_id'])

        return pd.concat( [nodes,virtual_nodes], axis=0, ignore_index=True )

    def render_im(self, slider=True, **kwargs):
        """ Shows image slices with a slider."""

        fig, ax = plt.subplots(**kwargs)
        plt.subplots_adjust(bottom=0.25)

        mpl_img = ax.imshow(self.img[:,:,0], cmap='gray')

        if slider:
            axcolor = 'grey'
            axslice = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)

            sslice = Slider(axslice, 'Slice', 1, self.img.shape[2],
                            valinit=0, valfmt='%i')

            def update(val):
                slice_ix = int( round( sslice.val ) )
                sslice.valtext.set_text( str(slice_ix) )
                mpl_img.set_data(self.img[:,:,slice_ix-1])
                fig.canvas.draw_idle()

            sslice.on_changed(update)

        return ax

def test_response_time(url, calls=5):
    """ Returns server response time. If unresponsive returns float("inf") """
    resp_times = []
    for i in range(calls):
        start = time.time()
        try:
            resp = urllib.request.urlopen(url)
            resp_times.append(time.time()-start)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return float('inf')
            if err.code == 401:
                resp_times.append(time.time()-start)
        except:
            return float('inf')

    return np.mean(resp_times)

class _retrieveTileThreaded(threading.Thread):
    """ Class to retrieve a URL by threading
    """

    def __init__(self, url):
        try:
            self.url = url
            threading.Thread.__init__(self)
        except:
            module_logger.error(
                'Failed to initiate thread for ' + self.url)

    def run(self):
        """
        Retrieve data from single url
        """
        with urllib.request.urlopen(self.url) as req:
            self.tile = imageio.imread( req.read() )

        return

    def join(self):
        try:
            threading.Thread.join(self)
            return self.tile
        except:
            module_logger.error(
                'Failed to join thread for ' + self.url)
            return None
