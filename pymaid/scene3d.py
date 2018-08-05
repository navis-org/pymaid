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


""" This module contains classes for 3d visualization using vispy.
"""

# TO-DO:
# - keyboard shortcuts to cycle back and forth through neurons
# - DONE set_color method (takes colormap as input)
# - method and shortcut for screenshot (generic filename for shortcut)
# - animate method: makes camera rotate?
# - CANCELLED grey, transparent background for legend
# - DONE logging
# - DONE modifier keys for selection (shift)
# - how to deal with duplicate skeleton IDs? Use id() or hex(id())?
#   -> would have to link somas & connectors to that ID (set as parent)
# - dragging selection (ctrl+shift?) - see gist
# - DONE show shortcuts at bottom in overlay
# - function to show/hide connectors (if available)
# - crosshair for picking? shows on_mouse_move with modifier key
# -> could snap to closest position on given neuron?
# -> for line visuals, `.pos` contains all points of that visual
# - make ctrl-click display a marker at given position
# - keyboard shortcut to toggle volumes

import uuid

import vispy as vp
import numpy as np
import scipy
import seaborn as sns
import png

import matplotlib.colors as mcl

from pymaid import utils, plotting, fetch, config

__all__ = ['Viewer']

logger = config.logger


class Viewer:
    """
    Vispy 3D viewer.

    Parameters
    ----------
    picking :   bool, default = False
                If ``True``, allow selecting neurons by shift-clicking on
                neurons and placing a 3D cursor via control-click (for OSX:
                command-click).
    **kwargs
              Keyword arguments passed to ``vispy.scene.SceneCanvas``.

    Attributes
    ----------
    picking :       bool,
                    Set to ``True`` to allow picking via shift-clicking.
    selected :      np.array
                    List of currently selected neurons. Can also be used to
                    set the selection.
    show_legend :   bool
                    Set to ``True`` or press ``L`` to show legend. This may
                    impact performance.
    legend_font_size : int
                    Font size for legend.

    Examples
    --------
    This viewer is what :func:`pymaid.plot3d` uses when ``backend='vispy'``.
    Instead of :func:`pymaid.plot3d` we can interact with the viewer directly:

    >>> # Open a 3D viewer
    >>> v = pymaid.Viewer()
    >>> # Get and add neurons
    >>> nl = pymaid.get_neuron('annotation:glomerulus DA1')
    >>> v.add(nl)
    >>> # Colorize
    >>> v.colorize()
    >>> # Assign specific colors
    >>> v.set_colors({nl[0].skeleton_id: (1, 0, 0)})
    >>> # Clear the canvas
    >>> v.clear()    

    """
    def __init__(self, picking=False, **kwargs):
        # Update some defaults as necessary
        defaults = dict(keys=None,
                        show=True,
                        bgcolor='white')
        defaults.update(kwargs)

        # Generate canvas
        self.canvas = vp.scene.SceneCanvas(**defaults)

        # Add and setup 3d view
        self.view3d = self.canvas.central_widget.add_view()
        self.camera3d = vp.scene.ArcballCamera()
        self.view3d.camera = self.camera3d

        # Add permanent overlays
        self.overlay = self._draw_overlay()

        self.canvas.unfreeze()
        self.canvas._overlay = self.overlay
        self.canvas._view3d = self.view3d
        self.canvas._wrapper = self
        self.canvas.freeze()

        # Add picking functionality
        if picking:
            self.picking = True
        else:
            self.picking = False

        # Add keyboard shortcuts
        self.canvas.connect(on_key_press)

        # Add resize control to keep overlay in position
        self.canvas.connect(on_resize)

        # Legend settings
        self.__show_legend = False
        self.__selected = []
        self._cycle_index = -1
        self.__legend_font_size = 7

        # Color to use when selecting neurons
        self.highlight_color = (1, .9, .6)

        # Keep track of initial camera position
        self._camera_default = self.view3d.camera.get_state()

        # Cycle mode can be 'hide' or 'alpha'
        self._cycle_mode = 'alpha'

        # Cursors
        self._cursor = None
        self._picking_radius = 20

    def _draw_overlay(self):
        overlay = vp.scene.widgets.ViewBox(parent=self.canvas.scene)
        self.view3d.add_widget(overlay)

        """
        # Legend title
        t = vp.scene.visuals.Text('Legend', pos=(10,10),
                                  anchor_x='left', name='permanent',
                                  parent=overlay,
                                  color=(0,0,0), font_size=9)
        """

        # Define shortcuts here: key -> description
        self._available_shortcuts = {

            'O': 'toggle overlay',
            'L': 'toggle legend',
            'shift+click': 'select neuron',
            'D': 'deselect all',
            'Q/W': 'cycle neurons',
            'U': 'unhide all',
            'F': 'show/hide FPS',

        }

        # Keyboard shortcuts
        shorts_text = 'SHORTCUTS | ' + \
            ' '.join(['<{0}> {1} |'.format(k, v)
                      for k, v in self._available_shortcuts.items()])
        self._shortcuts = vp.scene.visuals.Text(shorts_text,
                                                pos=(10, overlay.size[1]),
                                                anchor_x='left',
                                                anchor_y='bottom',
                                                name='permanent',
                                                parent=overlay,
                                                color=(0, 0, 0), font_size=6)

        # FPS (hidden at start)
        self._fps_text = vp.scene.visuals.Text(
            'FPS',
            pos=(overlay.size[0] - 10, 10),
            anchor_x='right', anchor_y='top',
            name='permanent',
            parent=overlay,
            color=(0, 0, 0), font_size=6)
        self._fps_text.visible = False

        return overlay

    @property
    def show_legend(self):
        """ Set to ``True`` to hide neuron legend."""
        return self.__show_legend

    @show_legend.setter
    def show_legend(self, v):
        if not isinstance(v, bool):
            raise TypeError('Need boolean, got "{}"'.format(type(v)))

        if v != self.show_legend:
            self.__show_legend = v
            # Make sure changes take effect
            self.update_legend()

    @property
    def legend_font_size(self):
        """ Change legend's font size."""
        return self.__legend_font_size

    @legend_font_size.setter
    def legend_font_size(self, val):
        self.__legend_font_size = val
        self.update_legend()

    @property
    def picking(self):
        """ Set to ``True`` to allow picking."""
        return self.__picking

    @picking.setter
    def picking(self, v):
        if not isinstance(v, bool):
            raise TypeError('Need bool, got {}'.format(type(v)))

        self.__picking = v

        if self.picking:
            self.canvas.connect(on_mouse_press)
        else:
            self.canvas.events.mouse_press.disconnect(on_mouse_press)

    @property
    def visible(self):
        """ Returns skeleton IDs of currently selected visible. """
        return [s for s in self.neurons if self.neurons[s][0].visible]

    @property
    def selected(self):
        """ Skeleton IDs of currently selected neurons. """
        return self.__selected

    @selected.setter
    def selected(self, val):
        skids = utils.eval_skids(val)

        if not isinstance(skids, np.ndarray):
            skids = np.array(skids)

        neurons = self.neurons

        logger.debug('{0} neurons selected ({1} previously)'.format(
            len(skids), len(self.selected)))

        # First un-highlight neurons no more selected
        for s in [s for s in self.__selected if s not in set(skids)]:
            for v in neurons[s]:
                if isinstance(v, vp.scene.visuals.Mesh):
                    v.color = v._stored_color
                else:
                    v.set_data(color=v._stored_color)

        # Highlight new additions
        for s in skids:
            if s not in self.__selected:
                for v in neurons[s]:
                    # Keep track of old colour
                    v.unfreeze()
                    v._stored_color = v.color
                    v.freeze()
                    if isinstance(v, vp.scene.visuals.Mesh):
                        v.color = self.highlight_color
                    else:
                        v.set_data(color=self.highlight_color)

        self.__selected = skids

        self.update_legend()

    @property
    def visuals(self):
        """ Returns list of all 3D visuals on this canvas. """
        return [v for v in self.view3d.children[0].children if isinstance(v, vp.scene.visuals.VisualNode)]

    @property
    def neurons(self):
        """ Returns visible and invisible neuron visuals currently on the canvas.

        Returns
        -------
        dict
                    {skeleton_ID : [ neurites, soma ]}
        """
        # Collect neuron objects
        neuron_obj = [c for c in self.visuals if 'neuron' in getattr(
            c, '_object_type', '')]

        # Collect skeleton IDs
        skids = set([ob._skeleton_id for ob in neuron_obj])

        # Map visuals to unique skids
        return {s: [ob for ob in neuron_obj if ob._skeleton_id == s] for s in skids}

    @property
    def _neuron_obj(self):
        """ Returns neurons by their object id. """
        # Collect neuron objects
        neuron_obj = [c for c in self.visuals if 'neuron' in getattr(
            c, '_object_type', '')]

        # Collect skeleton IDs
        obj_ids = set([ob._object_id for ob in neuron_obj])

        # Map visuals to unique skids
        return {s: [ob for ob in neuron_obj if ob._object_id == s] for s in obj_ids}

    def clear_legend(self):
        """ Clear legend. """
        # Clear legend except for title
        for l in [l for l in self.overlay.children if isinstance(l, vp.scene.visuals.Text) and l.name != 'permanent']:
            l.parent = None

    def clear(self):
        """ Clear canvas. """
        for v in self.visuals:
            v.parent = None

        self.clear_legend()

    def update_legend(self):
        """ Update legend. """

        # Get existing labels
        labels = {l._object_id: l for l in self.overlay.children if getattr(
            l, '_object_id', None)}

        # If legend is not meant to be shown, make sure everything is hidden and return
        if not self.show_legend:
            for v in labels.values():
                if v.visible:
                    v.visible = False
            return
        else:
            for v in labels.values():
                if not v.visible:
                    v.visible = True

        # Labels to be removed
        to_remove = [s for s in labels if s not in self._neuron_obj]
        for s in to_remove:
            labels[s].parent = None

        # Generate new labels
        to_add = [s for s in self._neuron_obj if s not in labels]
        for s in to_add:
            l = vp.scene.visuals.Text('{0} - #{1}'.format(self._neuron_obj[s][0]._neuron_name,
                                                          self._neuron_obj[s][0]._skeleton_id),
                                      anchor_x='left',
                                      anchor_y='top',
                                      parent=self.overlay,
                                      font_size=self.legend_font_size)
            l.interactive = True
            l.unfreeze()
            l._object_id = s
            l._skeleton_id = self._neuron_obj[s][0]._skeleton_id
            l.freeze()

        # Position and color labels
        labels = {l._object_id: l for l in self.overlay.children if getattr(
            l, '_object_id', None)}
        for i, s in enumerate(sorted(self._neuron_obj)):
            if self._neuron_obj[s][0].visible:
                color = self._neuron_obj[s][0].color
            else:
                color = (.3, .3, .3)

            offset = 10 * (self.legend_font_size / 7)

            labels[s].pos = (10, offset * (i + 1))
            labels[s].color = color
            labels[s].font_size = self.legend_font_size

    def toggle_overlay(self):
        """ Toggle legend on and off. """
        self.overlay.visible = self.overlay.visible == False

    def center_camera(self):
        """ Center camera on visuals. """
        if not self.visuals:
            return

        xbounds = np.array([v.bounds(0) for v in self.visuals]).flatten()
        ybounds = np.array([v.bounds(1) for v in self.visuals]).flatten()
        zbounds = np.array([v.bounds(2) for v in self.visuals]).flatten()

        self.camera3d.set_range((xbounds.min(), xbounds.max()),
                                (ybounds.min(), ybounds.max()),
                                (zbounds.min(), zbounds.max()))

    def add(self, x, center=True, clear=False, **kwargs):
        """ Add objects to canvas.

        Parameters
        ----------
        x :         skeleton IDs | CatmaidNeuron/List | Dotprops | Volumes | Points | vispy Visuals
                    Object(s) to add to the canvas.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.
        clear :     bool, optional
                    If True, clear canvas before adding new objects.
        **kwargs
                    Keyword arguments passed when generating visuals. See
                    :func:`~pymaid.plot3d` for options.

        Returns
        -------
        None
        """

        skids, skdata, dotprops, volumes, points, visuals = utils._parse_objects(x)

        if skids:
            visuals += plotting._neuron2vispy(fetch.get_neurons(skids),
                                              **kwargs)
        if skdata:
            visuals += plotting._neuron2vispy(skdata, **kwargs)
        if not dotprops.empty:
            visuals += plotting._dp2vispy(dotprops, **kwargs)
        if volumes:
            visuals += plotting._volume2vispy(volumes, **kwargs)
        if points:
            visuals += plotting._points2vispy(points,
                                              **kwargs.get('scatter_kws', {}))

        if not visuals:
            raise ValueError('No visuals generated.')

        if clear:
            self.clear()

        for v in visuals:
            self.view3d.add(v)

        if center:
            self.center_camera()

        # self.update_legend()

    def show(self):
        """ Show viewer. """
        self.canvas.show()

    def close(self):
        """ Close viewer. """
        if self == globals().get('viewer', None):
            globals().pop('viewer')
        self.canvas.close()

    def hide_neurons(self, n):
        """ Hide given neuron(s). """
        skids = utils.eval_skids(n)

        neurons = self.neurons

        for s in skids:
            for v in neurons[s]:
                if v.visible:
                    v.visible = False

        self.update_legend()

    def unhide_neurons(self, n=None, check_alpha=False):
        """ Unhide given neuron(s). Use ``n`` to unhide specific neurons. """
        if not isinstance(n, type(None)):
            skids = utils.eval_skids(n)
        else:
            skids = list(self.neurons.keys())

        neurons = self.neurons

        for s in skids:
            for v in neurons[s]:
                if not v.visible:
                    v.visible = True
            if check_alpha:
                c = list(mcl.to_rgba(neurons[s][0].color))
                if c[3] != 1:
                    c[3] = 1
                    self.set_colors({s: c})

        self.update_legend()

    def toggle_neurons(self, n):
        """ Toggle neuron(s) visibility. """

        n = utils._make_iterable(n)

        if False not in [isinstance(u, uuid.UUID) for u in n]:
            obj = self._neuron_obj
        else:
            n = utils.eval_skids(n)
            obj = self.neurons

        for s in n:
            for v in obj[s]:
                v.visible = v.visible == False

        self.update_legend()

    def toggle_select(self, n):
        """ Toggle selected of given neuron. """
        skids = utils.eval_skids(n)

        neurons = self.neurons

        for s in skids:
            if self.selected != s:
                self.selected = s
                for v in neurons[s]:
                    self._selected_color = v.color
                    v.set_data(color=self.highlight_color)
            else:
                self.selected = None
                for v in neurons[s]:
                    v.set_data(color=self._selected_color)

        self.update_legend()

    def set_colors(self, c, include_connectors=False):
        """ Set neuron color.

        Parameters
        ----------
        c :      tuple | dict
                 RGB color(s) to apply. Values must be 0-1. Accepted:
                   1. Tuple of single color. Applied to all visible neurons.
                   2. Dictionary mapping skeleton IDs to colors.

        """

        if isinstance(c, (tuple, list, np.ndarray, str)):
            cmap = {s: c for s in self.neurons}
        elif isinstance(c, dict):
            cmap = c
        else:
            raise TypeError(
                'Unable to use colors of type "{}"'.format(type(c)))

        for n in self.neurons:
            if n in cmap:
                for v in self.neurons[n]:
                    if v._neuron_part == 'connectors' and not include_connectors:
                        continue
                    if isinstance(v, vp.scene.visuals.Mesh):
                        v.color = mcl.to_rgba(cmap[n])
                    else:
                        v.set_data(color=mcl.to_rgba(cmap[n]))

        self.update_legend()

    def colorize(self, palette='hls', include_connectors=False):
        """ Colorize neurons using a seaborn color palette."""

        colors = sns.color_palette(palette, len(self.neurons))
        cmap = {s: colors[i] for i, s in enumerate(self.neurons)}

        self.set_colors(cmap, include_connectors=include_connectors)

    def _cycle_neurons(self, increment):
        """ Cycle through neurons. """
        self._cycle_index += increment

        if self._cycle_index < 0:
            self._cycle_index = len(self.neurons) - 1
        elif self._cycle_index > len(self.neurons) - 1:
            self._cycle_index = 0

        neurons_sorted = sorted(self.neurons.keys())

        to_hide = [n for i, n in enumerate(
            neurons_sorted) if i != self._cycle_index]
        to_show = [neurons_sorted[self._cycle_index]]

        if self._cycle_mode == 'hide':
            self.hide_neurons(to_hide)
            self.unhide_neurons(to_show)
        elif self._cycle_mode == 'alpha':
            # Get current colors
            new_cmap = {}
            for n in self.neurons:
                this_c = list(self.neurons[n][0].color)
                # Make sure colors are (r,g,b,a)
                if len(this_c) < 4:
                    this_c = np.append(this_c, 1).astype(float)
                # If neuron needs to be hidden, add to cmap
                if n in to_hide and this_c[3] != .1:
                    this_c[3] = .1
                    new_cmap[n] = this_c
                elif n in to_show and this_c[3] != 1:
                    this_c[3] = 1
                    new_cmap[n] = this_c
            self.set_colors(new_cmap)
        else:
            raise ValueError(
                'Unknown cycle mode "{}". Use "hide" or "alpha"!'.format(self._cycle_mode))

    def _draw_fps(self, fps):
        """ Callback for ``canvas.measure_fps``. """
        self._fps_text.text = '{:.2f} FPS'.format(fps)

    def _toggle_fps(self):
        """ Switch FPS measurement on and off. """
        if not self._fps_text.visible:
            self.canvas.measure_fps(1, self._draw_fps)
            self._fps_text.visible = True
        else:
            self.canvas.measure_fps(1, None)
            self._fps_text.visible = False

    def _snap_cursor(self, pos, visual):
        """ Snap cursor to clostest vertex of visual."""
        if not getattr(self, '_cursor', None):
            self._cursor = vp.scene.visuals.Arrow(pos=np.array([(0, 0, 0), (1000, 0, 0)]),
                                                  color=(1, 0, 0, 1),
                                                  arrow_color=(1, 0, 0, 1),
                                                  arrow_size=10,
                                                  arrows=np.array([[800, 0, 0, 1000, 0, 0]]))

        if not self._cursor.parent:
            self.add(self._cursor, center=False)

        # Snap cursor to closest vertex
        if isinstance(visual, vp.scene.visuals.Line):
            verts = visual.pos
        elif isinstance(visual, vp.scene.visuals.Mesh):
            verts = visual.mesh_data.get_vertices()

        # Find the closest vertex to this position
        tree = scipy.spatial.cKDTree(verts)
        dist, ix = tree.query(pos[:-1])
        snap_pos = verts[ix]

        # Generate arrow coords
        snap_pos = np.array(snap_pos)
        start = snap_pos - 10000
        arrows = np.array([np.append(snap_pos - 10, snap_pos)])

        self._cursor.set_data(pos=np.array([start, snap_pos]),
                              arrows=arrows)

    def screenshot(self, filename='screenshot.png', pixel_scale=2,
                   alpha=True, hide_overlay=True):
        """ Save a screenshot of this viewer.

        Parameters
        ----------
        filename :      str, optional
                        Filename to save to.
        pixel_scale :   int, optional
                        Factor by which to scale canvas. Determines image
                        dimensions.
        alpha :         bool, optional
                        If True, will export transparent background.
        hide_overlay :  bool, optional
                        If True, will hide overlay for screenshot.
        """

        if alpha:
            bgcolor = list(self.canvas.bgcolor.rgb) + [0]
        else:
            bgcolor = list(self.canvas.bgcolor.rgb)

        region = (0, 0, self.canvas.size[0], self.canvas.size[1])
        size = tuple(np.array(self.canvas.size) * pixel_scale)

        if hide_overlay:
            prev_state = self.overlay.visible
            self.overlay.visible = False

        m = self.canvas.render(region=region, size=size, bgcolor=bgcolor)

        if hide_overlay:
            self.overlay.visible = prev_state

        im = png.from_array(m, mode='RGBA')
        im.save(filename)

def on_mouse_press(event):
    """ Manage picking on canvas. """
    canvas = event.source
    viewer = canvas._wrapper

    logger.debug('Mouse press at {0}: {1}'.format(
        event.pos, canvas.visuals_at(event.pos)))

    if event.modifiers:
        logger.debug('Modifiers found: {0}'.format(event.modifiers))

    # Iterate over visuals in this canvas at cursor position
    for v in canvas.visuals_at(event.pos, viewer._picking_radius):
        # Skip views
        if isinstance(v, vp.scene.widgets.ViewBox):
            continue
        # If legend entry, toggle visibility
        elif isinstance(v, vp.scene.visuals.Text):
            viewer.toggle_neurons(v._object_id)
            break
        # If shift modifier, add to/remove from current selection
        elif isinstance(v, vp.scene.visuals.VisualNode) and getattr(v, '_skeleton_id', None) and 'Shift' in [key.name for key in event.modifiers]:
            if v._skeleton_id not in set(viewer.selected):
                viewer.selected = np.append(viewer.selected, v._skeleton_id)
            else:
                viewer.selected = viewer.selected[viewer.selected !=
                                                  v._skeleton_id]
            break

        if 'Control' in [key.name for key in event.modifiers]:
            tr = canvas._view3d.node_transform(v)
            co = tr.map(event.pos)
            viewer._snap_cursor(co, v)
            logger.debug('World coordinates: {}'.format(co))
            logger.debug('URL: {}'.format(
                fetch.url_to_coordinates(co, 5)))
            break


def on_key_press(event):
    """ Manage keyboard shortcuts for canvas. """

    canvas = event.source
    viewer = canvas._wrapper

    if event.text.lower() == 'o':
        viewer.toggle_overlay()
    elif event.text.lower() == 'l':
        viewer.show_legend = viewer.show_legend == False
    elif event.text.lower() == 'd':
        viewer.selected = []
    elif event.text.lower() == 'q':
        viewer._cycle_neurons(-1)
    elif event.text.lower() == 'w':
        viewer._cycle_neurons(1)
    elif event.text.lower() == 'u':
        viewer.unhide_neurons(check_alpha=True)
    elif event.text.lower() == 'f':
        viewer._toggle_fps()


def on_resize(event):
    """ Keep overlay in place upon resize. """
    viewer = event.source._wrapper
    viewer._shortcuts.pos = (10, event.size[1])

    viewer._fps_text.pos = (event.size[0] - 10, 10)
