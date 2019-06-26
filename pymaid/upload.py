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


""" This module contains functions to push data to a Catmaid server.
"""

import json
import numbers
import os
import tempfile
import traceback

import requests

import numpy as np
import pandas as pd

import seaborn as sns

from scipy.spatial.distance import cdist

from . import core, utils, morpho, config, cache, fetch, scene3d

__all__ = sorted(['add_annotations', 'remove_annotations',
                  'add_tags', 'delete_tags',
                  'delete_neuron',
                  'rename_neurons',
                  'add_meta_annotations', 'remove_meta_annotations',
                  'upload_neuron', 'upload_volume',
                  'update_radii', 'update_skeleton',
                  'join_skeletons', 'join_nodes',
                  'link_connector'])

# Set up logging
logger = config.logger


@cache.never_cache
def upload_volume(x, name, comments=None, remote_instance=None):
    """ Upload volume/mesh to CatmaidInstance.

    Parameters
    ----------
    x :                 Volume | dict
                        Volume to export. Can be::
                          - pymaid.Volume
                          - dict: {
                                   'faces': array-like,
                                   'vertices':  array-like
                                   }
    name :              str
                        Name of volume. If ``None`` will use the Volume's
                        ``.name`` property.
    comments :          str, optional
                        Comments to upload.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                         Server response.

    """

    if isinstance(x, core.Volume):
        verts = x.vertices.astype(int).tolist()
        faces = x.faces.astype(int).tolist()
    elif isinstance(x, dict):
        verts = x['vertices'].astype(int).tolist()
        faces = x['faces'].astype(int).tolist()
    else:
        raise TypeError('Expected pymaid.Volume or dictionary, '
                        'got "{}"'.format(type(x)))

    if not isinstance(name, str) and isinstance(x, core.Volume):
        name = getattr(x, 'name', 'not named')

    remote_instance = utils._eval_remote_instance(remote_instance)

    postdata = {'title': name,
                'type': 'trimesh',
                'mesh': json.dumps([verts, faces]),
                'comment': comments if comments else ''
                }

    url = remote_instance._upload_volume_url()

    response = remote_instance.fetch(url, postdata)

    if 'success' in response and response['success'] is True:
        pass
    else:
        logger.error('Error exporting volume {}'.format(name))

    return response


@cache.never_cache
def upload_neuron(x, import_tags=False, import_annotations=False,
                  skeleton_id=None, neuron_id=None, force_id=False,
                  remote_instance=None):
    """ Export neuron(s) to CatmaidInstance.

    Currently only imports treenodes and (optionally) tags and annotations.
    Also note that skeleton and treenode IDs will change (see server response
    for old->new mapping). Neuron to import must not have more than one
    skeleton (i.e. disconnected components = more than one root node).

    Parameters
    ----------
    x :                  CatmaidNeuron/List
                         Neurons to upload.
    import_tags :        bool, optional
                         If True, will export treenode tags from ``x.tags``.
    import_annotations : bool, optional
                         If True will export annotations from ``x.annotations``.
    skeleton_id :        int, optional
                         Use this to set the new skeleton's ID. If not
                         provided will will generate a new ID upon export.
    neuron_id :          int, optional
                         Use this to associate the new skeleton with an
                         existing neuron.
    force_id :           bool, optional
                         If True and neuron/skeleton IDs already exist in
                         project, their instances will be replaced. If False
                         and you pass ``neuron_id`` or ``skeleton_id`` that
                         already exist, an error will be thrown.
    remote_instance :    CatmaidInstance, optional
                         If not passed directly, will try using global.

    Returns
    -------
    dict
                         Server response with new skeleton/treenode IDs::

                            {
                             'neuron_id': new neuron ID,
                             'skeleton_id': new skeleton ID,
                             'node_id_map': {'old_node_id': new_node_id, ...},
                             'annotations': if import_annotations=True,
                             'tags': if tags=True
                            }

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(x, core.CatmaidNeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            # Check if any neurons has multiple skeletons
            many = [n.skeleton_id for n in x if n.n_skeletons > 1]
            if many:
                logger.warning('Neurons with multiple disconnected skeletons'
                               'found: {}'.format(','.join(many)))

                answer = ""
                while answer not in ["y", "n"]:
                    answer = input("Fragments will be joined before import. "
                                   "Continue? [Y/N] ").lower()

                if answer != 'y':
                    logger.warning('Import cancelled.')
                    return

                x = morpho.heal_fragmented_neuron(x, min_size=0, inplace=False)

            resp = {n.skeleton_id: upload_neuron(n,
                                                 neuron_id=neuron_id,
                                                 import_tags=import_tags,
                                                 import_annotations=import_annotations,
                                                 remote_instance=remote_instance)
                    for n in config.tqdm(x,
                                         desc='Import',
                                         disable=config.pbar_hide,
                                         leave=config.pbar_leave)}

            errors = {n: r for n, r in resp.items() if 'error' in r}
            if errors:
                logger.error('{} error(s) during upload. Check neuron(s): '
                             '{}'.format(len(errors), ','.join(errors.keys())))

            return resp

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Expected CatmaidNeuron/List, got "{}"'.format(type(x)))

    if x.n_skeletons > 1:
        logger.warning('Neuron has multiple disconnected skeletons. Will heal'
                       ' fragments before import!')
        x = morpho.heal_fragmented_neuron(x, min_size=0, inplace=False)

    import_url = remote_instance._import_skeleton_url()

    import_post = {'neuron_id': neuron_id,
                   'skeleton_id': skeleton_id,
                   'name': x.neuron_name,
                   'force': force_id,
                   'auto_id': False}

    f = os.path.join(tempfile.gettempdir(), 'temp.swc')

    # Keep SWC node map
    swc_map = utils.to_swc(x, filename=f, export_synapses=False, min_radius=-1)

    with open(f, 'rb') as file:
        # Large files can cause a 504 Gateway timeout. In that case, we want
        # to have a log of it without interrupting potential subsequent uploads.
        try:
            resp = remote_instance.fetch(import_url,
                                         post=import_post,
                                         files={'file': file})
        except requests.exceptions.HTTPError as err:
            if 'gateway time-out' in str(err).lower():
                logger.debug('Gateway time-out while uploading {}. Retrying..'.format(x.neuron_name))
                try:
                    resp = remote_instance.fetch(import_url,
                                                 post=import_post,
                                                 files={'file': file})
                except requests.exceptions.HTTPError as err:
                    logger.error('Timeout uploading neuron "{}"'.format(x.neuron_name))
                    return {'error': err}
                except BaseException:
                    raise
            else:
                # Any other error should just raise
                raise
        except BaseException:
            raise

    # If error is returned
    if 'error' in resp:
        logger.error('Error uploading neuron "{}"'.format(x.neuron_name))
        return resp

    # Exporting to SWC changes the node IDs -> we will revert this in the
    # response of the server
    nmap = {n: resp['node_id_map'].get(str(swc_map[n]), None) for n in swc_map}
    resp['node_id_map'] = nmap

    if import_tags and getattr(x, 'tags', {}):
        # Map old to new nodes
        tags = {t: [nmap[n] for n in v] for t, v in x.tags.items()}
        # Invert tag dictionary: map node ID -> list of tags
        ntags = {}
        for t in tags:
            ntags.update({n: ntags.get(n, []) + [t] for n in tags[t]})

        resp['tags'] = add_tags(list(ntags.keys()),
                                ntags,
                                'TREENODE',
                                remote_instance=remote_instance)

    # Make sure to not access `.annotations` directly to not trigger
    # fetching annotations
    if import_annotations and 'annotations' in x.__dict__:
        an = x.__dict__.get('annotations', [])
        resp['annotations'] = add_annotations(resp['skeleton_id'], an,
                                              remote_instance=remote_instance)

    return resp


@cache.never_cache
def update_skeleton(x, skeleton_id=None, remote_instance=None):
    """ Update skeleton in CatmaidInstance.

    This will override existing skeleton data and map back annotations, tags
    and connectors.

    Parameters
    ----------
    x :                  CatmaidNeuron
                         Neuron to update.
    skeleton_id :        int, optional
                         ID of skeleton to update. If not provided will use
                         `.skeleton_id` of input neuron.
    remote_instance :    CatmaidInstance, optional
                         If not passed directly, will try using global.

    Returns
    -------
    dict
                         Server response with new skeleton/treenode IDs::

                            {
                             'neuron_id': new neuron ID,
                             'skeleton_id': new skeleton ID,
                             'node_id_map': {'old_node_id': new_node_id, ...},
                             'annotations': if import_annotations=True,
                             'tags': if tags=True
                            }
    """
    pass


@cache.never_cache
def join_skeletons(x, winner=None, no_prompt=False, method='LEAFS',
                   remote_instance=None):
    """ Join multiple skeletons based on minimum spanning tree.

    Parameters
    ----------
    x :                 CatmaidNeuronList
                        Skeletons to join.
    winner :            CatmaidNeuron | skeleton ID, optional
                        Winning skeleton that gets to keep its skeleton ID.
                        If not provided, will use the largest fragment.
    method :            'LEAFS' | 'ALL'', optional
                        Set stitching method:
                            (1) 'LEAFS': Only leaf (including root) nodes will
                                be allowed to make new edges.
                            (2) 'ALL': All treenodes are considered.
    no_prompt :         bool, optional
                        If True, will NOT prompt before joining.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Server response
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, core.CatmaidNeuronList):
        raise TypeError('Expected CatmaidNeuronList, got "{}"'.format(type(x)))

    if len(x) < 2:
        raise ValueError('Must provide at least two skeletons')

    if winner and winner not in x:
        raise ValueError('Winner must be in list of skeletons')
    elif not winner:
        winner = sorted([n for n in x], key=lambda x: x.n_nodes, reverse=True)[0]

    ALLOWED_METHODS = ['LEAFS', 'ALL']
    if method.upper() not in ALLOWED_METHODS:
        raise ValueError('Method "{}" not allowed'.format(method))

    # Get edges that need adding
    edges_to_add = morpho.stitch_neurons(x, method=method, suggest_only=True)

    if not no_prompt:
        # Create mock neuron for visualization
        coords = x.nodes.set_index('treenode_id')[['x', 'y', 'z']].to_dict()
        swc = pd.DataFrame([])
        swc['treenode_id'] = np.arange(0, len(edges_to_add) * 2)
        swc['parent_id'] = None  # we need this to prevent conversion to floats
        swc.loc[np.arange(0, len(edges_to_add)), 'parent_id'] = np.arange(len(edges_to_add), len(edges_to_add) * 2)

        swc.loc[np.arange(0, len(edges_to_add)), 'x'] = [coords['x'][e[0]] for e in edges_to_add]
        swc.loc[np.arange(0, len(edges_to_add)), 'y'] = [coords['y'][e[0]] for e in edges_to_add]
        swc.loc[np.arange(0, len(edges_to_add)), 'z'] = [coords['z'][e[0]] for e in edges_to_add]
        swc.loc[np.arange(len(edges_to_add), len(edges_to_add) * 2), 'x'] = [coords['x'][e[1]] for e in edges_to_add]
        swc.loc[np.arange(len(edges_to_add), len(edges_to_add) * 2), 'y'] = [coords['y'][e[1]] for e in edges_to_add]
        swc.loc[np.arange(len(edges_to_add), len(edges_to_add) * 2), 'z'] = [coords['z'][e[1]] for e in edges_to_add]

        swc['radius'] = 200

        mock = core.CatmaidNeuron(1)
        mock.neuron_name = 'Mock'
        mock.nodes = swc
        mock.tags = {}
        mock.connectors = pd.DataFrame([])
        mock._clear_temp_attr()

        colors = sns.color_palette('bright', len(x))

        # Visualise before prompting
        v = scene3d.Viewer()
        v.add(x, color=colors, connectors=False)
        v.add(mock.nodes[['x', 'y', 'z']].values,
              scatter_kws={'size': 10,
                           'color': 'w'})
        v.add(mock, color='w', use_radius=False, connectors=False)

        answer = ""
        print('Please check suggested joins in 3D viewer before proceeding.')
        while answer not in ["y", "n"]:
            answer = input("Proceed? [Y/N] ").lower()
            if answer != 'y':
                return

    responses = []
    for e in config.tqdm(edges_to_add,
                         desc='Joining',
                         disable=config.pbar_hide,
                         leave=config.pbar_leave):
        # Make sure that we keep the winner on top
        if e[1] in winner.nodes.treenode_id.values:
            win, loose = e[1], e[0]
        else:
            win, loose = e[0], e[1]

        responses.append(join_nodes(win, loose,
                                    no_prompt=True,
                                    remote_instance=remote_instance))

        # Stop early if error encountered
        if 'error' in responses[-1]:
            return responses

    return responses


@cache.never_cache
def join_nodes(winner_node, looser_node, no_prompt=False,
               remote_instance=None):
    """ Join two skeletons by nodes.

    All annotations are being kept. Reference to original neuron will be
    added.

    Parameters
    ----------
    winner_node :       int
                        Treenode ID of winning skeleton to merge onto.
                        Skeleton ID of this neuron will persist.
    looser_node :       int
                        Treenode ID of loosing skeleton to merge. Skeleton ID
                        of this neuron will be lost!
    no_prompt :         bool, optional
                        If True, will NOT prompt before joining.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Server response
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    try:
        winner_node = int(winner_node)
        looser_node = int(looser_node)
    except BaseException:
        raise ValueError('winner/looser_node must be numeric IDs')

    # We need to provide a state for each node
    details = fetch.get_node_details([winner_node, looser_node],
                                     convert_ts=False,
                                     remote_instance=remote_instance)
    details.node_id = details.node_id.astype(int)
    edition_times = details.set_index('node_id').edition_time.to_dict()

    if winner_node not in edition_times:
        raise ValueError('winner_node does not exist')
    if looser_node not in edition_times:
        raise ValueError('looser_node does not exist')

    skids = fetch.get_skid_from_treenode([winner_node, looser_node],
                                         remote_instance=remote_instance)
    winner_skid = skids[winner_node]
    looser_skid = skids[looser_node]

    names = fetch.get_names([winner_skid, looser_skid],
                            remote_instance=remote_instance)
    winner_name = names[str(winner_skid)]
    looser_name = names[str(looser_skid)]

    # Get annotations
    annotations = fetch.get_annotation_details([winner_skid, looser_skid],
                                               remote_instance=remote_instance)
    # Turn annotations into dictionary
    annotation_set = annotations.set_index('annotation').user_id.to_dict()
    # Add reference to original neuron
    login = remote_instance.fetch(remote_instance._get_login_info_url())
    annotation_set[looser_name] = login['userid']

    if not no_prompt:
        print('Joining {} ({}) into {} ({})'.format(looser_name,
                                                    looser_skid,
                                                    winner_name,
                                                    winner_skid))
        print('Skeleton ID {} will cease to exist'.format(looser_skid))
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("Proceed? [Y/N] ").lower()
            if answer != 'y':
                return

    join_url = remote_instance._join_skeletons_url()
    join_post = {'from_id': winner_node,
                 'to_id': looser_node,
                 'annotation_set': json.dumps(annotation_set),
                 'edition_times': json.dumps([[n, edition_times[n]] for n in [winner_node, looser_node]]),
                 'sampler_handling': 'domain-end'}

    resp = remote_instance.fetch(join_url, join_post)
    if 'error' in resp:
        logger.error('Error joining nodes: see response for details.')
    return resp


@cache.never_cache
def link_connector(links, remote_instance=None):
    """ Link connectors with treenode.

    Parameters
    ----------
    links :             tuple | list of tuples
                        Tuple (or list thereof) of node IDs describing the
                        link to make::

                            (treenode_id, connector_id, 'presynaptic_to') will make treenode presynaptic to connector
                            (treenode_id, connector_id, 'postsynaptic_to') will make treenode postsynaptic to connector

    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    # Some sanity checks
    if not utils._is_iterable(links):
        raise TypeError('Expected tuple(s) of node, connector IDs.')

    # Turn into list of tuples
    if not utils._is_iterable(links[0]):
        links = [links]

    # Make sure all tuples have correct length
    if any([len(l) != 3 for l in links]):
        raise ValueError('Tuples must have exactly three entries: '
                         '(treenode_id, connector_id, relation)')

    # Make sure all relations are correct
    if any([l[2] not in ['presynaptic_to', 'postsynaptic_to'] for l in links]):
        raise ValueError('Tuple relationships must only be "presynaptic_to" '
                         'or "postsynaptic_to".')

    create_link_url = [remote_instance._create_link_url()] * len(links)

    # We need to provide a state for each node
    all_ids = [n for l in links for n in l[:2]]
    details = fetch.get_node_details(all_ids,
                                     convert_ts=False,
                                     remote_instance=remote_instance)
    edition_times = details.set_index('node_id').edition_time.to_dict()

    # State has to be provided as {'state': [(node_id, edition_time), ..]}
    # We have to explicitly convert the state in a json string because passing
    # it to requests as "post" will fuck this up otherwise
    link_post = [{'from_id': l[0],
                  'to_id': l[1],
                  'state': json.dumps([[n, edition_times[str(n)]] for n in l[:2]]),
                  'link_type': l[2]} for l in links]

    return remote_instance.fetch(create_link_url, link_post)


@cache.never_cache
def update_radii(radii, remote_instance=None):
    """ Change radii [nm] of given treenodes.

    Parameters
    ----------
    radii :             dict, CatmaidNeuron/List
                        Dictionary mapping treenode IDs to new radii or a
                        CatmaidNeuron.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        Server response with::

                            {
                             'success': True/False,
                             'update_nodes': {
                                        node_id: {'old': old_radius,
                                                  'new': new_radius
                                                  'edition_time': new_edition_time,
                                                  'skeleton_id': skeleton_id,
                                                 },
                                             }
                            }


    Examples
    --------
    >>> radii = {41500568: 50, 41500567: 100, 41500564: 200}
    >>> pymaid.update_radii(radii)

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(radii, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        radii = radii.nodes.set_index('treenode_id').radius.to_dict()

    if not isinstance(radii, dict):
        raise TypeError('Expected dictionary, got "{}"'.format(type(radii)))

    if any([not isinstance(v, numbers.Number) for v in radii.keys()]):
        raise ValueError('Expecting only numerical treenode IDs.')

    if any([not isinstance(v, numbers.Number) for v in radii.values()]):
        raise ValueError('New radii must be numerical.')

    update_radii_url = remote_instance._update_treenode_radii()

    update_post = {"treenode_ids[{}]".format(i): k for i, k in enumerate(radii.keys())}
    update_post.update({"treenode_radii[{}]".format(i): k for i, k in enumerate(radii.values())})

    # We need to provide a state for each node
    details = fetch.get_node_details(list(radii.keys()),
                                     convert_ts=False,
                                     remote_instance=remote_instance)
    edition_times = details.set_index('node_id').edition_time.to_dict()

    # State has to be provided as {'state': [(node_id, edition_time), ..]}
    update_post.update({"state": [(str(k), edition_times[str(k)]) for k in radii]})

    # We have to explicitly convert the state in a json string because passing
    # it to requests as "post" will fuck this up otherwise
    update_post['state'] = json.dumps(update_post['state'])

    return remote_instance.fetch(update_radii_url, update_post)


@cache.never_cache
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
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Examples
    --------
    Add your initials to a set of neurons

    >>> # Get their current names
    >>> names = pymaid.get_names('annotation:Philipps neurons')
    >>> # Add initials to names
    >>> new_names = {skid: name + ' PS' for skid, name in names.items()}
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
        temp = {str(n): new_names[n] for n in new_names}
        # Generate a list from the dict
        new_names = [temp[n] for n in x if n in temp]
    elif not isinstance(new_names, (list, np.ndarray)):
        new_names = [new_names]

    if len(x) != len(new_names):
        raise ValueError('Need a name for every single neuron to rename.')

    if not no_prompt:
        old_names = fetch.get_names(x)
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
    neuron_ids = fetch.get_neuron_id(x, remote_instance=remote_instance)
    for skid, name in zip(x, new_names):
        # Renaming works with neuron ID, which we can get via this API endpoint
        nid = neuron_ids[str(skid)]
        url_list.append(remote_instance._rename_neuron_url(nid))
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


@cache.never_cache
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
    remote_instance :   CatmaidInstance, optional
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
        post_data = [{'tags': ','.join(tags[n]),
                      'delete_existing': override_existing}
                     for n in node_list]
    else:
        post_data = [{'tags': ','.join(tags),
                      'delete_existing': override_existing}
                     for n in node_list]

    return remote_instance.fetch(add_tags_urls,
                                 post=post_data,
                                 desc='Modifying tags')


@cache.never_cache
def delete_neuron(x, no_prompt=False, remote_instance=None):
    """ Completely delete neurons.

    .. danger::

        **Use this with EXTREME caution as this is irreversible!**

    Important
    ---------
    Deletes a neuron if (and only if!) two things are the case:

     1. You own all treenodes of the skeleton making up the neuron in question
     2. The neuron is not annotated by other users

    Parameters
    ----------
    x
                        Neurons to check if they exist in Catmaid. Can be
                        either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    no_prompt :         bool, optional
                        By default, you will be prompted to confirm before
                        deleting the neuron(s). Set this to True to skip that
                        step.
    remote_instance :   CatmaidInstance, optional
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


@cache.never_cache
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
    remote_instance :   CatmaidInstance, optional
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

    >>> # Load neuron
    >>> n = pymaid.get_neuron(16)
    >>> # Get non-end nodes
    >>> non_leaf_nodes = n.nodes[n.nodes.type != 'end']
    >>> # Define which tags to remove
    >>> tags_to_remove = ['ends', 'uncertain end', 'uncertain continuation',
    ...                   'TODO']
    >>> # Remove tags
    >>> resp = pymaid.delete_tags(non_leaf_nodes.treenode_id.values,
    ...                           tags_to_remove, 'TREENODE')

    """

    PERM_NODE_TYPES = ['TREENODE', 'CONNECTOR']

    if node_type not in PERM_NODE_TYPES:
        raise ValueError('Unknown node_type "{0}". Please use either: '
                         '{1}'.format(node_type, ','.join(PERM_NODE_TYPES)))

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(node_list, (list, np.ndarray)):
        node_list = [node_list]

    # Make sure node list is strings
    node_list = [str(n) for n in node_list]

    if not isinstance(tags, (list, np.ndarray)):
        tags = [tags]

    if tags != [None]:
        # First, get existing tags for these nodes
        existing_tags = fetch.get_node_tags(node_list, node_type,
                                            remote_instance=remote_instance)

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


@cache.never_cache
def add_meta_annotations(to_annotate, to_add, remote_instance=None):
    """ Add meta-annotation(s) to annotation(s).

    Parameters
    ----------
    to_annotate :       str | list of str
                        Annotation(s) to meta-annotate.
    to_add :            str | list of str
                        Meta-annotation(s) to add to annotations.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Nothing

    See Also
    --------
    :func:`~pymaid.remove_meta_annotations`
                        Delete given annotations from neurons.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    # Get annotation IDs
    to_annotate = utils._make_iterable(to_annotate)
    an = fetch.get_annotation_list(remote_instance=remote_instance)
    an = an[an.annotation.isin(to_annotate)]

    if an.shape[0] != len(to_annotate):
        missing = set(to_annotate).difference(an.annotation.values)
        raise ValueError('Annotation(s) not found: {}'.format(','.join(missing)))

    an_ids = an.annotation_id.values

    to_add = utils._make_iterable(to_add)

    add_annotations_url = remote_instance._get_add_annotations_url()

    add_annotations_postdata = {}

    for i, x in enumerate(an_ids):
        key = 'entity_ids[%i]' % i
        add_annotations_postdata[key] = str(x)

    for i, x in enumerate(to_add):
        key = 'annotations[%i]' % i
        add_annotations_postdata[key] = str(x)

    logger.info(remote_instance.fetch(
        add_annotations_url, add_annotations_postdata))

    return


@cache.never_cache
def remove_meta_annotations(remove_from, to_remove, remote_instance=None):
    """ Remove meta-annotation(s) from annotation(s).

    Parameters
    ----------
    remove_from :       str | list of str
                        Annotation(s) to de-meta-annotate.
    to_remove :         str | list of str
                        Meta-annotation(s) to remove from annotations.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Nothing

    See Also
    --------
    :func:`~pymaid.add_meta_annotations`
                        Delete given annotations from neurons.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    an = fetch.get_annotation_list(remote_instance=remote_instance)

    # Get annotation IDs
    remove_from = utils._make_iterable(remove_from)
    rm = an[an.annotation.isin(remove_from)]
    if rm.shape[0] != len(remove_from):
        missing = set(remove_from).difference(rm.annotation.values)
        raise ValueError('Annotation(s) not found: {}'.format(','.join(missing)))
    an_ids = rm.annotation_id.values

    # Get meta-annotation IDs
    to_remove = utils._make_iterable(to_remove)
    rm = an[an.annotation.isin(to_remove)]
    if rm.shape[0] != len(to_remove):
        missing = set(to_remove).difference(rm.annotation.values)
        raise ValueError('Meta-annotation(s) not found: {}'.format(','.join(missing)))
    rm_ids = rm.annotation_id.values

    add_annotations_url = remote_instance._get_remove_annotations_url()

    remove_annotations_postdata = {}

    for i, x in enumerate(an_ids):
        key = 'entity_ids[%i]' % i
        remove_annotations_postdata[key] = str(x)

    for i, x in enumerate(rm_ids):
        key = 'annotation_ids[%i]' % i
        remove_annotations_postdata[key] = str(x)

    print(remove_annotations_postdata)

    logger.info(remote_instance.fetch(
        add_annotations_url, remove_annotations_postdata))

    return


@cache.never_cache
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
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    None

    See Also
    --------
    :func:`~pymaid.add_annotations`
                        Add given annotations to neuron(s).

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    annotations = utils._make_iterable(annotations)

    # Translate into annotations ID
    an_list = fetch.get_annotation_list().set_index('annotation')

    an_ids = []
    for a in annotations:
        if a not in an_list.index:
            logger.warning(
                'Annotation {0} not found. Skipping.'.format(a))
            continue
        an_ids.append(an_list.loc[a, 'annotation_id'])

    remove_annotations_url = remote_instance._get_remove_annotations_url()

    remove_annotations_postdata = {}

    neuron_ids = fetch.get_neuron_id(x, remote_instance=remote_instance)

    for i, s in enumerate(x):
        # This requires neuron IDs
        key = 'entity_ids[%i]' % i
        remove_annotations_postdata[key] = neuron_ids[str(s)]

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
            logger.info('Removed "{0}" from {1} entities ({2} uses left)'.format(an_list.loc[int(a), 'annotation'],
                                                                                 len(resp['deleted_annotations'][a]['targetIds']),
                                                                                 resp['left_uses'][a]))
    else:
        logger.info('No annotations removed.')

    return


@cache.never_cache
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
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        Server response.

    See Also
    --------
    :func:`~pymaid.remove_annotations`
                        Delete given annotations from neurons.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    annotations = utils._make_iterable(annotations)

    add_annotations_url = remote_instance._get_add_annotations_url()

    add_annotations_postdata = {}

    neuron_ids = fetch.get_neuron_id(x, remote_instance=remote_instance)

    for i, s in enumerate(x):
        key = 'entity_ids[%i]' % i
        add_annotations_postdata[key] = neuron_ids[str(s)]

    for i in range(len(annotations)):
        key = 'annotations[%i]' % i
        add_annotations_postdata[key] = str(annotations[i])

    return remote_instance.fetch(add_annotations_url,
                                 add_annotations_postdata)
