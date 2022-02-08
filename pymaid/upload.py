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

from datetime import datetime as dt
from datetime import timezone
import json
import numbers
import os
import tempfile
import traceback

import numpy as np
import pandas as pd
import navis as ns
import requests
import seaborn as sns

from scipy.spatial.distance import cdist

from . import (core, utils, config, cache, fetch, client)

__all__ = sorted(['add_annotations', 'remove_annotations',
                  'add_tags', 'delete_tags',
                  'delete_neuron',
                  'rename_neurons',
                  'add_meta_annotations', 'remove_meta_annotations',
                  'upload_neuron', 'upload_volume',
                  'update_radii', 'replace_skeleton',
                  'join_skeletons', 'join_nodes',
                  'link_connector', 'delete_nodes',
                  'add_connector', 'transfer_neuron',
                  'differential_upload', 'move_nodes',
                  'push_new_root', 'add_node',
                  'update_node_confidence',
                  'delete_volume', 'set_nodes_reviewed'])

# Set up logging
logger = config.logger


@cache.never_cache
def upload_volume(x, name, comments=None, remote_instance=None):
    """Upload volume/mesh to CatmaidInstance.

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
    if isinstance(x, ns.Volume):
        verts = x.vertices.astype(int).tolist()
        faces = x.faces.astype(int).tolist()
    elif isinstance(x, dict):
        verts = x['vertices'].astype(int).tolist()
        faces = x['faces'].astype(int).tolist()
    else:
        raise TypeError('Expected navis or pymaid Volume or dictionary, '
                        'got "{}"'.format(type(x)))

    if not isinstance(name, str) and isinstance(x, ns.Volume):
        name = getattr(x, 'name', 'not named')

    remote_instance = utils._eval_remote_instance(remote_instance)

    postdata = {'title': name,
                'type': 'trimesh',
                'mesh': json.dumps([verts, faces]),
                'comment': comments if comments else ''
                }

    url = remote_instance._upload_volume_url()

    response = remote_instance.fetch(url, post=postdata)

    if 'success' in response and response['success'] is True:
        pass
    else:
        logger.error('Error exporting volume {}'.format(name))

    return response


@cache.never_cache
def delete_volume(x, no_prompt=False, remote_instance=None):
    """Delete volume from Catmaid Instance.

    Parameters
    ----------
    x :                 int | str
                        Name (str) or ID (int) of volume to delete.
    no_prompt :         bool, optional
                        If True, will skip prompt to confirm deletion.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                         Server response.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, (str, int, np.integer)):
        raise TypeError('Expected volume name (str) or ID (int), '
                        'got "{}"'.format(type(x)))

    # First, get volume IDs
    get_volumes_url = remote_instance._get_volumes()
    resp = remote_instance.fetch(get_volumes_url)

    all_vols = pd.DataFrame(resp['data'], columns=resp['columns'])

    # Get volume name + ID
    if isinstance(x, (int, np.integer)):
        id2name = all_vols.set_index('id').name.to_dict()

        if x not in id2name:
            raise ValueError('Volume "{}" not found'.format(x))
        vol_name = id2name[x]
        vol_id = x
    elif isinstance(x, str):
        name2id = all_vols.set_index('name').id.to_dict()

        if x not in name2id:
            raise ValueError('Volume "{}" not found'.format(x))
        vol_name = x
        vol_id = name2id[x]

    if not no_prompt:
        # Now prompt
        answer = ""
        q = 'Please confirm deletion of Volume "{}" (ID {}) [Y/N] '.format(vol_name, vol_id)
        while answer not in ["y", "n"]:
            answer = input(q).lower()

        if answer != 'y':
            return

    url = remote_instance._get_volume_details(vol_id)

    req = remote_instance._session.delete(url)
    req.raise_for_status()
    resp = req.json()

    if 'error' in resp:
        logger.error('Error deleting volume {}: {}'.format(x, resp))

    return resp


def transfer_neuron(x, source_instance, target_instance, move_tags=False,
                    move_annotations=False, move_connectors=False,
                    force_id=False, no_prompt=False):
    """Copy neuron(s) from one CatmaidInstance to another.

    Note that skeleton, node and connector IDs will change (see server
    response for old->new mapping). Also: node confidences are currently not
    transferred.

    Parameters
    ----------
    x :                  Skeleton ID(s)
                         Neuron(s) to move from ``source_instance`` to
                         ``target_instance``.
    source_instance :    CatmaidInstance
                         Instance that the neuron(s) currently live in.
    target_instance :    CatmaidInstance
                         Instance to copy the neuron(s) to.
    move_tags :          bool, optional
                         If True, will upload node tags from ``x.tags``.
    move_annotations :   bool, optional
                         If True will upload annotations from ``x.annotations``.
    move_connectors :    bool, optional
                         If True will upload connectors from ``x.connectors``.
    force_id :           bool, optional
                         If True and neuron/skeleton IDs already exist in
                         target instance, they will be replaced. **Use this with
                         extrem caution as this will destroy the existing
                         skeleton!**
    no_prompt :          bool, optional
                         If True, will not prompt before transferring neurons!

    Returns
    -------
    dict
                         Server response with new skeleton/node IDs::

                            {
                             'neuron_id': new neuron ID,
                             'skeleton_id': new skeleton ID,
                             'node_id_map': {'old_node_id': new_node_id, ...},
                             'annotations': if import_annotations=True,
                             'tags': if tags=True
                            }

    """
    # TODOs:
    # - move node confidences
    if not isinstance(source_instance, client.CatmaidInstance):
        raise TypeError('"source_instance" must be CatmaidInstance not "{}"'.format(type(source_instance)))

    if not isinstance(target_instance, client.CatmaidInstance):
        raise TypeError('"target_instance" must be CatmaidInstance not "{}"'.format(type(target_instance)))

    if source_instance == target_instance:
        raise ValueError('source_instance must the same as target_instance')

    # We can't use the decorator in this case because the remote instances are
    # not a "remote_instance" keyword argument
    old_caching = source_instance.caching
    source_instance.caching = False
    try:
        skids = utils.eval_skids(x, remote_instance=source_instance)

        neurons = fetch.get_neurons(skids, remote_instance=source_instance)

        if not isinstance(neurons, core.CatmaidNeuronList):
            neurons = core.CatmaidNeuronList(neurons)

        if move_annotations:
            neurons.get_annotations()
    except BaseException:
        raise
    finally:
        source_instance.caching = old_caching

    if not no_prompt:
        summary = neurons.summary()[['name', 'id', 'n_nodes']]

        if move_connectors:
            summary['n_connectors'] = neurons.n_connectors

        if move_tags:
            summary['n_tags'] = [len(n.tags) for n in neurons]

        if move_annotations:
            summary['n_annotations'] = [len(n.annotations) for n in neurons]

        print(summary.to_string())

        q = 'Transferring above neurons from {} (project ID {}) to {} (project ID {}). Proceed? [Y/N] '
        q = q.format(source_instance.server,
                     source_instance.project_id,
                     target_instance.server,
                     target_instance.project_id)

        answer = ""
        while answer not in ["y", "n"]:
            answer = input(q).lower()
            if answer != 'y':
                return

    return upload_neuron(neurons,
                         import_tags=move_tags,
                         import_annotations=move_annotations,
                         import_connectors=move_connectors,
                         skeleton_id=neurons.skeleton_id.astype(int) if force_id else None,
                         force_id=force_id,
                         source_id=neurons.skeleton_id.astype(int),
                         source_project_id=source_instance.project_id,
                         source_url=source_instance.server,
                         remote_instance=target_instance)


@cache.never_cache
def upload_neuron(x, import_tags=False, import_annotations=False,
                  import_connectors=False, reuse_existing_connectors=True,
                  skeleton_id=None, neuron_id=None, force_id=False,
                  source_id=None, source_project_id=None,
                  source_url=None, source_type=None, remote_instance=None):
    """Export (upload) neurons to CatmaidInstance.

    Note that skeleton, node and connector IDs will change (see server
    response for old->new mapping). Neuron to import must not have more than one
    skeleton (i.e. disconnected components = more than one root node).

    Parameters
    ----------
    x :                  CatmaidNeuron/List
                         Neurons to upload.
    import_tags :        bool, optional
                         If True, will upload node tags from ``x.tags``.
    import_annotations : bool, optional
                         If True will upload annotations from ``x.annotations``.
    import_connectors :  bool, optional
                         If True will upload connectors from ``x.connectors``.
    reuse_existing_connectors : bool, optional
                         Only matters when import_connectors is True.
                         If True will look in the remote_instance at the
                         location of each of ``x``'s connectors, and if
                         present, ``x`` will be linked to that existing
                         connector instead of a new (duplicate) connector being
                         created at that location. If False all of
                         ``x.connectors`` are uploaded as new.
    skeleton_id :        int, optional
                         Use this to set the Id of the new skeleton(s). If not
                         provided will will generate a new ID upon export.
    neuron_id :          int, optional
                         Use this to associate the new skeleton(s) with an
                         existing neuron.
    force_id :           bool, optional
                         If True and neuron/skeleton IDs already exist in
                         project, their instances will be replaced. If False
                         and you pass ``neuron_id`` or ``skeleton_id`` that
                         already exist, an error will be thrown.
    source_id :          int, optional
    source_project_id :  int, optional
    source_url :         str, optional
    source_type :        "skeleton" | "segmentation", optional
                         ``source_{}`` are optional fields that will be
                         associated with the newly uploaded neuron to help keep
                         track of neurons' origins. You can use
                         :func:`~pymaid.get_origin`, :func:`~pymaid.get_skids_by_origin`
                         and :func:`~pymaid.find_neurons` to look up neurons
                         by their origin.
    remote_instance :    CatmaidInstance, optional
                         CatmaidInstance to upload to. If not passed directly,
                         will try using global.

    Returns
    -------
    dict
                         Server response with new skeleton/node IDs::

                            {
                             'neuron_id': new neuron ID,
                             'skeleton_id': new skeleton ID,
                             'node_id_map': {'old_node_id': new_node_id, ...},
                             'annotations': if import_annotations=True,
                             'tags': if tags=True
                            }

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(x, ns.NeuronList):
        # Check if any neurons are only single nodes
        if any(x.n_nodes <= 1) and (not isinstance(skeleton_id, type(None))
                                    or not isinstance(neuron_id, type(None))):
            raise ValueError('Single-node neurons can currently not be uploaded '
                             'with a given skeleton or neuron ID.')

        vars = dict(neuron_id=neuron_id,
                    skeleton_id=skeleton_id,
                    source_id=source_id,
                    source_project_id=source_project_id,
                    source_url=source_url,
                    source_type=source_type)

        # Parse variables that require unique values for each neuron
        for k in ['neuron_id', 'skeleton_id', 'source_id']:
            if not isinstance(vars[k], (type(None), bool)):
                # Make sure it is iterable
                vars[k] = list(utils._make_iterable(vars[k]))

                if len(vars[k]) != len(x):
                    raise ValueError('Must provide "{}" for each uploaded neuron.'.format(k))
            else:
                vars[k] = [vars[k]] * len(x)

        # Parse variables that can (but don't have to) be the same for all neurons
        for k in ['source_project_id', 'source_url', 'source_type']:
            if not utils._is_iterable(vars[k]):
                vars[k] = [vars[k]] * len(x)
            elif len(vars[k]) != len(x):
                raise ValueError('Must provide "{}" for each uploaded neuron.'.format(k))

        # Check if any neurons has multiple skeletons
        many = [n.id for n in x if n.n_skeletons > 1]
        if many:
            logger.warning('Neurons with multiple disconnected skeletons'
                           'found: {}'.format(', '.join(many)))

            answer = ""
            while answer not in ["y", "n"]:
                answer = input("Fragments will be joined before import. "
                               "Continue? [Y/N] ").lower()

            if answer != 'y':
                logger.warning('Import cancelled.')
                return

            x = ns.heal_skeleton(x, min_size=0, inplace=False)

        resp = {n.id: upload_neuron(n,
                                    neuron_id=vars['neuron_id'][i],
                                    skeleton_id=vars['skeleton_id'][i],
                                    import_tags=import_tags,
                                    import_annotations=import_annotations,
                                    import_connectors=import_connectors,
                                    force_id=force_id,
                                    source_id=vars['source_id'][i],
                                    source_project_id=vars['source_project_id'][i],
                                    source_url=vars['source_url'][i],
                                    source_type=vars['source_type'][i],
                                    remote_instance=remote_instance)
                for i, n in config.tqdm(enumerate(x),
                                        desc='Uploading',
                                        total=len(x),
                                        disable=config.pbar_hide,
                                        leave=config.pbar_leave)}

        errors = {n: r for n, r in resp.items() if 'error' in r}
        if errors:
            logger.error('{} error(s) during upload. Check neuron(s): '
                         '{}'.format(len(errors), ','.join(errors.keys())))

        return resp

    if not isinstance(x, ns.TreeNeuron):
        raise TypeError('Expected CatmaidNeuron/List, got "{}"'.format(type(x)))

    if x.nodes.empty:
        raise ValueError('{} #{}: Unable to upload neuron without nodes'.format(x.name, x.id))

    if x.n_skeletons > 1:
        logger.warning('Neuron has multiple disconnected skeletons. Will heal'
                       ' fragments before import!')
        x = ns.heal_skeleton(x, min_size=0, inplace=False)

    if source_type and source_type not in ['skeleton', 'segmentation']:
        raise ValueError('Expected source_type to be "skeleton" or '
                         '"segmentation", got "{}"'.format(source_type))

    for v, n, t in zip([source_id, source_url, source_project_id],
                       ['source_id', 'source_url', 'source_project_id'],
                       [(int, np.integer), str, (int, np.integer)]):
        if not isinstance(v, (type(None), t)):
            raise TypeError('{} must be None or {}, got {}'.format(n, t, type(v)))

    # Check if any neurons are only single nodes
    # -> these need to be uploaded differently
    if x.n_nodes <= 1:
        if not isinstance(skeleton_id, type(None)) or not isinstance(neuron_id, type(None)):
            raise ValueError('Single-node neurons can currently not be uploaded '
                             'with a given skeleton or neuron ID.')

        node = x.nodes.iloc[0]

        vars = dict(coords=node[['x', 'y', 'z']].values,
                    parent_id=None,
                    remote_instance=remote_instance)

        if hasattr(node, 'confidence'):
            vars['confidence'] = node.confidence if node.confidence else None

        if hasattr(node, 'radius'):
            vars['radius'] = node.radius

        resp = add_node(**vars)

        # If error is returned
        if 'error' in resp:
            logger.error('Error uploading neuron "{}"'.format(x.name))
            return resp

        # Add node ID map to match with normal upload
        resp['node_id_map'] = {node.node_id: resp['treenode_id']}

    else:
        import_url = remote_instance._import_skeleton_url()

        import_post = {'neuron_id': neuron_id,
                       'skeleton_id': skeleton_id,
                       'name': x.name,
                       'force': force_id,
                       'auto_id': False}

        # Add ID fields
        for k, v in zip(['source_id', 'source_url', 'source_project_id', 'source_type'],
                        [source_id, source_url, source_project_id, source_type]):
            if v:
                import_post[k] = v

        f = os.path.join(tempfile.gettempdir(), 'temp.swc')

        # Keep SWC node map
        swc_map = ns.write_swc(x,
                               filepath=f,
                               export_connectors=False,
                               labels=False,
                               return_node_map=True)

        with open(f, 'rb') as file:
            # Large files can cause a 504 Gateway timeout. In that case, we want
            # to have a log of it without interrupting potential subsequent uploads.
            try:
                resp = remote_instance.fetch(import_url,
                                             post=import_post,
                                             files={'file': file})
            except requests.exceptions.HTTPError as err:
                if 'gateway time-out' in str(err).lower():
                    logger.debug('Gateway time-out while uploading {}. Retrying..'.format(x.name))
                    try:
                        resp = remote_instance.fetch(import_url,
                                                     post=import_post,
                                                     files={'file': file})
                    except requests.exceptions.HTTPError as err:
                        logger.error('Timeout uploading neuron "{}"'.format(x.name))
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
            logger.error('Error uploading neuron "{}"'.format(x.name))
            return resp

        # Exporting to SWC changes the node IDs -> we will revert this in the
        # response of the server
        n_map = {n: resp['node_id_map'].get(str(swc_map[n]), None) for n in swc_map}
        resp['node_id_map'] = n_map

    if import_tags and getattr(x, 'tags', {}):
        # Map old to new nodes
        tags = {t: [resp['node_id_map'][n] for n in v] for t, v in x.tags.items()}
        # Invert tag dictionary: map node ID -> list of tags
        ntags = {}
        for t in tags:
            ntags.update({n: ntags.get(n, []) + [t] for n in tags[t]})

        resp['tags'] = add_tags(list(ntags.keys()),
                                ntags,
                                'NODE',
                                remote_instance=remote_instance)

    # Make sure to not access `.annotations` directly to not trigger
    # fetching annotations
    if import_annotations and '_annotations' in x.__dict__:
        an = x.__dict__.get('_annotations', [])
        resp['annotations'] = add_annotations(resp['skeleton_id'], an,
                                              remote_instance=remote_instance)

    if import_connectors and x.has_connectors and not x.connectors.empty:
        # Connectors that make multiple links onto the neuron will be listed
        # more than once but only want to upload them once
        connectors_no_duplicates = x.connectors.drop_duplicates(subset=['connector_id'])

        # First create new connectors
        cn_resp = add_connector(connectors_no_duplicates[['x', 'y', 'z']].values,
                                check_existing=reuse_existing_connectors,
                                remote_instance=remote_instance)

        resp['connector_response'] = cn_resp

        # Create old to new IDs map
        cn_map = {old: new['connector_id'] for old, new in zip(connectors_no_duplicates.connector_id.values,
                                                               cn_resp)}

        # Add map to server response
        resp['connector_id_map'] = cn_map

        # Hard-coded relation map
        rl_map = config.compact_skeleton_relations

        # Link connectors
        links = [[resp['node_id_map'][n.node_id],
                  cn_map[n.connector_id],
                  rl_map[n.type]] for n in x.connectors.itertuples()]

        ln_resp = link_connector(links, remote_instance=remote_instance)

        resp['link_response'] = ln_resp

        if import_tags and getattr(x, 'connector_tags', {}):
            # Map old to new connectors
            cn_tags = {t: [cn_map[n] for n in v] for t, v in x.connector_tags.items()}
            # Invert connector tag dictionary: map connctor ID -> list of tags
            ctags = {}
            for t in cn_tags:
                ctags.update({n: ctags.get(n, []) + [t] for n in cn_tags[t]})

            resp['connector_tags'] = add_tags(list(ctags.keys()),
                                              ctags,
                                              'CONNECTOR',
                                              override_existing=True,
                                              remote_instance=remote_instance)

    return resp


@cache.never_cache
def differential_upload(x, skeleton_id=None, no_prompt=False, remote_instance=None):
    """Upload only changes made to a neuron.

    In brief, this function takes the input neuron ``x``, compares with its
    live version on the server and makes incremental changes:
        1. Remove nodes not present in ``x`` from live neuron
        2. Add nodes present in ``x`` but not in live neuron
        3. Move nodes present in ``x`` and live neuron that have changed positions

    .. danger::

        **Use this with EXTREME caution as this is irreversible!**

    Parameters
    ----------
    x :                 CatmaidNeuron/List
                        Neurons to upload.
    skeleton_id :       int, optional
                        Use this to set the target live neuron. If not
                        provided will will use ``x.skeleton_id``.
    no_prompt :         bool, optional
                        If True, will not prompt before uploading changes!
    remote_instance :   CatmaidInstance, optional
                        CatmaidInstance to upload to. If not passed directly,
                        will try using global.

    Returns
    -------
    None
                        If everything went well.

    dict
                        On error, returns dict with server response.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(x, core.CatmaidNeuronList):
        if len(x) > 1:
            raise ValueError('Expected a single CatmaidNeuron, got {}'.format(len(x)))
        x = x[0]

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Expected CatmaidNeuron, got "{}"'.format(x))

    skeleton_id = x.skeleton_id if not skeleton_id else skeleton_id

    # Check if neuron actually exist
    if not fetch.neuron_exists(skeleton_id, remote_instance=remote_instance):
        raise ValueError('No neuron with skeleton ID {} found on {} (PID )'.format(skeleton_id,
                                                                                   remote_instance.server,
                                                                                   remote_instance.project_id))

    # Get live neuron
    live = fetch.get_neuron(skeleton_id, remote_instance=remote_instance)

    # Generate report on differences
    report = _diff_report(a=x, b=live)

    if not report['nodes_mutual']:
        raise ValueError('Input and live neuron have no nodes in common!')

    if not no_prompt:
        q = 'Neuron "{}" (#{}) on {} (PID {}) will have:\n{} nodes deleted\n' \
            '{} nodes moved\n{} nodes added\nPlease confirm [Y/N] '
        q = q.format(live.name,
                     live.id,
                     remote_instance.server,
                     remote_instance.project_id,
                     len(report['nodes_b_only']),
                     len(report['nodes_moved']),
                     len(report['nodes_a_only']))
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(q).lower()
            if answer != 'y':
                return

    # We need to reroot our input neuron to one of the mutual nodes so that
    # the fragments we're attaching later have their roots at a node adjacent
    # to the live neuron.
    if not set(x.root) & set(report['nodes_mutual']):
        x.reroot(report['nodes_mutual'][0], inplace=True)

    # First off: delete extra nodes
    # If any of this is a sequence of connected nodes, we have to delete
    # them sequentially anyway - so we'll just go through the pain in any
    # event
    if report['nodes_b_only']:
        for n in config.tqdm(report['nodes_b_only'],
                             desc='Removing nodes',
                             leave=config.pbar_leave,
                             disable=config.pbar_hide):
            resp = delete_nodes(n, 'NODE',
                                no_prompt=True,
                                remote_instance=remote_instance)
            if 'error' in resp:
                # Error is already logged by delete_nodes
                return resp

    # Next add additional nodes
    if report['nodes_a_only']:
        # Generate a neuron consisting only of nodes to be added
        x_ss = ns.subset_neuron(x, report['nodes_a_only'], inplace=False)
        # Turn disconnected trees into separate neurons
        frags = ns.break_fragments(x_ss)

        # Upload each fragment and connect to live neuron
        for f in config.tqdm(frags,
                             desc='Uploading & Joining',
                             leave=config.pbar_leave,
                             disable=config.pbar_hide):
            # Single nodes can't be uploaded as SWC neurons
            if f.nodes.shape[0] == 1:
                parent_id = x.nodes.set_index('node_id').loc[f.root[0],
                                                             'parent_id']
                coords = f.nodes.iloc[0][['x', 'y', 'z']].values
                radius = f.nodes.iloc[0].radius
                resp = add_node(coords,
                                parent_id=parent_id,
                                radius=radius,
                                remote_instance=remote_instance)
            else:
                # Keep track of new skeleton and node IDs
                nmap = upload_neuron(f, remote_instance=remote_instance)

                if 'error' in nmap:
                    # Error is already logged by upload_neuron
                    return nmap

                # Now connect this fragment's root with it's former parent in
                # the input neuron (which is a mutual node)
                looser_node = nmap['node_id_map'][f.root[0]]
                winner_node = x.nodes.set_index('node_id').loc[f.root[0],
                                                               'parent_id']

                resp = join_nodes(winner_node, looser_node, no_prompt=True,
                                  remote_instance=remote_instance)

            if 'error' in resp:
                # Error is already logged by join_nodes
                return resp

    # Last but not least: move nodes
    if report['nodes_moved']:
        # Generate new positions
        new_locs = x.nodes.loc[x.nodes.node_id.isin(report['nodes_moved']),
                               ['node_id', 'x', 'y', 'z']].values

        resp = move_nodes(new_locs,
                          node_type='NODE',
                          no_prompt=True,
                          remote_instance=remote_instance)

        if 'error' in resp:
            # Error is already logged by move_nodes
            return resp

    return


@cache.never_cache
def replace_skeleton(x, skeleton_id=None, force_mapping=False,
                     cold_run=False, remote_instance=None):
    """Replace skeleton in CatmaidInstance.

    This will override existing skeleton data and tries to map back tags
    and connectors. Requires user to have import and API token write access
    privileges!

    Connectors are re-connected by: For each connector,

      1. Get node this connector is connected to in current skeleton.
      2. Get distance to the nodes up- and downstream of it as proxy for
         sampling resolution.
      3. Find the closest node in new skeleton.
      4. Connect automatically if closest node within sampling resolution.
         Else flag and return connector as "requires manual review".

    Node tags are mapped back by: For each tagged node:

      1. Get distance to the nodes up- and downstream of it as proxy for
         sampling resolution.
      2. Find the closest node in new skeleton.
      3. Map tag automatically if closest node is within the sampling
         resolution. Else flag and return connector as "requires manual review".

    Note that this does not respect types of nodes. E.g. an "ends" tag could end
    up on a non-leaf node.

    Any connectors/tags that have not been automatically fixed will be returned
    as DataFrame for manual review. See examples.

    .. danger::

        **Use this with EXTREME caution as this is irreversible!**

    Parameters
    ----------
    x :                 CatmaidNeuron
                        Neuron to update.
    skeleton_id :       int, optional
                        ID of skeleton to update. If not provided will use
                        ``.skeleton_id`` property of input neuron.
    force_mapping :     bool, optional
                        If True, will always re-connect connectors and map tags
                        onto the closest node in new skeleton regardless of
                        distance.
    cold_run :          bool, optional
                        If True, will only calculate and return table of nodes
                        to fix without actually uploading anything.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
                        DataFrame listing connectors and node tags that were
                        either fixed automatically or need manual review.

                            auto_fix    type        connector_id    old_node_id  ...
                         0  False       'connector' 123456          11111
                         1  True        'tag'       None            22222

                            sugg_node_id    relation    tags        x   y   z
                         0  33333           0           None        ...
                         1  44444           None        ['ends']    ...

                        In this example node `11111` in the old skeleton was
                        connected presynaptically (``relation=0``) to a
                        connector. The closest node in the new skeleton is
                        `33333` but it's too far away to be automatically
                        reconnected.

                        Node `22222` had an `ends` tag in the old skeleton and
                        the closest node in the new skeleton is `44444`. Because
                        this node was close enough, it was automatically fixed.

                        ``x/y/z`` coordinates always refer to the position of
                        the old node!

    Examples
    --------
    Pull a neuron from CATMAID, smooth it and upload it again:

    >>> n = pymaid.get_neuron(16)
    >>> n_smoothed = pymaid.smooth_neuron(n, inplace=False)
    >>> to_fix = pymaid.replace_skeleton(n_smoothed)
    Updating skeleton 16
    # of nodes:	12853 -> 12853 (+0)
    Cable length:	2904.2 -> 3027.3 (+123.1)
    1983 of 2116 connectors will be automatically re-connected
    654 of 654 tagged nodes will be automatically mapped back
    Remaining connectors/tagged nodes will be returned as DataFramefor manual review
    Proceed? [Y/N] Y
    >>> # There are 133 items (connectors and tags) to check manually:
    >>> to_fix[~to_fix.auto_fix].shape[0]
    133
    >>> to_fix[~to_fix.auto_fix].head()
        connector_id  old_node_id  relation  sugg_node_id tags       type       x       y       z
    20        304403       125722       NaN        125717  NaN  connector  452034  139101  204160
    47        553830       123430       NaN          2698  NaN  connector  437855  165228  216280
    77        653778       123637       NaN        123636  NaN  connector  450508  134607  188720
    86        666783       127623       NaN        127620  NaN  connector  438700  147328  219840

    To facilitate fixing , we can add urls to the positions and then copy the
    DataFrame to e.g. a spreadsheet:

    >>> fix_manual = to_fix[~to_fix.auto_fix]
    >>> fix_manual['url'] = pymaid.url_to_coordinates(coords=fix_manual,
    ...                                               stack_id=5,  # change this according to your projects
    ...                                               active_skeleton_id=n.skeleton_id,
    ...                                               active_node_id=fix_manual.sugg_node_id.values)
    >>> # Copy to clipboard
    >>> fix_manual.to_clipboard()

    """
    # TODO:
    # - use tangent vector to map connectors/tags back?
    # - constrain certain tags (e.g. "ends" only on leafs)

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Expected CatmaidNeuron, got "{}"'.format(type(x)))

    if isinstance(skeleton_id, type(None)):
        skeleton_id = x.skeleton_id

    if not fetch.neuron_exists(skeleton_id, remote_instance=remote_instance):
        raise ValueError('Neuron with skeleton ID "{}" does not exist'.format(skeleton_id))

    # Get current skeleton that is should be replaced
    y = fetch.get_neuron(skeleton_id, remote_instance=remote_instance)

    # Because compact-skeleton does not return all types of connectors, we have
    # to get them via a separate endpoint
    lk = fetch.get_connector_links(skeleton_id, remote_instance=remote_instance)

    # Find out which connectors we can automatically reconnect:
    # First get distance between each connector node and its neighbours
    cn_nodes = lk.node_id.values
    g = y.graph.to_undirected()
    cn_nodes_dist = []
    for n in cn_nodes:
        cn_nodes_dist.append(np.mean([g.edges[(n, n2)]['weight'] for n2 in g.neighbors(n)]))
    cn_nodes_dist = np.array(cn_nodes_dist)

    # Now find closest node in the new neuron
    cn_dist_new = cdist(y.nodes.set_index('node_id').loc[cn_nodes,
                                                         ['x', 'y', 'z']],
                        x.nodes[['x', 'y', 'z']].values)
    cn_closest_ix = np.argmin(cn_dist_new, axis=1)
    cn_closest_id = x.nodes.iloc[cn_closest_ix]['node_id'].values
    cn_closest_dist = np.amin(cn_dist_new, axis=1)

    if not force_mapping:
        cn_is_close = cn_closest_dist <= cn_nodes_dist
    else:
        cn_is_close = cn_closest_dist <= float('inf')

    # Create dictionary mapping old to new connector node ID
    cn_to_tn = {int(c): int(t) for c, t in zip(cn_nodes, cn_closest_id)}

    # Find out which tags we can automatically map back:
    # First get distance between each tagged node and its connected nodes
    tg_nodes = np.array(list(set([n for t in y.tags for n in y.tags[t]])))
    tg_nodes_dist = []
    for n in tg_nodes:
        tg_nodes_dist.append(np.mean([g.edges[(n, n2)]['weight'] for n2 in g.neighbors(n)]))
    tg_nodes_dist = np.array(tg_nodes_dist)

    # Find closest node in the new neuron
    tg_dist_new = cdist(y.nodes.set_index('node_id').loc[tg_nodes,
                                                         ['x', 'y', 'z']].values,
                        x.nodes[['x', 'y', 'z']].values)
    tg_closest_ix = np.argmin(tg_dist_new, axis=1)
    tg_closest_id = x.nodes.iloc[tg_closest_ix]['node_id'].values
    tg_closest_dist = np.amin(tg_dist_new, axis=1)

    if not force_mapping:
        tg_is_close = tg_closest_dist <= tg_nodes_dist
    else:
        tg_is_close = tg_closest_dist <= float('inf')

    # Create dictionary mapping old to new node ID
    tn_to_tn = {int(c): int(t) for c, t in zip(tg_nodes, tg_closest_id)}

    # Compile list of items to fix after replacing skeleton in case we
    # encounter an error and need to dump this
    cn_to_fix = y.nodes.set_index('node_id').loc[cn_nodes, ['x', 'y', 'z']]
    cn_to_fix = cn_to_fix.copy().reset_index(drop=True)
    cn_to_fix['type'] = 'connector'
    # Do not remove .astype(object) as this prevents conversion to float later
    cn_to_fix['connector_id'] = lk.connector_id.values.astype(object)
    cn_to_fix['old_node_id'] = lk.node_id.values
    cn_to_fix['sugg_node_id'] = cn_to_fix.old_node_id.astype(int).map(cn_to_tn)
    cn_to_fix['relation'] = lk.relation.values
    cn_to_fix['auto_fix'] = cn_is_close

    tg_to_fix = y.nodes.set_index('node_id').loc[tg_nodes, ['x', 'y', 'z']]
    tg_to_fix = tg_to_fix.copy().reset_index(drop=True)
    tg_to_fix['type'] = 'tags'
    tg_to_fix['old_node_id'] = tg_nodes
    tg_to_fix['sugg_node_id'] = tg_to_fix.old_node_id.astype(int).map(tn_to_tn)
    tg_to_fix['tags'] = [[t for t in y.tags if n in y.tags[t]] for n in tg_nodes]
    tg_to_fix['auto_fix'] = tg_is_close

    # Concatenate both dataframes
    to_fix = pd.concat([cn_to_fix, tg_to_fix], axis=0, sort=True).reset_index(drop=True)

    if cold_run:
        return to_fix

    # Prepare some summary to be signed off by user
    print('Updating skeleton {}: {}'.format(y.id, y.name))
    print('# of nodes:\t{} -> {} ({:+})'.format(y.n_nodes,
                                                x.n_nodes,
                                                x.n_nodes - y.n_nodes))
    print('Cable length:\t{:.1f} -> {:.1f} ({:+.1f})'.format(y.cable_length,
                                                             x.cable_length,
                                                             x.cable_length - y.cable_length))
    print('{} of {} connectors will be automatically re-connected'.format(sum(cn_is_close), len(cn_is_close)))
    print('{} of {} tagged nodes will be automatically mapped back'.format(sum(tg_is_close), len(tg_is_close)))
    print('Remaining connectors/tagged nodes will be returned as DataFrame'
          'for manual review')

    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Proceed? [Y/N] ").lower()
        if answer != 'y':
            return to_fix

    # Now things are getting serious!

    # Get the neuron ID
    neuron_id = fetch.get_neuron_id(y,
                                    remote_instance=remote_instance)[y.skeleton_id]

    # Update the neuron + skeleton
    resp = upload_neuron(x,
                         neuron_id=neuron_id,
                         skeleton_id=skeleton_id,
                         force_id=True,
                         import_annotations=False,
                         import_tags=False,
                         remote_instance=remote_instance)

    if 'error' in resp:
        return resp

    # From now on, if anything goes wrong we will return the entirety of
    # connectors / tags for manual review

    try:
        # First, map new node IDs to old node IDs
        to_fix['sugg_node_id'] = to_fix['sugg_node_id'].astype(int).map(resp['node_id_map'])

        # Now re-connect stuff:
        auto_fix = to_fix[to_fix['auto_fix']]

        # Create list of links to make
        new_links = auto_fix[auto_fix['type'] == 'connector'].copy()
        # Create tuples of links
        link_tuples = new_links[['sugg_node_id', 'connector_id', 'relation']]
        # Make sure IDs check out
        link_tuples = link_tuples.apply(tuple, axis=1).tolist()
        # Make links
        resp = link_connector(link_tuples, remote_instance=remote_instance)

        # Generate dictionary of node -> tags
        new_tags = auto_fix[auto_fix['type'] == 'tags'].copy()
        tags_dict = new_tags.set_index('sugg_node_id').tags.to_dict()
        # Add tags
        resp = add_tags(node_list=new_tags.sugg_node_id.values,
                        tags=tags_dict,
                        node_type='NODE',
                        remote_instance=remote_instance)

    except BaseException:
        traceback.print_exc()
        logger.error('Something went wrong! Returning full list of stuff to '
                     'manually review!')
        return to_fix

    return to_fix


@cache.never_cache
def join_skeletons(x, winner=None, no_prompt=False, method='LEAFS',
                   remote_instance=None):
    """Join multiple skeletons by minimizing the length of the newly added
    edges (minimum spanning tree).

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
                            (2) 'ALL': All nodes are considered.
    no_prompt :         bool, optional
                        If True, will NOT prompt before joining.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Server response

    See Also
    --------
    :func:`~pymaid.join_nodes`
                        If you know exactly which nodes to join.

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
    edges_to_add = ns.stitch_neurons(x, method=method, suggest_only=True)

    if not no_prompt:
        # Create mock neuron for visualization
        coords = x.nodes.set_index('node_id')[['x', 'y', 'z']].to_dict()
        swc = pd.DataFrame([])
        swc['node_id'] = np.arange(0, len(edges_to_add) * 2)
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
        mock.name = 'Mock'
        mock.nodes = swc
        mock.tags = {}
        mock.connectors = pd.DataFrame([])
        mock._clear_temp_attr()

        colors = sns.color_palette('bright', len(x))

        # Visualise before prompting
        v = ns.Viewer()
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
        if e[1] in winner.nodes.node_id.values:
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
def join_nodes(winner_node, looser_node, no_prompt=False, tag_nodes=True,
               remote_instance=None):
    """Join two skeletons by nodes.

    All annotations are being kept. Reference to original neuron will be
    added.

    Parameters
    ----------
    winner_node :       int
                        Node ID of winning skeleton to merge onto.
                        Skeleton ID of this neuron will persist.
    looser_node :       int
                        Node ID of loosing skeleton to merge. Skeleton ID
                        of this neuron will be lost!
    no_prompt :         bool, optional
                        If True, will NOT prompt before joining.
    tag_nodes :         bool | str | tuple of str, optional
                        If not False, will add tags to nodes::

                            If True add "joined into {WINNING SKELETON ID}"
                            to loosing and "joined from {LOOSING SKELETON ID}"
                            to winning node.

                            If str, will add string as tag to both nodes.

                            If tuple of two strings (str1, str2), will add
                            str1 to winning and str2 to loosing node.

    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    Server response

    See Also
    --------
    :func:`~pymaid.join_skeletons`
                        If you don't know how at what nodes to join skeletons.

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
        raise ValueError('winner_node "{}" does not exist'.format(winner_node))
    if looser_node not in edition_times:
        raise ValueError('looser_node "{}" does not exist'.format(looser_node))

    skids = fetch.get_skid_from_node([winner_node, looser_node],
                                     remote_instance=remote_instance)
    winner_skid = skids[winner_node]
    looser_skid = skids[looser_node]

    names = fetch.get_names([winner_skid, looser_skid],
                            remote_instance=remote_instance)
    winner_name = names[str(winner_skid)]
    looser_name = names[str(looser_skid)]

    n_samplers = fetch.get_sampler_counts([winner_skid, looser_skid],
                                          remote_instance=remote_instance)

    # Get annotations
    annotations = fetch.get_annotation_details([winner_skid, looser_skid],
                                               remote_instance=remote_instance)
    # Turn annotations into dictionary
    annotation_set = annotations.set_index('annotation').user_id.to_dict()
    # Add reference to original neuron
    login = remote_instance.fetch(remote_instance._get_login_info_url())
    annotation_set[looser_name] = login['userid']

    if not no_prompt:
        print('Joining "{}" #{} into "{}"" #{}'.format(looser_name,
                                                       looser_skid,
                                                       winner_name,
                                                       winner_skid))
        print('Skeleton ID {} will cease to exist'.format(looser_skid))

        n_loosing_samplers = n_samplers[str(looser_skid)]
        if n_loosing_samplers:
            print('Loosing skeleton has {}'.format(n_loosing_samplers),
                  'sampler(s) that will be lost on merge!')

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

    resp = remote_instance.fetch(join_url, post=join_post)
    if 'error' in resp:
        logger.error('Error joining nodes {} and {}. See response for details.'.format(winner_node, looser_node))
        return resp

    if tag_nodes:
        if isinstance(tag_nodes, bool):
            tags = {winner_node: 'joined from {}'.format(looser_skid),
                    looser_node: 'joined into {}'.format(winner_skid)}
        elif isinstance(tag_nodes, str):
            tags = tag_nodes
        elif isinstance(tag_nodes, (tuple, list, np.ndarray)):
            if any([not isinstance(s, str) for s in tag_nodes]):
                raise TypeError('Tags must be strings')
            tags = {winner_node: tag_nodes[0],
                    looser_node: tag_nodes[1]}
        else:
            raise TypeError('Unable to parse node tags of type "{}"'.format(type(tag_nodes)))

        tag_resp = add_tags([winner_node, looser_node],
                            tags=tags,
                            node_type='NODE',
                            remote_instance=remote_instance,
                            override_existing=False)

        if 'error' in tag_resp:
            return tag_resp

    return resp


@cache.never_cache
def link_connector(links, remote_instance=None):
    """Link connectors with nodes.

    Parameters
    ----------
    links :             tuple | list of tuples
                        Tuple (or list thereof) of node IDs describing the
                        link to make::

                            (node_id, connector_id, 'presynaptic_to') will make node presynaptic to connector
                            (node_id, connector_id, 'postsynaptic_to') will make node postsynaptic to connector

                        See ``pymaid.config.link_types`` for allowed `relation`.

    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    list
                        List of dictionary with server reponses.

    See Also
    --------
    :func:`~pymaid.add_connector`
                        To create new connectors.

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
                         '(node_id, connector_id, relation)')

    # Make sure all relations are correct
    ALLOWED_RELATIONS = [l['relation'] for l in config.link_types]
    if any([l[2] not in ALLOWED_RELATIONS for l in links]):
        raise ValueError('Tuple relationships must be either: {}'.format(', '.join(ALLOWED_RELATIONS)))

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

    resp = remote_instance.fetch(create_link_url, post=link_post,
                                 desc='Linking connectors')

    if any(['error' in r for r in resp]):
        logger.error('Error creating link(s)! Check server response')

    return resp


@cache.never_cache
def update_node_confidence(confidences, to_connector=False, remote_instance=None):
    """Change node confidences.

    Parameters
    ----------
    confidences :       dict
                        Dictionary mapping node ID -> confidences::

                            {node_id[int]: new_confidence[int]}

    to_connector:       bool, optional
                        If True, will change confidence between this node and
                        linked connector.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    list
                        List of dictionary with server reponses.

    See Also
    --------
    :func:`~pymaid.add_node`
                        To create new nodes.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    # Some sanity checks
    if not isinstance(confidences, dict):
        raise TypeError('Expected dict, got "{}"'.format(type(confidences)))

    if any([5 <= c <= 1 for c in confidences.values()]):
        raise ValueError('New confidences must be 1-5')

    all_ids = list(confidences.keys())

    update_conf_url = [remote_instance._update_node_confidence_url(n) for n in all_ids]

    details = fetch.get_node_details(all_ids,
                                     convert_ts=False,
                                     remote_instance=remote_instance)
    edition_times = details.set_index('node_id').edition_time.to_dict()

    missing = [str(n) for n in all_ids if str(n) not in edition_times]
    if missing:
        raise ValueError('Node(s) do not exist: {}'.format(', '.join(missing)))

    # We have to explicitly convert the state in a json string because passing
    # it to requests as "post" will fuck this up otherwise
    conf_post = [{'new_confidence': confidences[n],
                  'state': json.dumps({'edition_time': edition_times[str(n)]}),
                  'to_connector': to_connector} for n in all_ids]

    resp = remote_instance.fetch(update_conf_url, post=conf_post,
                                 desc='Updating confidences')

    if any(['error' in r for r in resp]):
        logger.error('Error changing confidences! Check server response')

    return resp


@cache.never_cache
def update_radii(radii, chunk_size=1000, remote_instance=None):
    """Change radii [nm] of given nodes.

    Parameters
    ----------
    radii :             dict, CatmaidNeuron/List
                        Dictionary mapping node IDs to new radii or a
                        CatmaidNeuron.
    chunk_size :        int, optional
                        Update radii in chunks of this size. The maximal
                        chunk size depends on your server's configuration.
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
        radii = radii.nodes.set_index('node_id').radius.to_dict()

    if not isinstance(radii, dict):
        raise TypeError('Expected dictionary, got "{}"'.format(type(radii)))

    if any([not isinstance(v, numbers.Number) for v in radii.keys()]):
        raise ValueError('Expecting only numerical node IDs.')

    if any([not isinstance(v, numbers.Number) for v in radii.values()]):
        raise ValueError('New radii must be numerical.')

    # We have to force node IDs to integer to avoid np.int32 and np.in64 as
    # these upset json.dumps() later on
    radii = {int(n): r for n, r in radii.items()}

    update_radii_url = remote_instance._update_node_radii()

    # We need to provide a state for each node
    details = fetch.get_node_details(list(radii.keys()),
                                     convert_ts=False,
                                     remote_instance=remote_instance)
    edition_times = details.set_index('node_id').edition_time.to_dict()

    tn_ids = list(radii.keys())
    resp = {}
    for i in config.trange(0, len(radii), int(chunk_size),
                           desc='Updating radii',
                           disable=config.pbar_hide,
                           leave=config.pbar_leave):
        this_chunk = {n: radii[n] for n in tn_ids[i: i + chunk_size]}

        update_post = {"treenode_ids[{}]".format(i): k for i, k in enumerate(this_chunk.keys())}
        update_post.update({"treenode_radii[{}]".format(i): k for i, k in enumerate(this_chunk.values())})

        # State has to be provided as {'state': [(node_id, edition_time), ..]}
        update_post.update({"state": [(int(k), edition_times[str(k)]) for k in this_chunk]})

        # We have to explicitly convert the state in a json string because passing
        # it to requests as "post" will f*** this up otherwise
        update_post['state'] = json.dumps(update_post['state'])

        this_resp = remote_instance.fetch(update_radii_url, post=update_post,
                                          on_error='pass')

        # Merge responses
        for r, v in this_resp.items():
            if isinstance(v, dict):
                d = resp.get(r, {})
                d.update(v)
                resp[r] = d
            elif isinstance(v, list):
                resp[r] = resp.get(r, []) + v
            else:
                resp[r] = resp.get(r, []) + [v]

    if 'errors' in resp and len(resp['errors']):
        errors = resp['errors']
        logger.error('{} errors when updating radii. See server response for details'.format(len(errors)))

    return resp


@cache.never_cache
def rename_neurons(x, new_names, remote_instance=None, no_prompt=False):
    """Rename neuron(s).

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
    Server response

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
        old_names = fetch.get_names(x, remote_instance=remote_instance)
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
        failed = [n for i, n in enumerate(x) if responses[i]['success'] is False]
        logger.error('Error renaming neuron(s): {}'.format(','.join(failed)))

    return responses


@cache.never_cache
def add_node(coords, parent_id=None, radius=-1, confidence=5, remote_instance=None):
    """Create single (!) node at given location.

    Parameters
    ----------
    coords :            tuple
                        Tuple containing x/y/z coordinates.
    parent_id :         int | None, optional
                        If not None, will connect new node to this parent.
    radius :            int, optional
                        Radius of new node.
    confidence :        int, optional
                        Edge confidence to parent (if applicable).
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        Response from Catmaid server.

    See Also
    --------
    :func:`~pymaid.add_connector`
                        Use this to add connectors.
    :func:`~pymaid.delete_nodes`
                        Use this to delete nodes or connectors.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    coords = np.array(coords) if not isinstance(coords, np.ndarray) else coords

    if coords.ndim != 1 or coords.shape[0] != 3:
        raise ValueError('Must provide a SINGLE x/y/z coordinate')

    url = remote_instance._create_node_url()

    if confidence is np.nan:
        confidence = None
    if radius is np.nan:
        radius = None

    post = {'confidence': confidence,
            'radius': radius,
            'useneuron': -1,
            'neuron_name': None,
            'x': coords[0],
            'y': coords[1],
            'z': coords[2]}

    # If parent is provided, we have to get the link
    if parent_id:
        nd = fetch.get_node_details(parent_id,
                                    convert_ts=False,
                                    remote_instance=remote_instance)
        state = {"parent": [int(parent_id), nd.iloc[0]['edition_time']]}
        post['parent_id'] = int(parent_id)
    else:
        # If no parent we just pass an empy state
        state = {"parent": [-1, ""]}

    post['state'] = json.dumps(state)

    resp = remote_instance.fetch(url, post=post, on_error='pass')

    if 'error' in resp:
        logger.error('Error adding node. See server response for details.')

    return resp


@cache.never_cache
def add_connector(coords, check_existing=True, remote_instance=None):
    """Create connector(s) at given location.

    Parameters
    ----------
    coords :            list-like
                        Either single or list of [x, y, z] coordinates at which
                        to create new connectors.
    check_existing :    bool, optional
                        If True will search the remote_instance at the _exact_
                        location of each connector to be uploaded, and if a
                        connector is already present, a new connector will not
                        be created, instead the existing connector's ID will be
                        returned.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        Response from Catmaid server containing new connector ID.

    See Also
    --------
    :func:`~pymaid.link_connector`
                        To link your newly created connectors to nodes.
    :func:`~pymaid.add_node`
                        Use this to add nodes.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)
    resp = []
    if not utils._is_iterable(coords[0]):
        coords = [coords]

    coords = np.array(coords)

    if coords.shape[1] != 3:
        raise ValueError('Expected x/y/z coordinates, got {}'.format(coords.shape[1]))

    if check_existing:
        already_existing = []  # list of bool, order corresponds to 'coord' argument
        existing_resp = []  # building return values for existing connectors

        # Preparing for upload of new connectors
        create_connector_url = remote_instance._create_connector_url()
        create_urls = []
        create_post = []

        for coord in coords:
            existing_connector = fetch.get_connectors_in_bbox(
                [[coord[0]-1, coord[0]+1],
                 [coord[1]-1, coord[1]+1],
                 [coord[2]-1, coord[2]+1]],
                ret='IDS',
                remote_instance=remote_instance
            )

            if len(existing_connector) == 0:
                already_existing.append(False)
                create_urls.append(create_connector_url)
                create_post.append({'pid': remote_instance.project_id,
                                    'confidence': 5,
                                    'x': coord[0],
                                    'y': coord[1],
                                    'z': coord[2]})
            else:
                already_existing.append(True)
                existing_resp.append({'connector_id': existing_connector[0][0]})

        # Create only the connectors that didn't already exist. Save server response
        created_resp = remote_instance.fetch(create_urls, post=create_post,
                                             desc='Creating connectors')

        # Interleave existing_resp and created_resp to make a complete server resp
        # with an order that matches the input 'coord' argument
        resp = [existing_resp.pop(0) if e else created_resp.pop(0)
                for e in already_existing]
    else:
        url = [remote_instance._create_connector_url()] * coords.shape[0]

        post = [{'pid': remote_instance.project_id,
                 'confidence': 5,
                 'x': c[0],
                 'y': c[1],
                 'z': c[2]} for c in coords]

        resp = remote_instance.fetch(url, post=post, desc='Creating connectors')

    if 'error' in resp:
        logger.error('Error adding connector(s). See server response for details.')

    return resp


@cache.never_cache
def add_tags(node_list, tags, node_type, remote_instance=None,
             override_existing=False):
    """Add or edit tag(s) for a list of node(s) or connector(s).

    Parameters
    ----------
    node_list :         list
                        Node or connector IDs to edit.
    tags :              str | list | dict
                        Tags(s) to add to provided node/connector ids. If
                        a dictionary is provided `{node_id: [tag1, tag2], ...}`
                        each node gets individual tags. If string or list
                        are provided, all nodes will get the same tags.
    node_type :         'NODE' | 'CONNECTOR'
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

    if node_type in ['TREENODE', 'TREENODES', 'NODE', 'NODES']:
        add_tags_urls = [
            remote_instance._node_add_tag_url(n) for n in node_list]
    elif node_type in ['CONNECTOR', 'CONNECTORS']:
        add_tags_urls = [
            remote_instance._connector_add_tag_url(n) for n in node_list]
    else:
        raise TypeError('Unknown node_type parameter: %s' % str(node_type))

    if isinstance(tags, dict):
        # Make sure that it's a dict of {node: [tag1, tag2]}
        tags = {n: t if utils._is_iterable(t) else [t] for n, t in tags.items()}

        post_data = [{'tags': ','.join(tags[n]),
                      'delete_existing': override_existing}
                     for n in node_list]
    else:
        post_data = [{'tags': ','.join(tags),
                      'delete_existing': override_existing}
                     for n in node_list]

    resp = remote_instance.fetch(add_tags_urls,
                                 post=post_data,
                                 desc='Modifying tags')

    errors = [r for r in resp if 'error' in r]
    if errors:
        logger.error('{} error(s) tagging nodes. See response for details.'.format(len(errors)))

    return resp


@cache.never_cache
def delete_neuron(x, no_prompt=False, remote_instance=None):
    """Completely delete neurons.

    .. danger::

        **Use this with EXTREME caution as this is irreversible!** Your only
        chance to recover accidentally deleted neurons is asking a server
        admin for help.

    Important
    ---------
    Deletes a neuron if (and only if!) two conditions are met:

     1. You own all nodes of the skeleton making up the neuron in question.
     2. The neuron is not annotated by other users.

    Parameters
    ----------
    x
                        Neurons to delete. Can be
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
        return {n: delete_neuron(n, remote_instance=remote_instance) for n in x}
    else:
        x = x[0]

    # Need to get the neuron ID
    remote_get_neuron_name = remote_instance._get_single_neuronname_url(x)
    neuronid = remote_instance.fetch(remote_get_neuron_name).get('neuronid')

    if not neuronid:
        raise ValueError("Can't find neuron with skeleton ID {}".format(x))

    if not no_prompt:
        # Get name
        name = fetch.get_names(x, remote_instance=remote_instance).get(str(x))

        # Get annotations
        an = fetch.get_annotations(x, remote_instance=remote_instance).get(str(x), [])

        # Get review status and # nodes
        review = fetch.get_review(x, remote_instance=remote_instance).set_index('skeleton_id')

        # Prompt
        answer = ""
        s = 'Neuron "{}" (#{}) has {} nodes ({} reviewed) and {} annotation(s)'
        s = s.format(name, x,
                     review.loc[str(x), 'total_node_count'],
                     review.loc[str(x), 'percent_reviewed'],
                     len(an))
        print(s)
        q = 'Please confirm deletion [Y/N] '
        while answer not in ["y", "n"]:
            answer = input(q).lower()

        if answer != 'y':
            return

    url = remote_instance._delete_neuron_url(neuronid)

    resp = remote_instance.fetch(url)

    if 'error' in resp:
        logger.error('Error deleting neuron {}. See server response for details'.format(x.skeleton_id))

    return resp


@cache.never_cache
def push_new_root(new_root, no_prompt=False, remote_instance=None):
    """Reroot neuron on server.

    Parameters
    ----------
    new_root :          int
                        ID of new root.
    no_prompt :         bool, optional
                        By default, you will be prompted to confirm before
                        reroot. Set this to True to skip that step.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    server response

    See Also
    --------
    :func:`~navis.reroot_skeleton`
                        Use to reroot a local CatmaidNeuron.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if utils._is_iterable(new_root):
        return {r: push_new_root(r,
                                 no_prompt=no_prompt,
                                 remote_instance=remote_instance)
                for r in config.tqdm(new_root,
                                     desc='Rerooting',
                                     disable=config.pbar_hide,
                                     leave=config.pbar_leave)}

    try:
        new_root = int(new_root)
    except BaseException:
        raise TypeError('New root must be numeric node ID')

    # Get the associated skeleton and neuron name
    skid = fetch.get_skid_from_node(new_root,
                                    remote_instance=remote_instance)[new_root]
    name = fetch.get_names(skid, remote_instance=remote_instance)[str(skid)]

    # We need to provide new_root's state, the states of its
    # children and any connector links - the best way to get this is
    # using the list query. Matter of fact: there is currently no other
    # query that gives us link IDs AFAIK and we need this as state.

    # First get new_root's locations
    loc = fetch.get_node_location(new_root,
                                  remote_instance=remote_instance)
    n = loc.iloc[0]
    params = {'left':   n.x - 2500,
              'right':  n.x + 2500,
              'top':    n.y - 2500,
              'bottom': n.y + 2500,
              'z1':     n.z - 40,
              'z2':     n.z + 40,
              'treenode_ids[0]': int(n.node_id)}
    url = remote_instance._get_node_list_url(**params)

    # Format of the response is
    # [[nodes], [connectors], {labels}, node_limit_reached, {relation_map}, {extraNodes}]
    # Format of [nodes] is
    # [id, parent_id, location_x, location_y, location_z, confidence, radius, skeleton_id, edition_time, user_id]
    # Format of [connectors] is
    # [id, location_x, location_y, location_z, confidence, edition_time, user_id, [partners]]
    resp = remote_instance.fetch(url)

    # Turn state into dict
    # Get edition times and parents for each node
    states = {n[0]: {'edition_time': dt.fromtimestamp(n[-2],
                                                      tz=timezone.utc).isoformat(),
                     'parent_id': n[1]} for n in resp[0]}
    # Get parent edition time, children and links
    for n in states:
        states[n]['parent'] = [states[n]['parent_id'],
                               states.get(states[n]['parent_id'], {}).get('edition_time')] if states[n]['parent_id'] else None
        states[n]['children'] = [[c,
                                  states[c]['edition_time']] for c in [c for c in states if states[c]['parent_id'] == n]]
        states[n]['links'] = [[l[-1],
                               dt.fromtimestamp(l[-2],
                                                tz=timezone.utc).isoformat()
                               ] for c in resp[1] for l in c[-1] if l[0] == n]

    # Generate postdata for each node
    post = {'treenode_id': new_root,
            'state': json.dumps({'edition_time': states[new_root]['edition_time'],
                                 'parent':       states[new_root]['parent'],
                                 'children':     states[new_root]['children'],
                                 'links':        states[new_root]['links']
                                })
            }

    if not no_prompt:
        q = 'Please confirm rerooting neuron "{}" #{} to node {} [Y/N] '
        q = q.format(name, skid, new_root)
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(q).lower()

        if answer != 'y':
            return

    url = remote_instance._reroot_skeleton_url()
    resp = remote_instance.fetch(url, post=post)

    if 'error' in resp:
        logger.error('Error rerooting neuron! See server response for details.')

    return resp


@cache.never_cache
def delete_nodes(node_ids, node_type, no_prompt=False, remote_instance=None):
    """Delete given nodes or connectors.

    Due to the way CATMAID's node deletion API works, this function is not
    suited for deleting directly linked nodes (e.g. A->B->C). For this to work,
    you will have to call this function in a for loop!

    .. danger::

        **Use this with EXTREME caution as this is irreversible!**

    Parameters
    ----------
    node_ids
                        Single or list of node or connector IDs. Must not
                        be a mix of connectors and nodes.
    node_type :         'NODE' | 'CONNECTOR'
                        Set which type of node you want to delete as they use
                        different API endpoints!
    no_prompt :         bool, optional
                        By default, you will be prompted to confirm before
                        deleting the node(s). Set this to True to skip that
                        step.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    server response

    See Also
    --------
    :func:`~pymaid.delete_neuron`
                        Use to delete entire neurons.
    :func:`~pymaid.update_nodes`
                        Use to move neurons

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(node_ids, (list, np.ndarray)):
        node_ids = [node_ids]

    # Make sure node_ids are integers
    node_ids = [int(n) for n in node_ids]

    # For each node, we need to provide its state, the states of its
    # children and any connector links - the best way to get this is
    # using the list query. Matter of fact: there is currently no other
    # query that gives us link IDs AFAIK and we need this as state.
    # First get node locations
    node_locs = fetch.get_node_location(node_ids,
                                        remote_instance=remote_instance)

    missing = set(node_ids) - set(node_locs.node_id.values.astype(int))
    if missing:
        missing = np.array(list(missing)).astype(str)
        raise ValueError('Node(s) {} not found.'.format(', '.join(missing)))

    # For each node query a window to get its state
    urls = []
    for n in node_locs.itertuples():
        params = {'left':   n.x - 2500,
                  'right':  n.x + 2500,
                  'top':    n.y - 2500,
                  'bottom': n.y + 2500,
                  'z1':     n.z - 40,
                  'z2':     n.z + 40}
        if 'node' in node_type.lower():
            params['treenode_ids[0]'] = int(n.node_id)
        elif 'connector' in node_type.lower():
            params['connector_ids[0]'] = int(n.node_id)

        u = remote_instance._get_node_list_url(**params)
        urls.append(u)

    # Format of the response is
    # [[nodes], [connectors], {labels}, node_limit_reached, {relation_map}, {extraNodes}]
    # Format of [nodes] is
    # [id, parent_id, location_x, location_y, location_z, confidence, radius, skeleton_id, edition_time, user_id]
    # Format of [connectors] is
    # [id, location_x, location_y, location_z, confidence, edition_time, user_id, [partners]]
    resp = remote_instance.fetch(urls, desc='Fetching node states')

    if node_type.lower() in ['treenode', 'treenodes', 'node', 'nodes']:
        # Turn state into dict
        # Get edition times and parents for each node
        states = {n[0]: {'edition_time': dt.fromtimestamp(n[-2],
                                                          tz=timezone.utc).isoformat(),
                         'parent_id': n[1]} for r in resp for n in r[0]}
        # Get parent edition time, children and links
        for n in states:
            states[n]['parent'] = [states[n]['parent_id'],
                                   states.get(states[n]['parent_id'], {}).get('edition_time')] if states[n]['parent_id'] else None
            states[n]['children'] = [[c,
                                      states[c]['edition_time']] for c in [c for c in states if states[c]['parent_id'] == n]]
            states[n]['links'] = [[l[-1],
                                   dt.fromtimestamp(l[-2],
                                                    tz=timezone.utc).isoformat()
                                   ] for r in resp for c in r[1] for l in c[-1] if l[0] == n]

        # Generate postdata for each node
        post = [{'treenode_id': n,
                 'state': json.dumps({'edition_time': states[n]['edition_time'],
                                      'parent':       states[n]['parent'],
                                      'children':     states[n]['children'],
                                      'links':        states[n]['links']
                                       })
                } for n in node_ids]

        # Sanity check:
        # If any of the nodes shows up as another nodes parent, we won't be able
        # to delete as states will have changed between deletes.
        parents = [states[n]['parent'][0] for n in node_ids if states[n]['parent']]
        if set(node_ids) & set(parents):
            raise ValueError('Unable to delete linked nodes in a single go. '
                             'Please use for-loops for this.')

        urls = [remote_instance._delete_node_url()] * len(post)
    elif node_type.lower() in ['connector', 'connectors']:
        # Filter each response to just the connector we need
        states = [[c for c in r[1] if c[0] == n][0] for n, r in zip(node_ids, resp)]

        post = [{'connector_id': n,
                 'state': json.dumps({'edition_time': dt.fromtimestamp(st[-3],
                                                                       tz=timezone.utc).isoformat(),
                                      'c_links': [[l[-1],
                                                   dt.fromtimestamp(l[-2],
                                                                    tz=timezone.utc).isoformat()
                                                  ]
                                                  for l in st[-1]]})}
                 for n, st in zip(node_ids, states)]

        urls = [remote_instance._delete_connector_url()] * len(post)
    else:
        raise TypeError('Unknown node_type parameter "{}"'.format(node_type))

    if not no_prompt:
        answer = ""
        while answer not in ["y", "n"]:
            q = "Please confirm deletion of {} nodes [Y/N] ".format(len(node_ids))
            answer = input(q).lower()

        if answer != 'y':
            return

    resp = remote_instance.fetch(urls, post=post, desc='Deleting nodes')
    errors = [str(n) for i, n in enumerate(node_ids) if 'error' in resp[i]]

    if errors:
        logger.error('Error deleting node(s) {}. See server responses.'.format(', '.join(errors)))

    return resp


@cache.never_cache
def move_nodes(new_locs, node_type, no_prompt=False, remote_instance=None):
    """Update location of given nodes or connectors.

    .. danger::

        **Use this with EXTREME caution as this is irreversible!**

    Parameters
    ----------
    new_locs            dict | list
                        Either dictionary or list mapping node IDs to new
                        x/y/z locations::

                          {node_id: [x, y , z], ...}
                          [[node_id, x, y, z], ... ]

                        Must not be a mix of connectors and nodes!
    node_type :         'NODE' | 'CONNECTOR'
                        Set which type of nodes you want to move as they
                        need to be parsed differently!
    no_prompt :         bool, optional
                        By default, you will be prompted to confirm before
                        moving the node(s). Set this to True to skip that
                        step.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    server response

    See Also
    --------
    :func:`~pymaid.delete_nodes`
                        Use to delete nodes.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    # Parse node ID
    if isinstance(new_locs, dict):
        new_locs = [[k] + v for k, v in new_locs.items()]
    elif not isinstance(new_locs, (list, np.ndarray)):
        raise TypeError('`new_locs` must be dict or list-like, got "{}"'.format(type(new_locs)))

    # Make sure node_ids and locations are ints/float
    new_locs = [[int(n[0])] + [float(l) for l in n[1:]] for n in new_locs]
    node_ids = [n[0] for n in new_locs]

    # For each node, we need to provide its state
    states = fetch.get_node_details(node_ids,
                                    convert_ts=False,
                                    remote_instance=remote_instance)

    missing = set(node_ids) - set(states.node_id.values.astype(int))
    if missing:
        missing = np.array(list(missing)).astype(str)
        raise ValueError('Node(s) {} not found.'.format(', '.join(missing)))

    # Make sure node_id is integer
    states['node_id'] = states.node_id.astype(int)

    # Turn state into list in order of nodes -> we need the double bracket!
    states = states.loc[states.node_id.isin(node_ids), ['node_id', 'edition_time']].values.tolist()

    if node_type.lower() in ['treenode', 'treenodes', 'node', 'nodes']:
        prefix = 't'
    elif node_type.lower() in ['connector', 'connectors']:
        prefix = 'c'
    else:
        raise TypeError('Unknown node_type parameter "{}"'.format(node_type))

    # Generate postdata for each node: for node must be:
    # {t[0][0]: node_id, t[0][1]: x,  t[0][2]: y, t[0][3]: z}
    post = {'{}[{}][{}]'.format(prefix, i, k): v
            for i, n in enumerate(new_locs) for k, v in enumerate(n)}
    post['state'] = json.dumps(states)

    if not no_prompt:
        answer = ""
        while answer not in ["y", "n"]:
            q = "Please confirm moving of {} nodes [Y/N] ".format(len(node_ids))
            answer = input(q).lower()

        if answer != 'y':
            return

    url = remote_instance._update_node_url()
    resp = remote_instance.fetch(url, post=post)

    if 'error' in resp:
        logger.error('Error moving nodes. See server response.')

    return resp


@cache.never_cache
def delete_tags(node_list, tags, node_type, remote_instance=None):
    """Remove tag(s) for a list of node(s) or connector(s).

    Works by getting existing tags, removing given tag(s) and then using
    pymaid.add_tags() to push updated tags back to CATMAID.

    .. danger::

        **Use this with EXTREME caution as this is irreversible!**

    Parameters
    ----------
    node_list :         list
                        Node or connector IDs to delete tags from.
    tags :              list
                        Tags(s) to delete from provided nodes/connectors.
                        Use ``tags=None`` and to remove all tags from a set of
                        nodes.
    node_type :         'NODE' | 'CONNECTOR'
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
    Remove end-related tags from non-end nodes

    >>> # Load neuron
    >>> n = pymaid.get_neuron(16)
    >>> # Get non-end nodes
    >>> non_leaf_nodes = n.nodes[n.nodes.type != 'end']
    >>> # Define which tags to remove
    >>> tags_to_remove = ['ends', 'uncertain end', 'uncertain continuation',
    ...                   'TODO']
    >>> # Remove tags
    >>> resp = pymaid.delete_tags(non_leaf_nodes.node_id.values,
    ...                           tags_to_remove, 'NODE')

    """
    PERM_NODE_TYPES = ['NODE', 'TREENODE', 'CONNECTOR']

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

        # Check if our nodes actually exist
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
    """Add meta-annotation(s) to annotation(s).

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
    dict
                        Server response.

    See Also
    --------
    :func:`~pymaid.remove_meta_annotations`
                        Delete given annotations from neurons.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    # Get annotation IDs
    to_annotate = utils._make_iterable(to_annotate)
    an = fetch.get_annotation_list(remote_instance=remote_instance)
    an = an[an.name.isin(to_annotate)]

    missing = set(to_annotate).difference(an.name.values)
    if missing:
        raise ValueError('Annotation(s) not found: {}'.format(','.join(missing)))

    an_ids = an.id.values

    to_add = utils._make_iterable(to_add)

    add_annotations_url = remote_instance._get_add_annotations_url()

    add_annotations_postdata = {}

    for i, x in enumerate(an_ids):
        key = 'entity_ids[%i]' % i
        add_annotations_postdata[key] = str(x)

    for i, x in enumerate(to_add):
        key = 'annotations[%i]' % i
        add_annotations_postdata[key] = str(x)

    resp = remote_instance.fetch(add_annotations_url,
                                 post=add_annotations_postdata)

    if 'error' in resp:
        logger.error('Error adding annotation. See server response for details.')

    return resp


@cache.never_cache
def remove_meta_annotations(remove_from, to_remove, remote_instance=None):
    """Remove meta-annotation(s) from annotation(s).

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
    dict
                        Server response.

    See Also
    --------
    :func:`~pymaid.add_meta_annotations`
                        Delete given annotations from neurons.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    an = fetch.get_annotation_list(remote_instance=remote_instance)

    # Get annotation IDs
    remove_from = utils._make_iterable(remove_from)
    rm = an[an.name.isin(remove_from)]
    if rm.shape[0] != len(remove_from):
        missing = set(remove_from).difference(rm.name.values)
        raise ValueError('Annotation(s) not found: {}'.format(','.join(missing)))
    an_ids = rm.id.values

    # Get meta-annotation IDs
    to_remove = utils._make_iterable(to_remove)
    rm = an[an.name.isin(to_remove)]
    if rm.shape[0] != len(to_remove):
        missing = set(to_remove).difference(rm.name.values)
        raise ValueError('Meta-annotation(s) not found: {}'.format(','.join(missing)))
    rm_ids = rm.id.values

    add_annotations_url = remote_instance._get_remove_annotations_url()

    remove_annotations_postdata = {}

    for i, x in enumerate(an_ids):
        key = 'entity_ids[%i]' % i
        remove_annotations_postdata[key] = str(x)

    for i, x in enumerate(rm_ids):
        key = 'annotation_ids[%i]' % i
        remove_annotations_postdata[key] = str(x)

    resp = remote_instance.fetch(add_annotations_url,
                                 post=remove_annotations_postdata)

    if 'error' in resp:
        logger.error('Error adding annotation. See server response for details.')

    return resp


@cache.never_cache
def remove_annotations(x, annotations, remote_instance=None):
    """Remove annotation(s) from a list of neuron(s).

    Parameters
    ----------
    x
                        Neurons to remove given annotation(s) from. Can be
                        either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
                        5. ``None`` to remove annotation(s) from all neurons.
    annotations :       list
                        Annotation(s) to remove from neurons.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        Server response.

    See Also
    --------
    :func:`~pymaid.add_annotations`
                        Add given annotations to neuron(s).

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    annotations = utils._make_iterable(annotations)

    # Translate into annotations ID
    an_list = fetch.get_annotation_list(remote_instance=remote_instance).set_index('name')

    an_ids = []
    for a in annotations:
        if a not in an_list.index:
            logger.warning('Annotation {} not found. Skipping.'.format(a))
            continue
        an_ids.append(an_list.loc[a, 'id'])

    remove_annotations_url = remote_instance._get_remove_annotations_url()

    remove_annotations_postdata = {}

    if not isinstance(x, type(None)):
        neuron_ids = fetch.get_neuron_id(x, remote_instance=remote_instance)
        # Turn from skid -> ID into list of IDs
        neuron_ids = list(neuron_ids.values())
    else:
        an = fetch.get_annotated(annotations, remote_instance=remote_instance)
        neuron_ids = an.loc[an.type == 'neuron', 'id'].values

        # Now prompt
        answer = ""
        q = 'Please confirm removal of {} annotation(s) from {} neuron(s) ' \
            '[Y/N] '
        q = q.format(len(annotations), len(neuron_ids))
        while answer not in ["y", "n"]:
            answer = input(q).lower()

        if answer != 'y':
            return

    for i, s in enumerate(neuron_ids):
        # This requires neuron IDs
        key = 'entity_ids[%i]' % i
        remove_annotations_postdata[key] = s

    for i in range(len(an_ids)):
        key = 'annotation_ids[%i]' % i
        remove_annotations_postdata[key] = str(an_ids[i])

    if not an_ids:
        logger.info('No annotations removed.')
        return

    resp = remote_instance.fetch(remove_annotations_url,
                                 post=remove_annotations_postdata)

    an_list = an_list.reset_index().set_index('id')

    if 'error' in resp:
        logger.error('Error deleting annotation(s). See server response for details.')
    elif len(resp['deleted_annotations']) == 0:
        logger.info('No annotations removed.')
    else:
        for a in resp['deleted_annotations']:
            logger.info('Removed "{}" from {} entities ({} uses left)'.format(an_list.loc[int(a), 'name'],
                                                                              len(resp['deleted_annotations'][a]['targetIds']),
                                                                              resp['left_uses'][a]))
    return resp


@cache.never_cache
def add_annotations(x, annotations, remote_instance=None):
    """Add annotation(s) to a list of neuron(s).

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

    resp = remote_instance.fetch(add_annotations_url,
                                 post=add_annotations_postdata)

    if 'error' in resp:
        logger.error("Error adding annotations. See server response for details.")

    return resp


@cache.never_cache
def set_nodes_reviewed(x, remote_instance=None):
    """Set a list of nodes as reviewed.

    Parameters
    ----------
    x :                 int | list-like of int | CatmaidNeuron/List
                        Node(s) to set to reviewed.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        Server response.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    node_ids = utils.eval_node_ids(x, connectors=False, nodes=True)

    urls = [remote_instance._set_nodes_reviewed_url(n) for n in node_ids]

    resp = remote_instance.fetch(urls)

    if 'error' in resp:
        logger.error("Error setting nodes reviewed. See server response for details.")

    return resp


def _diff_report(a, b):
    """Compare neurons ``a`` and ``b`` and report on differences.

    This function compares node IDs!

    Parameters
    ----------
    a,b :       CatmaidNeurons
                Neurons to compare.

    Returns
    -------
    dict
                Report with differences::

                    {'nodes_mutual':    [nodeA, nodeB, ...],
                     'nodes_a_only':    [nodeC, ...],
                     'nodes_by_only':   [nodeD, ...],
                     'nodes_moved':     [nodeB, ...]}

    """
    if not isinstance(a, core.CatmaidNeuron):
        raise TypeError('Expected CatmaidNeuron, got "{}"'.format(type(a)))

    if not isinstance(b, core.CatmaidNeuron):
        raise TypeError('Expected CatmaidNeuron, got "{}"'.format(type(b)))

    nA = set(a.nodes.treenode_id.values)
    nB = set(b.nodes.treenode_id.values)

    report = {}

    report['nodes_mutual'] = list(nA & nB)
    report['nodes_a_only'] = list(nA - nB)
    report['nodes_b_only'] = list(nB - nA)

    posA = a.nodes.set_index('treenode_id').loc[report['nodes_mutual'],
                                                ['x', 'y', 'z']].values
    posB = b.nodes.set_index('treenode_id').loc[report['nodes_mutual'],
                                                ['x', 'y', 'z']].values
    diff = posA - posB
    moved = np.any(diff > 0, axis=1)

    report['nodes_moved'] = list(np.array(report['nodes_mutual'])[moved])

    return report
