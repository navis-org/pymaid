""" Test suite for pymaid

This test suite requires a bunch of variables that need to either defined
as environment variables or in a config_test.py file

    #Catmaid server url
    server_url = ''

    #Http user
    http_user = ''

    #Http password
    http_pw = ''

    #Auth token
    token = ''

    # Test skeleton IDs
    test_skids = []

    # Test annotations
    test_annotations = []

    # Test CATMAID volume
    test_volume = ''

If you use environmental variables, give lists as comma-separated string.


Examples
--------

From terminal:
>>> python test_pymaid.py

From shell:
>>> import unittest
>>> import test_pymaid
>>> suite = unittest.TestLoader().loadTestsFromModule(test_pymaid)
>>> unittest.TextTestRunner().run(suite)

"""

import unittest
import os

import pymaid
import pandas as pd
import networkx as nx

import importlib

# Silence module loggers
pymaid.set_loggers('ERROR')

# Deactivate progress bars
pymaid.set_pbars(hide=True)

# Try getting credentials from environment variable
required_variables = ['server_url', 'http_user', 'http_pw', 'token',
                      'test_skids', 'test_annotations', 'test_volume']

if False not in [v in os.environ for v in required_variables]:
    class conf:
        pass
    config_test = conf()

    for v in ['server_url', 'http_user', 'http_pw', 'token', 'test_volume']:
        setattr(config_test, v, os.environ[v])

    for v in ['test_skids', 'test_annotations']:
        setattr(config_test, v, os.environ[v].split(','))
else:
    missing = [v for v in required_variables if v not in os.environ]
    print('Missing environment variables:', ', '.join(missing))
    print('Falling back to config_test.py')
    try:
        import config_test
    except:
        raise ImportError('Unable to import configuration file.')


class TestModules(unittest.TestCase):
    """Test individual module import. """
    def test_imports(self):
        mods = ['morpho', 'core', 'plotting', 'graph', 'graph_utils', 'core',
                'connectivity', 'user_stats', 'cluster', 'resample',
                'intersect', 'fetch', 'scene3d']

        for m in mods:
            _ = importlib.import_module('pymaid.{}'.format(m))


class TestFetch(unittest.TestCase):
    """Test pymaid.fetch """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

    def test_get_annotation_list(self):
        remote_annotation_list_url = self.rm._get_annotation_list()
        self.assertIn('annotations', self.rm.fetch(remote_annotation_list_url))

    def test_eval_skids(self):
        """ Test skeleton ID evaluation. """
        self.assertEqual(pymaid.eval_skids(
            ['12345', '67890'], remote_instance=self.rm), ['12345', '67890'])
        self.assertEqual(pymaid.eval_skids(
            [12345, 67890], remote_instance=self.rm), ['12345', '67890'])
        self.assertIsNotNone(pymaid.eval_skids(
            'annotation:{}'.format(config_test.test_annotations[0]),
            remote_instance=self.rm))

    def test_neuron_exists(self):
        self.assertIsInstance(pymaid.neuron_exists(
            config_test.test_skids[0], remote_instance=self.rm), bool)
        self.assertIsInstance(pymaid.neuron_exists(
            config_test.test_skids, remote_instance=self.rm), dict)

    def test_get_neuron_list(self):
        self.assertIsInstance(pymaid.get_neuron_list(
            node_count=20000, remote_instance=self.rm), list)

    def test_get_user_list(self):
        self.assertIsInstance(pymaid.get_user_list(
            remote_instance=self.rm), pd.DataFrame)

    def test_get_history(self):
        self.assertIsInstance(pymaid.get_history(
            remote_instance=self.rm), pd.Series)

    def test_get_annotated_skids(self):
        self.assertIsInstance(pymaid.get_skids_by_annotation(
            config_test.test_annotations[0], remote_instance=self.rm), list)

    def test_get_annotations(self):
        self.assertIsInstance(pymaid.get_annotations(
            config_test.test_skids, remote_instance=self.rm), dict)
        self.assertIsInstance(pymaid.get_annotation_details(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)

    def test_get_neuron(self):
        self.assertIsInstance(pymaid.get_neuron(
            config_test.test_skids, remote_instance=self.rm), pymaid.CatmaidNeuronList)

    def test_get_neuron2(self):
        self.assertIsInstance(pymaid.get_arbor(
            config_test.test_skids[0], remote_instance=self.rm), pd.DataFrame)

    def test_get_treenode_table(self):
        self.assertIsInstance(pymaid.get_treenode_table(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)

    def test_get_review(self):
        self.assertIsInstance(pymaid.get_review(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)

    def test_get_volume(self):
        self.assertIsInstance(pymaid.get_volume(
            config_test.test_volume, remote_instance=self.rm), dict)

    def test_get_logs(self):
        self.assertIsInstance(pymaid.get_logs(
            remote_instance=self.rm), pd.DataFrame)

    def test_get_contributor_stats(self):
        self.assertIsInstance(pymaid.get_contributor_statistics(
            config_test.test_skids, remote_instance=self.rm), pd.Series)

    def test_get_partners(self):
        self.assertIsInstance(pymaid.get_partners(
            config_test.test_skids[0], remote_instance=self.rm), pd.DataFrame)

    def test_get_names(self):
        names = pymaid.get_names(
            config_test.test_skids, remote_instance=self.rm)
        self.assertIsInstance(names, dict)
        self.assertIsInstance(pymaid.get_skids_by_name(
            list(names.values()), remote_instance=self.rm), pd.DataFrame)

    def test_get_cn_table(self):
        self.assertIsInstance(pymaid.get_partners('annotation:%s' % config_test.test_annotations[
                              0], remote_instance=self.rm), pd.DataFrame)

    def test_get_connectors(self):
        cn = pymaid.get_connectors(
            config_test.test_skids, remote_instance=self.rm)
        self.assertIsInstance(cn, pd.DataFrame)
        self.assertIsInstance(pymaid.get_connector_details(
            cn.connector_id.tolist(), remote_instance=self.rm), pd.DataFrame)

    def test_get_partners_in_volume(self):
        self.assertIsInstance(pymaid.get_partners_in_volume(config_test.test_skids[0],
                                                            config_test.test_volume),
                              pd.DataFrame)

    def test_node_details(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_node_details(n.nodes.sample(100).treenode_id.values),
                              pd.DataFrame)

    def test_skid_from_treenode(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_skid_from_treenode(n.nodes.iloc[0].treenode_id),
                              dict)

    def test_get_edges(self):
        self.assertIsInstance(pymaid.get_edges(config_test.test_skids),
                              pd.DataFrame)

    def test_connectors_between(self):
        self.assertIsInstance(pymaid.get_connectors_between(config_test.test_skids,
                                                            config_test.test_skids),
                              pd.DataFrame)

    def test_user_annotations(self):
        ul = pymaid.get_user_list()
        self.assertIsInstance(pymaid.get_user_annotations(ul.sample(1).iloc[0].id),
                              pd.DataFrame)

    def test_has_soma(self):
        self.assertIsInstance(pymaid.has_soma(config_test.test_skids[0]),
                              dict)

    def test_treenode_info(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_treenode_info(n.nodes.treenode_id.values[0:100]),
                              pd.DataFrame)

    def test_treenode_tags(self):
        n = pymaid.get_neuron(config_test.test_skids[0])
        self.assertIsInstance(pymaid.get_node_tags(n.nodes.treenode_id.values[0:100],
                                                   node_type='TREENODE'),
                              dict)

    def test_review_details(self):
        self.assertIsInstance(pymaid.get_review_details(config_test.test_skids[0]),
                              pd.DataFrame)

    def test_find_neurons(self):
        self.assertIsInstance(pymaid.find_neurons(annotations=config_test.test_annotations),
                              pymaid.CatmaidNeuronList)

    def test_get_paths(self):
        paths, g = pymaid.get_paths(
            config_test.test_skids[0], config_test.test_skids[1], return_graph=True)
        self.assertIsInstance(g,
                              nx.Graph)
        self.assertIsInstance(paths,
                              list)

    def test_annotation_list(self):
        self.assertIsInstance(pymaid.get_annotation_list(),
                              pd.DataFrame)

    def test_url_to_coords(self):
        self.assertIsInstance(pymaid.url_to_coordinates((0, 0, 0), 1),
                              str)


class TestCore(unittest.TestCase):
    """Test pymaid.core """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.nl = pymaid.get_neuron('annotation:%s' % config_test.test_annotations[
            0], remote_instance=self.rm)

    def test_init(self):
        self.assertIsInstance(pymaid.CatmaidNeuron(
            config_test.test_skids[0]), pymaid.CatmaidNeuron)
        self.assertIsInstance(pymaid.CatmaidNeuronList(
            config_test.test_skids), pymaid.CatmaidNeuronList)

    def test_loaddata(self):
        # Test explicit loading
        nl = pymaid.CatmaidNeuron(config_test.test_skids[0])
        nl.get_skeleton()
        self.assertIsInstance(nl.nodes, pd.DataFrame)
        # Test implicit loading
        nl = pymaid.CatmaidNeuron(config_test.test_skids[0])
        self.assertIsInstance(nl.nodes, pd.DataFrame)

    def test_copy(self):
        self.assertIsInstance(self.nl[0].copy(), pymaid.CatmaidNeuron)
        self.assertIsInstance(self.nl.copy(), pymaid.CatmaidNeuronList)

    def test_summary(self):
        self.assertIsInstance(self.nl[0].summary(), pd.Series)
        self.assertIsInstance(self.nl.summary(), pd.DataFrame)

        self.assertIsInstance(self.nl.sum(), pd.Series)
        self.assertIsInstance(self.nl.mean(), pd.Series)

    def test_reload(self):
        self.nl.reload()
        self.assertIsInstance(self.nl, pymaid.CatmaidNeuronList)

    def test_graph_related(self):
        self.assertIsInstance(self.nl[0].graph, nx.Graph)
        self.assertIsInstance(self.nl[0].segments, list)

    def test_indexing(self):
        skids = self.nl.skeleton_id

        self.assertIsInstance(self.nl[0], pymaid.CatmaidNeuron)
        self.assertIsInstance(self.nl[:3], pymaid.CatmaidNeuronList)
        self.assertIsInstance(
            self.nl[self.nl.n_nodes > 1000], pymaid.CatmaidNeuronList)
        self.assertIsInstance(
            self.nl[list(skids[:-1])], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl - self.nl[0], pymaid.CatmaidNeuronList)
        self.assertIsInstance(
            self.nl[0] + self.nl[1], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl + self.nl[1], pymaid.CatmaidNeuronList)

        self.assertIsInstance(self.nl.sample(2), pymaid.CatmaidNeuronList)

    def test_neuron_attributes(self):
        attr = ['graph', 'simple', 'dps', 'annotations', 'partners',
                'review_status', 'nodes', 'connectors', 'presynapses',
                'postsynapses', 'gap_junctions', 'segments', 'soma',
                'root', 'tags', 'n_open_ends', 'n_end_nodes', 'n_connectors',
                'n_presynapses', 'n_postsynapses', 'cable_length']

        for a in attr:
            _ = getattr(self.nl[0], a)

    def test_neuron_functions(self):
        n = self.nl[0]
        slab = n.nodes[n.nodes.type == 'slab'].sample(1).iloc[0].treenode_id

        self.assertIsInstance(n.prune_distal_to(slab, inplace=False),
                              pymaid.CatmaidNeuron)
        self.assertIsInstance(n.prune_proximal_to(slab, inplace=False),
                              pymaid.CatmaidNeuron)
        self.assertIsInstance(n.prune_by_longest_neurite(inplace=False),
                              pymaid.CatmaidNeuron)

        self.assertIsInstance(n.reload(),
                              type(None))


class TestMorpho(unittest.TestCase):
    """ Test morphological operations """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.nl = pymaid.get_neuron(config_test.test_skids,
                                    remote_instance=self.rm)

    def test_downsampling(self):
        nl2 = self.nl.copy()
        nl2.downsample(4)
        self.assertLess(nl2.n_nodes.sum(), self.nl.n_nodes.sum())

    def test_resampling(self):
        nl2 = self.nl.resample(10000, inplace=False)
        self.assertNotEqual(nl2.n_nodes.sum(), self.nl.n_nodes.sum())

    def test_prune_by_strahler(self):
        nl2 = self.nl.prune_by_strahler(inplace=False)
        self.assertLess(nl2.n_nodes.sum(), self.nl.n_nodes.sum())

    def test_axon_dendrite_split(self):
        self.assertIsInstance(pymaid.split_axon_dendrite(self.nl[0]),
                              pymaid.CatmaidNeuronList)

    def test_segregation_index(self):
        self.assertIsInstance(pymaid.segregation_index(self.nl[0]),
                              float)

    def test_bending_flow(self):
        self.assertIsInstance(pymaid.bending_flow(self.nl[0]),
                              type(None))

    def test_flow_centrality(self):
        self.assertIsInstance(pymaid.flow_centrality(self.nl[0]),
                              type(None))

    def test_stitching(self):
        self.assertIsInstance(pymaid.stitch_neurons(self.nl[:2],
                                                    method='NONE'),
                              pymaid.CatmaidNeuron)
        self.assertIsInstance(pymaid.stitch_neurons(self.nl[:2],
                                                    method='LEAFS'),
                              pymaid.CatmaidNeuron)

    def test_averaging(self):
        self.assertIsInstance(pymaid.average_neurons(self.nl[:2]),
                              pymaid.CatmaidNeuron)

    def test_tortuosity(self):
        self.assertIsInstance(pymaid.tortuosity(self.nl[0]),
                              float)
        self.assertIsInstance(pymaid.tortuosity(self.nl),
                              pd.DataFrame)


class TestGraphs(unittest.TestCase):
    """Test pymaid.graph and pymaid.graph_utils """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.n = pymaid.get_neuron(config_test.test_skids[0],
                                   remote_instance=self.rm)

        self.n.reroot(self.n.soma)

        # Get some random leaf node
        self.leaf_id = self.n.nodes[self.n.nodes.type == 'end'].sample(
            1).iloc[0].treenode_id
        self.slab_id = self.n.nodes[self.n.nodes.type == 'slab'].sample(
            1).iloc[0].treenode_id

    def test_reroot(self):
        self.assertIsNotNone(self.n.reroot(self.leaf_id, inplace=False))

    def test_distal_to(self):
        self.assertTrue(pymaid.distal_to(self.n, self.leaf_id, self.n.root))
        self.assertFalse(pymaid.distal_to(self.n, self.n.root, self.leaf_id))

    def test_distance(self):
        leaf_id = self.n.nodes[self.n.nodes.type == 'end'].iloc[0].treenode_id

        self.assertIsNotNone(pymaid.dist_between(self.n,
                                                 leaf_id,
                                                 self.n.root))
        self.assertIsNotNone(pymaid.dist_between(self.n,
                                                 self.n.root,
                                                 leaf_id))

    def test_find_bp(self):
        self.assertIsNotNone(pymaid.find_main_branchpoint(self.n,
                                                          reroot_to_soma=False))

    def test_split_fragments(self):
        self.assertIsNotNone(pymaid.split_into_fragments(self.n,
                                                         n=2,
                                                         reroot_to_soma=False))

    def test_longest_neurite(self):
        self.assertIsNotNone(pymaid.longest_neurite(self.n,
                                                    n=2,
                                                    reroot_to_soma=False))

    def test_cut_neuron(self):
        dist, prox = pymaid.cut_neuron(
            self.n,
            self.slab_id,
        )
        self.assertNotEqual(dist.nodes.shape, prox.nodes.shape)

        # Make sure dist and prox check out
        self.assertTrue(pymaid.distal_to(self.n, dist.root, prox.root))

    def test_subset(self):
        self.assertIsInstance(pymaid.subset_neuron(self.n,
                                                   self.n.segments[0]),
                              pymaid.CatmaidNeuron)

    def test_node_sorting(self):
        self.assertIsInstance(pymaid.node_label_sorting(self.n),
                              list)


class TestConnectivity(unittest.TestCase):
    """Test pymaid.plotting """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.n = pymaid.get_neuron(config_test.test_skids[0],
                                   remote_instance=self.rm)

        self.cn_table = pymaid.get_partners(config_test.test_skids[0],
                                            remote_instance=self.rm)

        self.nB = pymaid.get_neuron(self.cn_table.iloc[0].skeleton_id,
                                    remote_instance=self.rm)

        self.adj = pymaid.adjacency_matrix(
            self.cn_table[self.cn_table.relation == 'upstream'].iloc[:10].skeleton_id.values)

    def test_adjacency_matrix(self):
        self.assertIsInstance(self.adj, pd.DataFrame)

    def test_connectivity_filter(self):
        dist, prox = pymaid.cut_neuron(
            self.n, pymaid.find_main_branchpoint(self.n))

        vol = pymaid.get_volume(config_test.test_volume)

        # Connectivity table by neuron
        self.assertIsInstance(pymaid.filter_connectivity(self.cn_table, prox),
                              pd.DataFrame)
        # Adjacency matrix by neuron
        self.assertIsInstance(pymaid.filter_connectivity(self.adj, prox),
                              pd.DataFrame)

        # Connectivity table by volume
        self.assertIsInstance(pymaid.filter_connectivity(self.cn_table, vol),
                              pd.DataFrame)
        # Adjacency matrix by volume
        self.assertIsInstance(pymaid.filter_connectivity(self.adj, vol),
                              pd.DataFrame)

    def test_calc_overlap(self):
        self.assertIsInstance(pymaid.cable_overlap(self.n, self.nB),
                              pd.DataFrame)

    def test_pred_connectivity(self):
        self.assertIsInstance(pymaid.predict_connectivity(self.n,
                                                          self.nB,
                                                          remote_instance=self.rm),
                              pd.DataFrame)


class TestCluster(unittest.TestCase):
    """Test pymaid.cluster """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

    def test_connectivity_cluster(self):
        self.assertIsInstance(pymaid.cluster_by_connectivity(config_test.test_skids),
                              pymaid.ClustResults)

    def test_synapse_cluster(self):
        self.assertIsInstance(pymaid.cluster_by_synapse_placement(config_test.test_skids),
                              pymaid.ClustResults)


class TestPlot(unittest.TestCase):
    """Test pymaid.plotting """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.nl = pymaid.get_neuron(config_test.test_skids,
                                    remote_instance=self.rm)

        self.vol = pymaid.get_volume(config_test.test_volume)

    """
    def test_plot3d_plotly(self):
        self.assertIsNotNone(self.nl.plot3d(backend='plotly'))
        self.assertIsNotNone(pymaid.plot3d(self.nl, backend='plotly'))
        self.assertIsNotNone(pymaid.plot3d([self.nl, self.vol], backend='plotly'))


    def test_plot2d(self):
        self.assertIsNotNone(self.nl.plot2d(method='2d'))
        self.assertIsNotNone(self.nl.plot2d(method='3d_complex'))
        self.assertIsNotNone(self.nl.plot2d(method='3d'))


    def test_plot3d_vispy(self):
        self.assertIsNotNone(self.nl.plot3d(backend='vispy'))
        pymaid.close3d()
        self.assertIsNotNone(pymaid.plot3d(self.nl, backend='vispy'))
        pymaid.close3d()
        self.assertIsNotNone(pymaid.plot3d([self.nl, self.vol], backend='vispy'))
        pymaid.close3d()

    def tearDown(self):
        pymaid.close3d()
        plt.clf()
    """


class TestUserStats(unittest.TestCase):
    """Test pymaid.user_stats """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.n = pymaid.get_neuron(config_test.test_skids[0],
                                   remote_instance=self.rm)

    def test_time_invested(self):
        self.assertIsInstance(pymaid.get_time_invested(
            self.n.downsample(10, inplace=False), remote_instance=self.rm), pd.DataFrame)

    def test_user_contributions(self):
        self.assertIsInstance(pymaid.get_user_contributions(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
