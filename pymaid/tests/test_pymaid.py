""" Test suite for pymaid.pymaid

This test suite requires that you have a config_test.py in the /pymaid/test 
that contains your CATMAID credentials credentials. 

config_test.py should look like this:

    #Catmaid server url
    server_url = ''

    #Http user
    http_user = ''

    #Http pw
    http_pw = ''

    #Auth token
    token = ''  


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

from pymaid import pymaid, core, plot, user_stats, igraph_catmaid, cluster
import pandas as pd
import numpy as np
import igraph
import matplotlib.pyplot as plt

# Silence most loggers
user_stats.module_logger.setLevel('ERROR')
igraph_catmaid.module_logger.setLevel('ERROR')
cluster.module_logger.setLevel('ERROR')
plot.module_logger.setLevel('ERROR')

try:
    import config_test
except:
    raise ImportError('Unable to import configuration file.')

# These skeleton ids and annotations will only work on FAFB
example_skids = ['16', '2333007']
example_annotations = ['glomerulus DA1', 'MVP2', 'mAL']
example_volumes = ['v13.LH_R', 'v13.AL_R']


class TestPymaid(unittest.TestCase):
    """Test pymaid.pymaid """

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
        self.assertEqual(pymaid.eval_skids(
            ['12345', '67890'], remote_instance=self.rm), ['12345', '67890'])
        self.assertEqual(pymaid.eval_skids(
            [12345, 67890], remote_instance=self.rm), ['12345', '67890'])
        self.assertIsNotNone(pymaid.eval_skids(
            example_annotations, remote_instance=self.rm))

    def test_neuron_exists(self):
        self.assertIsInstance(pymaid.neuron_exists(
            example_skids[0], remote_instance=self.rm), bool)
        self.assertIsInstance(pymaid.neuron_exists(
            example_skids, remote_instance=self.rm), dict)

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
            example_annotations[0], remote_instance=self.rm), list)

    def test_get_annotations(self):
        self.assertIsInstance(pymaid.get_annotations(
            example_skids, remote_instance=self.rm), dict)
        self.assertIsInstance(pymaid.get_annotation_details(
            example_skids, remote_instance=self.rm), pd.DataFrame)

    def test_get_neuron(self):
        self.assertIsInstance(pymaid.get_neuron(
            example_skids, remote_instance=self.rm), core.CatmaidNeuronList)

    def test_get_neuron(self):
        self.assertIsInstance(pymaid.get_arbor(
            example_skids[0], remote_instance=self.rm), pd.DataFrame)

    def test_get_treenode_table(self):
        self.assertIsInstance(pymaid.get_treenode_table(
            example_skids, remote_instance=self.rm), pd.DataFrame)

    def test_get_review(self):
        self.assertIsInstance(pymaid.get_review(
            example_skids, remote_instance=self.rm), pd.DataFrame)

    def test_get_volume(self):
        self.assertIsInstance(pymaid.get_volume(
            example_volumes[0], remote_instance=self.rm), dict)

    def test_get_logs(self):
        self.assertIsInstance(pymaid.get_logs(
            remote_instance=self.rm), pd.DataFrame)

    def test_get_contributor_stats(self):
        self.assertIsInstance(pymaid.get_contributor_statistics(
            example_skids, remote_instance=self.rm), pd.DataFrame)

    def test_get_partners(self):
        self.assertIsInstance(pymaid.get_partners(
            example_skids[0], remote_instance=self.rm), pd.DataFrame)

    def test_get_names(self):
        names = pymaid.get_names(example_skids, remote_instance=self.rm)
        self.assertIsInstance(names, dict)
        self.assertIsInstance(pymaid.get_skids_by_name(
            list(names.values()), remote_instance=self.rm), pd.DataFrame)

    def test_get_edges(self):
        self.assertIsInstance(pymaid.get_partners('annotation:%s' % example_annotations[
                              0], remote_instance=self.rm), pd.DataFrame)

    def test_get_connectors(self):
        cn = pymaid.get_connectors(example_skids, remote_instance=self.rm)
        self.assertIsInstance(cn, pd.DataFrame)
        self.assertIsInstance(pymaid.get_connector_details(
            cn.connector_id.tolist(), remote_instance=self.rm), pd.DataFrame)


class TestCore(unittest.TestCase):
    """Test pymaid.core """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

    def test_init(self):
        self.assertIsInstance(core.CatmaidNeuron(
            example_skids[0]), core.CatmaidNeuron)
        self.assertIsInstance(core.CatmaidNeuronList(
            example_skids), core.CatmaidNeuronList)

    def test_copy(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        self.assertIsInstance(nl[0].copy(), core.CatmaidNeuron)
        self.assertIsInstance(nl.copy(), core.CatmaidNeuronList)

    def test_summary(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        self.assertIsInstance(nl[0].summary(), pd.Series)
        self.assertIsInstance(nl.summary(), pd.DataFrame)

        self.assertIsInstance(nl.sum(), pd.Series)
        self.assertIsInstance(nl.mean(), pd.Series)

    def test_downsampling(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        nl2 = nl.copy()
        nl2.downsample(4)
        self.assertLess(nl2.n_nodes.sum(), nl.n_nodes.sum())

    def test_prune_by_strahler(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        nl2 = nl.copy()
        nl2.prune_by_strahler()
        self.assertLess(nl2.n_nodes.sum(), nl.n_nodes.sum())

    def test_reload(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        nl.reload()
        self.assertIsInstance(nl, core.CatmaidNeuronList)

    def test_igraph_related(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        self.assertIsInstance(nl[0].igraph, igraph.Graph)
        self.assertIsInstance(nl[0].slabs, list)

    def test_indexing(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        skids = nl.skeleton_id

        self.assertIsInstance(nl[0], core.CatmaidNeuron)
        self.assertIsInstance(nl[:3], core.CatmaidNeuronList)
        self.assertIsInstance(nl[nl.n_nodes > 1000], core.CatmaidNeuronList)
        self.assertIsInstance(nl[list(skids[:-5])], core.CatmaidNeuronList)
        self.assertIsInstance(nl - nl[0], core.CatmaidNeuronList)
        self.assertIsInstance(nl[0] + nl[1], core.CatmaidNeuronList)
        self.assertIsInstance(nl + nl[1], core.CatmaidNeuronList)

        self.assertIsInstance(nl.sample(2), core.CatmaidNeuronList)


class TestPlot(unittest.TestCase):
    """Test pymaid.plot """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

    def test_plot3d_vispy(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)

        self.assertIsNotNone(nl.plot3d())
        self.assertIsNotNone(nl.plot3d(volumes=example_volumes))

    def test_plot3d_plotly(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)

        self.assertIsNotNone(nl.plot3d(backend='plotly'))
        self.assertIsNotNone(
            nl.plot3d(backend='plotly', volumes=example_volumes))

    def test_plot2d(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)

        self.assertIsNotNone(nl.plot2d())

    def tearDown(self):
        plot.close3d()
        plt.clf()


class TestUserStats(unittest.TestCase):
    """Test pymaid.user_stats """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

    def test_time_invested(self):
        self.assertIsInstance(user_stats.get_time_invested(
            example_skids, remote_instance=self.rm), pd.DataFrame)

    def test_user_contributions(self):
        self.assertIsInstance(user_stats.get_user_contributions(
            example_skids, remote_instance=self.rm), pd.DataFrame)


class TestIgraphCatmaid(unittest.TestCase):
    """Test pymaid.igraph_catmaid """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

    def test_cluster_synapse_nodes(self):
        nl = pymaid.get_neuron('annotation:%s' % example_annotations[
                               0], remote_instance=self.rm)
        self.assertIsInstance(igraph_catmaid.cluster_nodes_w_synapses(
            nl[0]), np.ndarray)

    def test_network2graph(self):
        self.assertIsInstance(igraph_catmaid.network2graph(
            'annotation:' + example_annotations[0], remote_instance=self.rm),
            igraph.Graph)


if __name__ == '__main__':
    unittest.main()
