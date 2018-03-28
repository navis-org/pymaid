""" Test suite for pymaid

This test suite requires a bunch of variables that need to either defined
as environment variables or in a config_test.py file

    #Catmaid server url
    server_url = ''

    #Http user
    http_user = ''

    #Http pw
    http_pw = ''

    #Auth token
    token = ''  

    # Test skeleton IDs
    test_skids = []

    # Test annotations
    test_annotations = []

    # Test CATMAID volume
    test_volume = ''


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
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Silence module loggers
pymaid.set_loggers('ERROR')

# Deactivate progress bars
pymaid.set_pbars(hide=True)

# Try getting credentials from environment variable
required_variables = ['server_url', 'http_user', 'http_pw', 'token', 
                      'test_skids', 'test_annotations', 'test_volume']

if False not in [v in os.environ for v in required_variables]:
    class conf: pass
    config_test = conf()

    for v in ['server_url', 'http_user', 'http_pw', 'token', 'test_volume']:
        setattr(config_test, v, os.environ[v]) 

    for v in ['test_skids', 'test_annotations']:
        setattr(config_test, v, os.environ[v].split(',')) 
else:
    missing = [v for v in required_variables if v not in os.environ]
    print('Missing some environment variables:', ','.join(missing))
    print('Falling back to config_test.py')
    try:
        import config_test
    except:
        raise ImportError('Unable to import configuration file.')

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

    def test_get_neuron(self):
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
        names = pymaid.get_names(config_test.test_skids, remote_instance=self.rm)
        self.assertIsInstance(names, dict)
        self.assertIsInstance(pymaid.get_skids_by_name(
            list(names.values()), remote_instance=self.rm), pd.DataFrame)

    def test_get_edges(self):
        self.assertIsInstance(pymaid.get_partners('annotation:%s' % config_test.test_annotations[
                              0], remote_instance=self.rm), pd.DataFrame)

    def test_get_connectors(self):
        cn = pymaid.get_connectors(config_test.test_skids, remote_instance=self.rm)
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
        nl = pymaid.get_neuron('annotation:%s' % config_test.test_annotations[
                               0], remote_instance=self.rm)
        self.assertIsInstance(self.nl[0].graph, nx.Graph)
        self.assertIsInstance(self.nl[0].segments, list)

    def test_indexing(self):        
        skids = self.nl.skeleton_id

        self.assertIsInstance(self.nl[0], pymaid.CatmaidNeuron)
        self.assertIsInstance(self.nl[:3], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl[self.nl.n_nodes > 1000], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl[list(skids[:-1])], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl - self.nl[0], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl[0] + self.nl[1], pymaid.CatmaidNeuronList)
        self.assertIsInstance(self.nl + self.nl[1], pymaid.CatmaidNeuronList)

        self.assertIsInstance(self.nl.sample(2), pymaid.CatmaidNeuronList)
 

class TestMorpho(unittest.TestCase):
    """ Test morphological operations """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.nl = pymaid.get_neuron( config_test.test_skids, 
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


class TestPlot(unittest.TestCase):
    """Test pymaid.plot """

    def setUp(self):
        self.rm = pymaid.CatmaidInstance(
            config_test.server_url,
            config_test.http_user,
            config_test.http_pw,
            config_test.token,
            logger_level='ERROR')

        self.nl = pymaid.get_neuron( config_test.test_skids, 
                                     remote_instance=self.rm)

        self.vol = pymaid.get_volume( config_test.test_volume )

    def test_plot3d_vispy(self):
        self.assertIsNotNone(self.nl.plot3d(backend='vispy'))
        pymaid.close3d()
        self.assertIsNotNone(pymaid.plot3d(self.nl, backend='vispy'))
        pymaid.close3d()
        self.assertIsNotNone(pymaid.plot3d([self.nl, self.vol], backend='vispy'))
        pymaid.close3d()

    def test_plot3d_plotly(self):
        self.assertIsNotNone(self.nl.plot3d(backend='plotly'))        
        self.assertIsNotNone(pymaid.plot3d(self.nl, backend='plotly'))        
        self.assertIsNotNone(pymaid.plot3d([self.nl, self.vol], backend='plotly'))

    def test_plot2d(self):
        self.assertIsNotNone(self.nl.plot2d(method='2d'))
        self.assertIsNotNone(self.nl.plot2d(method='3d_complex'))
        self.assertIsNotNone(self.nl.plot2d(method='3d'))

    def tearDown(self):
        pymaid.close3d()
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
        self.assertIsInstance(pymaid.get_time_invested(
            config_test.test_skids[0], remote_instance=self.rm), pd.DataFrame)
    """
    
    def test_user_contributions(self):
        self.assertIsInstance(pymaid.get_user_contributions(
            config_test.test_skids, remote_instance=self.rm), pd.DataFrame)
    """  

if __name__ == '__main__':
    unittest.main()
