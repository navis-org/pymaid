import pytest
import pymaid
import navis as ns
import numpy as np

SEED = 1991

@pytest.fixture(scope="module")
def neuron_list(client, skids):
    return pymaid.get_neuron(skids[0:2], remote_instance=client)


@pytest.fixture(scope="module")
def neuron(neuron_list):
    n = neuron_list[0]
    return n.reroot(n.soma, inplace=False)

@pytest.fixture(scope="module")
def leaf_slab(neuron):
    rng = np.random.default_rng(SEED)
    leaf_id = neuron.nodes[neuron.nodes.type == 'end'].sample(
        1, random_state=rng).iloc[0].node_id
    slab_id = neuron.nodes[neuron.nodes.type == 'slab'].sample(
        1, random_state=rng).iloc[0].node_id
    return (leaf_id, slab_id)


def test_reroot(neuron_list, neuron, leaf_slab, use_igraph):
    leaf_id, slab_id = leaf_slab
    assert neuron.reroot(leaf_id, inplace=False) is not None
    assert neuron_list.reroot(neuron_list.soma, inplace=False) is not None


def test_distal_to(neuron, leaf_slab, use_igraph):
    leaf_id, slab_id = leaf_slab
    assert ns.distal_to(neuron, leaf_id, neuron.root)
    assert not ns.distal_to(neuron, neuron.root, leaf_id)


def test_distance(neuron, use_igraph):
    leaf_id = neuron.nodes[neuron.nodes.type == 'end'].iloc[0].node_id

    assert ns.dist_between(neuron, leaf_id, neuron.root) is not None
    assert ns.dist_between(neuron, neuron.root, leaf_id) is not None


def test_find_bp(neuron, use_igraph):
    assert ns.find_main_branchpoint(neuron, reroot_soma=False) is not None


def test_split_fragments(neuron, use_igraph):
    assert ns.split_into_fragments(neuron, n=2, reroot_soma=False) is not None


def test_longest_neurite(neuron, use_igraph):
    assert ns.longest_neurite(neuron, n=2, reroot_soma=False) is not None


def test_cut_neuron(neuron, leaf_slab, use_igraph):
    leaf_id, slab_id = leaf_slab
    dist, prox = ns.cut_skeleton(neuron, slab_id)
    assert dist.nodes.shape != prox.nodes.shape

    # Make sure dist and prox check out
    assert ns.distal_to(neuron, dist.root, prox.root)


def test_subset(neuron, use_igraph):
    assert isinstance(
        ns.subset_neuron(neuron, neuron.segments[0]),
        pymaid.CatmaidNeuron
    )


def test_node_sorting(neuron, use_igraph):
    result = ns.graph.node_label_sorting(neuron)
    assert isinstance(result, np.ndarray)


def test_geodesic_matrix(neuron, use_igraph):
    geo = ns.geodesic_matrix(neuron)
