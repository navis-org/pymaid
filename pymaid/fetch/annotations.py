from collections.abc import Sequence
from typing import (
    Optional,
    Literal,
    Callable,
    List,
    DefaultDict,
    Union,
    Tuple,
    Dict,
    Any,
)
from collections import defaultdict
from itertools import chain

import networkx as nx

from .. import config, cache, utils

logger = config.get_logger(__name__)


class UnknownEntityTypeError(RuntimeError):
    _known = {"neuron", "annotation", "volume"}

    def __init__(self, etype: str):
        super().__init__(
            f"Entity type {repr(etype)} unknown; should be one of {', '.join(sorted(self._known))}"
        )

    @classmethod
    def raise_for_etype(cls, etype: str):
        if etype not in cls._known:
            raise cls(etype)


class AmbiguousEntityNameError(RuntimeError):
    def __init__(self, name: str):
        super().__init__(f"Entity has non-unique name {repr(name)}; use IDs instead")


def get_id_key(by_name: bool):
    return "name" if by_name else "id"


def entities_to_ann_graph(data: dict, by_name: bool):
    g = nx.DiGraph()
    id_key = get_id_key(by_name)

    edges = []

    for e in data["entities"]:
        etype = e.get("type")

        UnknownEntityTypeError.raise_for_etype(etype)

        is_meta_ann = False

        ndata = {
            "name": e["name"],
            "id": e["id"],
            "type": etype,
        }
        node_id = ndata[id_key]

        if etype == "neuron":
            skids = e.get("skeleton_ids") or []
            ndata["skeleton_ids"] = skids

        elif etype == "annotation":
            is_meta_ann = True

        if by_name and node_id in g.nodes:
            raise AmbiguousEntityNameError(node_id)

        g.add_node(node_id, **ndata)

        for ann in e.get("annotations", []):
            edges.append((ann[id_key], node_id, {"is_meta_annotation": is_meta_ann}))

    g.add_edges_from(edges)

    return g


def noop(arg):
    return arg


def neurons_to_skeletons(
    g: nx.DiGraph,
    by_name: bool,
    select_skeletons: Callable[[List[int]], List[int]] = noop,
):
    id_key = get_id_key(by_name)

    nodes_to_replace: DefaultDict[
        Union[str, int], List[Tuple[Union[str, int], Dict[str, Any]]]
    ] = defaultdict(list)

    for node_id, data in g.nodes(data=True):
        if data["type"] != "neuron":
            continue

        skids = data["skeleton_ids"]
        if len(skids) == 0:
            logger.warning("Neuron %s is modelled by 0 skeletons; skipping")
            nodes_to_replace[node_id]  # ensure this exists
            continue

        if len(skids) > 1:
            skids = select_skeletons(skids)

        if by_name and len(skids) > 1:
            raise AmbiguousEntityNameError(data["name"])

        for skid in skids:
            sk_data = {
                "id": skid,
                "name": data["name"],
                "type": "skeleton",
                "neuron_id": data["id"],
            }
            nid = sk_data[id_key]

            nodes_to_replace[node_id].append((nid, sk_data))

    edges_to_add = []
    for src, tgt, edata in g.in_edges(nodes_to_replace, data=True):
        for new_tgt, _ in nodes_to_replace[tgt]:
            edges_to_add.append((src, new_tgt, edata))

    g.remove_nodes_from(nodes_to_replace)
    g.add_nodes_from(chain.from_iterable(nodes_to_replace.values()))
    g.add_edges_from(edges_to_add)

    return g


# todo: replace with strenum
EntityType = Literal["neuron", "annotation", "volume", "skeleton"]


@cache.undo_on_error
def _get_entities(entity_types, remote_instance):
    remote_instance = utils._eval_remote_instance(remote_instance)
    post = {
        "with_annotations": True,
    }
    if entity_types is not None:
        post["types"] = list(entity_types)

    query_url = remote_instance.make_url(
        remote_instance.project_id, "annotations", "query-targets"
    )
    return remote_instance.fetch(query_url, post)


def get_annotation_graph(
    types: Optional[Sequence[EntityType]] = None, by_name=False, remote_instance=None
) -> nx.DiGraph:
    """Get a networkx DiGraph of semantic objects.

    Can be slow for large projects.

    Note that CATMAID distinguishes between neurons
    (semantic objects which can be named and annotated)
    and skeletons (spatial objects which can model neurons).
    Most pymaid (and CATMAID) functions use the skeleton ID,
    rather than the neuron ID,
    and assume that a neuron is modeled by a single skeleton.
    To replace neurons in the graph with the skeletons they are modelled by,
    include ``"skeleton"`` in the ``types`` argument
    (this is mutually exclusive with ``"neuron"``).

    Nodes in the graph have data:

    - id: int
    - name: str
    - type: str, one of "neuron", "annotation", "volume", "skeleton"

    Neurons additionally have

    - skeleton_ids: list[int]

    Skeletons additionally have

    - neuron_id: int

    Edges in the graph have

    - is_meta_annotation (bool): whether it is between two annotations

    Parameters
    ----------
    types : optional sequence of str, default None
        Which types of entity to fetch.
        Choices are "neuron", "annotation", "volume", "skeleton";
        "neuron" and "skeleton" are mutually exclusive.
        None uses CATMAID default ("neuron", "annotation").
    by_name : bool, default False
        If True, use the entity's name rather than its integer ID.
        This can be convenient but has a risk of name collisions,
        which will raise errors.
        In particular, name collisions will occur if ``types`` includes ``"skeleton"``
        and a neuron is modelled by more than one skeleton.
    remote_instance : optional CatmaidInstance

    Returns
    -------
    networkx.DiGraph

    Raises
    ------
    UnknownEntityTypeError
        CATMAID returned an entity type pymaid doesn't know how to interpret.
    AmbiguousEntityNameError
        When ``by_name=True`` is used, and there are naming collisions.
    """

    use_skeletons = False

    if types is None:
        etypes = None
    else:
        etypes = set(types)
        if "skeleton" in etypes:
            if "neuron" in etypes:
                raise ValueError("'skeleton' and 'neuron' types are mutually exclusive")

            etypes.add("neuron")
            etypes.remove("skeleton")
            use_skeletons = True

        if not etypes:
            return nx.DiGraph()

    data = _get_entities(etypes, remote_instance)

    g = entities_to_ann_graph(data, by_name)

    if use_skeletons:
        g = neurons_to_skeletons(g, by_name)

    return g
