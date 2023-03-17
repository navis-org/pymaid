from collections.abc import Iterable
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
import warnings

import networkx as nx

from .. import config, cache, utils
from . import get_annotation_id

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
            logger.warning("Neuron %s is modelled by 0 skeletons; skipping", data["id"])
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


def join_ids(ids: Iterable[int]) -> str:
    return ",".join(str(n) for n in ids)


def join_id_sets(id_sets: Iterable[Iterable[int]]) -> List[str]:
    return [join_ids(ids) for ids in id_sets]


@cache.undo_on_error
def _get_entities(
    types: Optional[Iterable[str]] = None,
    with_annotations: Optional[bool] = None,
    annotated_with: Optional[Iterable[Iterable[int]]] = None,
    not_annotated_with: Optional[Iterable[Iterable[int]]] = None,
    sub_annotated_with: Optional[Iterable[int]] = None,
    *,
    remote_instance=None,
):
    logger.info("Fetching entity graph; may be slow")

    remote_instance = utils._eval_remote_instance(remote_instance)
    post: Dict[str, Any] = dict()

    if types is not None:
        post["types"] = list(types)
    if with_annotations is not None:
        post["with_annotations"] = bool(with_annotations)
    if annotated_with is not None:
        post["annotated_with"] = join_id_sets(annotated_with)
    if not_annotated_with is not None:
        post["not_annotated_with"] = join_id_sets(not_annotated_with)
    if sub_annotated_with is not None:
        post["sub_annotated_with"] = join_ids(sub_annotated_with)

    query_url = remote_instance.make_url(
        remote_instance.project_id, "annotations", "query-targets"
    )
    return remote_instance.fetch(query_url, post)


def to_nested_and_flat(objs):
    if isinstance(objs, (str, bytes)) or not isinstance(objs, Iterable):
        return objs, objs

    nested = []
    flattened = []

    for item in objs:
        inner_nested, inner_flattened = to_nested_and_flat(item)
        nested.append(inner_nested)
        flattened.extend(utils._make_iterable(inner_flattened))

    return nested, flattened


def map_nested(nested, mapping):
    if isinstance(nested, (str, bytes)) or not isinstance(nested, Iterable):
        return mapping[nested]

    return [map_nested(item, mapping) for item in nested]


def _get_annotation_ids(
    *ann_lols: Iterable[Iterable[Union[int, str]]], remote_instance=None
) -> List[List[List[int]]]:
    nested, flattened = to_nested_and_flat(ann_lols)
    id_mapping = {None: None}
    names = []
    for name_or_id in flattened:
        if isinstance(name_or_id, str):
            names.append(name_or_id)
        else:
            id_mapping[name_or_id] = name_or_id

    ann_ids = get_annotation_id(names, remote_instance=remote_instance)

    id_mapping.update((n, int(aid)) for n, aid in ann_ids.items())

    return map_nested(nested, id_mapping)


def get_entity_graph(
    types: Optional[Iterable[EntityType]] = None,
    by_name=False,
    annotated_with: Optional[Iterable[Iterable[Union[int, str]]]] = None,
    not_annotated_with: Optional[Iterable[Iterable[Union[int, str]]]] = None,
    sub_annotated_with: Optional[Iterable[Union[int, str]]] = None,
    *,
    remote_instance=None,
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
    annotated_with : Optional[Iterable[Iterable[Union[int, str]]]], default None
        If not None, only include entities annotated with these annotations.
        Can be integer IDs or str names (not IDs as strings!).
        The inner sets are combined with OR.
        The outer iterable is combined with AND.
        e.g. for ``[["a", "b"], ["c"]]``, entities must be annotated with ``"c"``,
        and at least one of ``"a"`` or ``"b"``,
    not_annotated_with: Optional[Iterable[Iterable[Union[int, str]]]], default None
        If not None, only include entites NOT annotated with these.
        See ``annotated_with`` for more usage details.
    sub_annotated_with: Optional[Iterable[Union[int, str]]], default None
        Which annotations in the ``annotated_with``, ``not_annotated_with``
        sets to expand into all their sub-annotations (each as an OR group).
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
    remote_instance = utils._eval_remote_instance(remote_instance)

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

    (
        annotated_with_ids,
        not_annotated_with_ids,
        sub_annotated_with_ids,
    ) = _get_annotation_ids(
        annotated_with,
        not_annotated_with,
        sub_annotated_with,
        remote_instance=remote_instance,
    )

    _, flattened_subs = to_nested_and_flat(sub_annotated_with_ids)

    data = _get_entities(
        types=etypes,
        with_annotations=True,
        annotated_with=annotated_with_ids,
        not_annotated_with=not_annotated_with_ids,
        sub_annotated_with=flattened_subs,
        remote_instance=remote_instance,
    )

    g = entities_to_ann_graph(data, by_name)

    if use_skeletons:
        g = neurons_to_skeletons(g, by_name)

    return g


def get_annotation_graph(
    annotations_by_id=False, skeletons_by_id=True, remote_instance=None
) -> nx.DiGraph:
    """DEPRECATED. Get a networkx DiGraph of (meta)annotations and skeletons.

    This function is deprecated.
    Use :func:`pymaid.get_entity_graph` instead.

    Can be slow for large projects.

    Nodes in the graph have data:

    Skeletons have

    - id
    - is_skeleton = True
    - neuron_id (different to the skeleton ID)
    - name

    Annotations have

    - id
    - name
    - is_skeleton = False

    Edges in the graph have

    - is_meta_annotation (whether it is between two annotations)

    Parameters
    ----------
    annotations_by_id : bool, default False
        Whether to index nodes representing annotations by their integer ID
        (uses name by default)
    skeletons_by_id : bool, default True
        whether to index nodes representing skeletons by their integer ID
        (True by default, otherwise uses the neuron name)
    remote_instance : optional CatmaidInstance

    Returns
    -------
    networkx.DiGraph
    """
    warnings.warn(
        DeprecationWarning("get_annotation_graph is deprecated; use get_entity_graph")
    )

    data = _get_entities(
        types=None, with_annotations=True, remote_instance=remote_instance
    )

    ann_ref = "id" if annotations_by_id else "name"
    skel_ref = "id" if skeletons_by_id else "name"

    g = nx.DiGraph()

    for e in data["entities"]:
        is_meta_ann = False

        if e.get("type") == "neuron":
            skids = e.get("skeleton_ids") or []
            if len(skids) != 1:
                logger.warning(
                    "Neuron with id %s is modelled by %s skeletons, ignoring",
                    e["id"],
                    len(skids),
                )
                continue
            node_data = {
                "name": e["name"],
                "neuron_id": e["id"],
                "is_skeleton": True,
                "id": skids[0],
            }
            node_id = node_data[skel_ref]
        else:  # is an annotation
            node_data = {
                "is_skeleton": False,
                "id": e["id"],
                "name": e["name"],
            }
            node_id = node_data[ann_ref]
            is_meta_ann = True

        anns = e.get("annotations", [])
        if not anns:
            g.add_node(node_id, **node_data)
            continue

        for ann in e.get("annotations", []):
            g.add_edge(
                ann[ann_ref],
                node_id,
                is_meta_annotation=is_meta_ann,
            )

        g.nodes[node_id].update(**node_data)

    return g
