from abc import ABC, abstractmethod
from functools import cache
from typing import Optional, Union, List, Tuple
import re

import networkx as nx

import pymaid
from .utils import _eval_remote_instance
from .client import CatmaidInstance
from .core import CatmaidNeuron
from .config import get_logger
from .fetch.annotations import get_annotation_table, get_entity_graph

__all__ = [
    "NeuronLabeller",
    "SkeletonId",
    "NeuronName",
    "Annotations",
    "ThinNeuron"
]

logger = get_logger(__name__)


component_re = re.compile(r"%(?P<idx>\d+|f)(\{(?P<sep>.*)\})?")
whitespace_re = re.compile(r"\s+")

DEFAULT_SEP = ", "


class ThinNeuron:
    """Class containing some very basic information about a neuron as needed by `NeuronLabeller`.

    Unknown fields are fetched lazily as required.
    """
    def __init__(
        self,
        skeleton_id: Optional[int] = None,
        name: Optional[str] = None,
        annotations: Optional[List[str]] = None,
        remote_instance: Optional[CatmaidInstance] = None,
    ) -> None:
        """
        At least one of ``skeleton_id`` and ``name`` should be given
        if additional fields need to be fetched.

        Parameters
        ----------
        skeleton_id : Optional[int], optional
            If None, determined from name.
        name : Optional[str], optional
            If None, determined from skeleton ID.
        annotations : Optional[List[str]], optional
            If None, determined from skeleton ID or name.
        remote_instance : Optional[CatmaidInstance], optional
            If None, uses global instance.
        """
        self._skeleton_id = skeleton_id
        self._name = name
        self._annotations = annotations
        self._remote_instance_inner = remote_instance

    @property
    def _remote_instance(self):
        if self._remote_instance_inner is None:
            self._remote_instance_inner = _eval_remote_instance(None)
        return self._remote_instance_inner

    @property
    def skeleton_id(self) -> int:
        if self._skeleton_id is None:
            if self._name is None:
                raise ValueError("Neither skeleton ID nor name is known")
            df = pymaid.get_skids_by_name([self._name])
            if len(df) != 1:
                raise ValueError(
                    f"Did not find unique skeleton ID for name '{self.name}'"
                )
            self._skeleton_id = int(df["skeleton_id"][0])
        return self._skeleton_id

    @property
    def name(self) -> str:
        if self._name is None:
            if self._skeleton_id is None:
                raise ValueError("Neither skeleton ID nor name is known")
            skid_to_name = pymaid.get_names([self._skeleton_id])
            self._name = skid_to_name[str(self._skeleton_id)]
        return self._name

    @property
    def annotations(self) -> List[str]:
        if self._annotations is None:
            skid = self.skeleton_id
            skid_to_anns = pymaid.get_annotations(skid)
            self._annotations = skid_to_anns.get(str(skid), [])
        return self._annotations

    def to_neuron(self, *args, **kwargs) -> CatmaidNeuron:
        x = self._skeleton_id if self._skeleton_id is not None else self.name
        return pymaid.get_neuron(x, *args, **kwargs)

    @classmethod
    def from_neuron(cls, nrn: CatmaidNeuron):
        return cls(nrn.skeleton_id, nrn.name, nrn.annotations, nrn._remote_instance)


class LabelComponent(ABC):
    @abstractmethod
    def label(self, nrn: Union[ThinNeuron, CatmaidNeuron], sep: Optional[str] = None):
        """Extract information from a neuron for labelling purposes.

        Parameters
        ----------
        nrn : Union[ThinNeuron, CatmaidNeuron]
            Neuron to get information about.
        sep : Optional[str], optional
            If the information has multiple parts (e.g. annotations),
            join them with this string.
            By default None (CATMAID default ``", "``).
        """
        pass


class SkeletonId(LabelComponent):
    """`LabelComponent` which adds a skeleton ID modelling a neuron to its label."""
    def label(self, nrn: Union[ThinNeuron, CatmaidNeuron], sep: Optional[str] = None):
        return str(nrn.skeleton_id)


class NeuronName(LabelComponent):
    """`LabelComponent` which adds a neuron's name to its label."""
    def label(self, nrn: Union[ThinNeuron, CatmaidNeuron], sep: Optional[str] = None):
        return nrn.name


class Annotations(LabelComponent):
    """`LabelComponent` which adds annotations to a neuron's label."""
    def __init__(
        self, annotator_name: Optional[str] = None, annotated_with: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        annotator_name : Optional[str], optional
            Only include annotations created by a user of this name,
            by default None (do not filter)
        annotated_with : Optional[str], optional
            Only include annotations which have this meta-annotation,
            by default None (do not filter)
        """
        self.annotator_name = annotator_name
        self.annotated_with = annotated_with
        super().__init__()

    def _filter_by_author(
        self, annotations: List[str], remote_instance: CatmaidInstance
    ) -> List[str]:
        if self.annotator_name is None or not annotations:
            return annotations

        allowed = annotations_by_user(
            self.annotator_name, remote_instance=remote_instance
        )
        return [a for a in annotations if a in allowed]

    def _filter_by_annotation(
        self, annotations: List[str], remote_instance: CatmaidInstance
    ) -> List[str]:
        if self.annotated_with is None or not annotations:
            return annotations

        allowed = annotations_by_annotation(
            self.annotated_with, False, remote_instance=remote_instance
        )
        return [a for a in annotations if a in allowed]

    def label(self, nrn: Union[ThinNeuron, CatmaidNeuron], sep: Optional[str] = None):
        sep_str = DEFAULT_SEP if sep is None else sep
        anns = list(nrn.annotations)
        anns = self._filter_by_author(anns, nrn._remote_instance)
        anns = self._filter_by_annotation(anns, nrn._remote_instance)
        return sep_str.join(anns)


def dedup_whitespace(s: str):
    if not s:
        return ""
    return whitespace_re.sub(s, " ")


@cache
def parse_components(
    fmt: str,
) -> Tuple[List[str], List[Tuple[str, int, Optional[str]]]]:
    joiners = []
    components = []
    last_end = 0
    for component in component_re.finditer(fmt):
        joiners.append(dedup_whitespace(fmt[last_end : component.start()]))
        last_end = component.end()
        d = component.groupdict()
        idx_str = d["idx"]
        idx = None if idx_str == "f" else int(idx_str)

        components.append(
            (
                fmt[component.start() : component.end()],
                idx,
                d.get("sep", None),
            )
        )

    joiners.append(dedup_whitespace(fmt[last_end:]))
    return joiners, components


@cache
def annotations_by_annotation(
    annotation: Union[int, str], subannotations, remote_instance: CatmaidInstance
) -> set[str]:
    g = get_entity_graph(["annotation"], remote_instance=remote_instance)
    aid = None
    if isinstance(annotation, str):
        for n, data in g.nodes(data=True):
            if data["name"] == annotation:
                aid = n
                break
        if aid is None:
            raise ValueError(f"Unknown annotation '{annotation}'")
    else:
        aid = int(annotation)

    if subannotations:
        aids = nx.dfs_preorder_nodes(g, aid)
    else:
        aids = g.successors(aid)

    return {g.nodes[child_aid]["name"] for child_aid in aids}


@cache
def annotations_by_user(
    user: Union[int, str], remote_instance: CatmaidInstance
) -> set[str]:
    if isinstance(user, str):
        key = "name"
    else:
        key = "id"
        user = int(user)

    ann_table = get_annotation_table(remote_instance=remote_instance)
    out = set()

    for aname, _aid, users in ann_table.itertuples(index=False):
        for user_info in users:
            if user_info[key] == user:
                out.add(aname)
    return out


class NeuronLabeller:
    """Class for calculating neurons' labels, as used in the CATMAID frontend."""
    def __init__(
        self,
        components: Optional[List[LabelComponent]] = None,
        fmt="%0",
        trim_empty=True,
        remove_neighboring_duplicates=True,
    ):
        """Create an object which can calculate labels for neurons based on some configuration.

        Parameters
        ----------
        components : List[LabelComponent], optional
            The label components as used in CATMAID's user settings.
            See `SkeletonId`, `NeuronName`, and `Annotations`.
            First component should be ``SkeletonId()`` for compatibility with CATMAID.
            If None (default), uses ``[SkeletonId()]``.
        fmt : str, optional
            Format string as used in CATMAID, by default ``"%0"``.
        trim_empty : bool, optional
            Trim whitespace around components which evaluate to empty strings, by default True
        remove_neighboring_duplicates : bool, optional
            Remove extra consecutive components which evaluate to the same value, by default True
        """
        if components is None:
            components = [SkeletonId()]
        self.components = components
        if not isinstance(self.components[0], SkeletonId):
            logger.warning(
                "First component is not skeleton ID (which is the immutable default in CATMAID). "
                "Format string may not produce the same label as the CATMAID frontend."
            )
        self.fmt = fmt
        self.trim_empty = trim_empty
        self.remove_neighboring_duplicates = remove_neighboring_duplicates

    def label(self, nrn: Union[CatmaidNeuron, ThinNeuron]) -> str:
        """Determine the label for the given neuron.

        Parameters
        ----------
        nrn : Union[CatmaidNeuron, ThinNeuron]
            If a `CatmaidNeuron` object is not available,
            use a `ThinNeuron`, which holds no morphological information,
            can be instantiated from minimal information,
            lazily fills out its own fields as required.

        Returns
        -------
        str
            Neuron label
        """
        joiners, components = parse_components(self.fmt)
        if not joiners:
            return ""
        to_join = []
        prev_component = ""
        lstrip_next = False
        for joiner, (raw, idx, sep) in zip(joiners, components):
            to_join.append(joiner.lstrip() or " " if lstrip_next else joiner)

            if idx is None:
                value = ""
                for comp in reversed(self.components):
                    value = comp.label(nrn, sep)
                    if value:
                        to_join.append(value)
                        break
            elif idx >= len(self.components):
                value = raw
            else:
                comp = self.components[idx]
                value = comp.label(nrn, sep)

            if self.trim_empty and not value:
                to_join.append(to_join.pop().rstrip())
                lstrip_next = True
            else:
                lstrip_next = False

            if self.remove_neighboring_duplicates and value == prev_component:
                to_join.pop()
            else:
                to_join.append(value)

        to_join.append(joiners[-1])
        return "".join(to_join)
