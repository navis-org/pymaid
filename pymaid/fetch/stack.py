from typing import Any, Optional, Union, Literal, Sequence, Tuple, Dict, List
import numpy as np
from ..utils import _eval_remote_instance
from ..client import CatmaidInstance
from enum import IntEnum
from dataclasses import dataclass, asdict

Dimension = Literal["x", "y", "z"]


class Orientation(IntEnum):
    XY = 0
    # todo: check these
    XZ = 1
    ZY = 2

    def __bool__(self) -> bool:
        return True

    def full_orientation(self, reverse=False) -> Tuple[Dimension, Dimension, Dimension]:
        out = [
            ("x", "y", "z"),
            ("x", "z", "y"),
            ("z", "y", "x"),
        ][self.value]
        if reverse:
            out = out[::-1]
        return out

    @classmethod
    def from_dims(cls, dims: Sequence[Dimension]):
        pair = (dims[0].lower(), dims[1].lower())
        out = {
            ("x", "y"): cls.XY,
            ("x", "z"): cls.XZ,
            ("z", "y"): cls.ZY,
        }.get(pair)
        if out is None:
            raise ValueError(f"Unknown dimensions: {dims}")
        return out


@dataclass
class StackSummary:
    id: int
    pid: int
    title: str
    comment: str


def get_stacks(remote_instance: Optional[CatmaidInstance] = None) -> List[StackSummary]:
    """Get summary of all stacks in the project.

    Parameters
    ----------
    remote_instance : Optional[CatmaidInstance], optional
        By default global instance.

    Returns
    -------
    stacks
        List of StackSummary objects.
    """
    cm = _eval_remote_instance(remote_instance)
    url = cm.make_url(cm.project_id, "stacks")
    return [StackSummary(**r) for r in cm.fetch(url)]


@dataclass
class MirrorInfo:
    id: int
    title: str
    image_base: str
    tile_width: int
    tile_height: int
    tile_source_type: int
    file_extension: str
    position: int

    def to_jso(self):
        return asdict(self)


@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float


@dataclass
class StackInfo:
    sid: int
    pid: int
    ptitle: str
    stitle: str
    downsample_factors: Optional[List[Dict[Dimension, float]]]
    num_zoom_levels: int
    translation: Dict[Dimension, float]
    resolution: Dict[Dimension, float]
    dimension: Dict[Dimension, int]
    comment: str
    description: str
    metadata: Optional[str]
    broken_slices: Dict[int, int]
    mirrors: List[MirrorInfo]
    orientation: Orientation
    attribution: str
    canary_location: Dict[Dimension, int]
    placeholder_color: Color

    @classmethod
    def from_jso(cls, sinfo: Dict[str, Any]):
        sinfo["orientation"] = Orientation(sinfo["orientation"])
        sinfo["placeholder_color"] = Color(**sinfo["placeholder_color"])
        sinfo["mirrors"] = [MirrorInfo(**m) for m in sinfo["mirrors"]]
        return StackInfo(**sinfo)

    def to_jso(self):
        return asdict(self)

    def get_downsample(self, scale_level=0) -> Dict[Dimension, float]:
        """Get the downsample factors for a given scale level.

        If the downsample factors are explicit in the stack info,
        use that value.
        Otherwise, use the CATMAID default:
        scale by a factor of 2 per scale level in everything except the slicing dimension.
        If number of scale levels is known,
        ensure the scale level exists.

        Parameters
        ----------
        scale_level : int, optional

        Returns
        -------
        dict[Dimension, float]

        Raises
        ------
        IndexError
            If the scale level is known not to exist
        """
        if self.downsample_factors is not None:
            return self.downsample_factors[scale_level]
        if self.num_zoom_levels > 0 and scale_level >= self.num_zoom_levels:
            raise IndexError("list index out of range")

        first, second, slicing = self.orientation.full_orientation()
        return {first: 2**scale_level, second: 2**scale_level, slicing: 1}

    def get_coords(self, scale_level: int = 0) -> Dict[Dimension, np.ndarray]:
        dims = self.orientation.full_orientation()
        dims = dims[::-1]

        downsamples = self.get_downsample(scale_level)

        out: Dict[Dimension, np.ndarray] = dict()
        for d in dims:
            c = np.arange(self.dimension[d], dtype=float)
            c *= self.resolution[d]
            c *= downsamples[d]
            c += self.translation[d]
            out[d] = c
        return out


def get_stack_info(
    stack: Union[int, str], remote_instance: Optional[CatmaidInstance] = None
) -> StackInfo:
    """Get information about an image stack.

    Parameters
    ----------
    stack : Union[int, str]
        Integer ID or string title of the stack.
    remote_instance : Optional[CatmaidInstance], optional
        By default global.

    Returns
    -------
    StackInfo

    Raises
    ------
    ValueError
        If an unknown stack title is given.
    """
    cm = _eval_remote_instance(remote_instance)
    if isinstance(stack, str):
        stacks = get_stacks(cm)
        for s in stacks:
            if s.title == stack:
                stack_id = s.id
                break
        else:
            raise ValueError(f"No stack with title '{stack}'")
    else:
        stack_id = int(stack)

    url = cm.make_url(cm.project_id, "stack", stack_id, "info")
    sinfo = cm.fetch(url)
    return StackInfo.from_jso(sinfo)


def get_mirror_info(
    stack: Union[int, str, StackInfo],
    mirror: Union[int, str],
    remote_instance: Optional[CatmaidInstance] = None,
) -> MirrorInfo:
    """Get information about a stack mirror.

    Parameters
    ----------
    stack : Union[int, str, StackInfo]
        Integer stack ID, string stack title,
        or an existing StackInfo object (avoids server request).
    mirror : Union[int, str]
        Integer mirror ID, or string mirror title.
    remote_instance : Optional[CatmaidInstance]
        By default, global.

    Returns
    -------
    MirrorInfo

    Raises
    ------
    ValueError
        No mirror matching given ID/ title.
    """
    if isinstance(stack, StackInfo):
        stack_info = stack
    else:
        stack_info = get_stack_info(stack, remote_instance)

    if isinstance(mirror, str):
        key = "title"
    else:
        key = "id"
        mirror = int(mirror)

    for m in stack_info.mirrors:
        if getattr(m, key) == mirror:
            return m

    raise ValueError(
        f"No mirror for stack '{stack_info.stitle}' with {key} {repr(mirror)}"
    )
