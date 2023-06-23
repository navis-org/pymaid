from __future__ import annotations
from io import BytesIO
from typing import Literal, Optional, Sequence, Type, TypeVar, Generic, TypedDict, Union
import numpy as np
from abc import ABC
from numpy.typing import DTypeLike, ArrayLike
import zarr
from pydantic import BaseModel
from pydantic.tools import parse_obj_as
from dask import array as da
from . import utils
from zarr.storage import BaseStore
import json
import sys
import requests
import imageio.v3 as iio

Dimension = Literal["x", "y", "z"]
Orientation = Literal["xy", "xz", "zy"]
HALF_PX = 0.5
ENDIAN = "<" if sys.byteorder == "little" else ">"


class MirrorInfo(BaseModel):
    id: int
    title: str
    image_base: str
    tile_width: int
    tile_height: int
    tile_source_type: int
    file_extension: str
    position: int


N = TypeVar("N", int, float)


class Coord(TypedDict, Generic[N]):
    x: N
    y: N
    z: N


class StackInfo(BaseModel):
    sid: int
    pid: int
    ptitle: str
    stitle: str
    downsample_factors: list[Coord[float]]
    num_zoom_levels: int
    translation: Coord[float]
    resolution: Coord[float]
    dimension: Coord[int]
    comment: str
    description: str
    metadata: str
    broken_slices: dict[int, int]
    mirrors: list[MirrorInfo]
    orientation: Orientation
    attribution: str
    canary_location: Coord[int]
    placeholder_colour: dict[str, float]  # actually {r g b a}


def to_array(
    coord: Union[Coord[N], ArrayLike],
    dtype: DTypeLike = np.float64,
    order: Sequence[Dimension] = ("z", "y", "x"),
) -> np.ndarray:
    if isinstance(coord, dict):
        coord = [coord[d] for d in order]
    return np.asarray(coord, dtype=dtype)


class TileStore(BaseStore, ABC):
    """
    Must include instance variable 'fmt',
    which is a format string with variables:
    image_base, zoom_level, file_extension, row, col, slice_idx
    """
    tile_source_type: int
    fmt: str
    _writeable = False
    _erasable = False
    _listable = False

    def __init__(
        self,
        stack_info: StackInfo,
        mirror_info: MirrorInfo,
        zoom_level: int,
        session: Optional[requests.Session] = None,
    ) -> None:
        if mirror_info.tile_source_type != self.tile_source_type:
            raise ValueError("Mismatched tile source type")
        self.stack_info = stack_info
        self.mirror_info = mirror_info
        self.zoom_level = zoom_level

        if session is None:
            cm = utils._eval_remote_instance(None)
            self.session = cm._session
        else:
            self.session = session

        order = full_orientation[self.stack_info.orientation]
        self.metadata_payload = json.dumps(
            {
                "zarr_format": 2,
                "shape": to_array(stack_info.dimension, order, int).tolist(),
                "chunks": [mirror_info.tile_width, mirror_info.tile_height, 1],
                "dtype": ENDIAN + "u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ).encode()

        self.empty = np.zeros(
            (
                self.mirror_info.tile_width,
                self.mirror_info.tile_height,
                1,
            ),
            "uint8",
        ).tobytes()

    def _format_url(self, row: int, col: int, slice_idx: int) -> str:
        return self.fmt.format(
            zoom_level=self.zoom_level,
            slice_idx=slice_idx,
            row=row,
            col=col,
            file_extension=self.mirror_info.file_extension,
        )

    def __getitem__(self, key):
        last = key.split("/")[-1]
        if last == ".zarray":
            return self.metadata_payload
        # todo: check order
        slice_idx, col, row = (int(i) for i in last.split("."))
        url = self._format_url(row, col, slice_idx)
        response = self.session.get(url)
        if response.status_code == 404:
            return self.empty
        response.raise_for_status()
        arr = iio.imread(
            BytesIO(response.content),
            extension=self.mirror_info.file_extension,
            mode="L",
        )
        return arr.tobytes()

    def to_array(self) -> zarr.Array:
        return zarr.open_array(self, "r")

    def to_dask(self) -> da.Array:
        return da.from_zarr(self.to_array())


class TileStore1(TileStore):
    tile_source_type = 1
    fmt = "{image_base}{slice_idx}/{row}_{col}_{zoom_level}.{file_extension}"


class TileStore4(TileStore):
    tile_source_type = 4
    fmt = "{image_base}{slice_idx}/{zoom_level}/{row}_{col}.{file_extension}"


class TileStore5(TileStore):
    tile_source_type = 5
    fmt = "{image_base}{zoom_level}/{slice_idx}/{row}/{col}.{file_extension}"


tile_stores: dict[int, Type[TileStore]] = {
    t.tile_source_type: t for t in [TileStore1, TileStore4, TileStore5]
}


class Stack:
    def __init__(self, stack_info: StackInfo, mirror_id: Optional[int] = None):
        self.stack_info = stack_info
        self.mirror_info: Optional[MirrorInfo] = None

        if mirror_id is not None:
            self.set_mirror(mirror_id)

    @classmethod
    def from_catmaid(
        cls, stack_id: int, mirror_id: Optional[int] = None, remote_instance=None
    ):
        cm = utils._eval_remote_instance(remote_instance)
        info = cm.make_url("stack", stack_id, "info")
        sinfo = parse_obj_as(StackInfo, info)
        return cls(sinfo, mirror_id)

    def _get_mirror_info(self, mirror_id: Optional[int] = None) -> MirrorInfo:
        if mirror_id is None:
            if self.mirror_info is None:
                raise ValueError("No default mirror ID set")
            return self.mirror_info
        for mirror in self.stack_info.mirrors:
            if mirror.id == mirror_id:
                return mirror
        raise ValueError(
            f"Mirror ID {mirror_id} not found for stack {self.stack_info.sid}"
        )

    def set_mirror(self, mirror_id: int):
        self.mirror_id = self._get_mirror_info(mirror_id)

    def _res_for_scale(self, scale_level: int) -> np.ndarray:
        return to_array(self.stack_info.resolution) * to_array(
            self.stack_info.downsample_factors[scale_level]
        )

    def _from_array(self, arr, scale_level: int) -> ImageVolume:
        return ImageVolume(
            arr,
            self.stack_info.translation,
            self._res_for_scale(scale_level),
            self.stack_info.orientation,
        )

    def get_scale(
        self, scale_level: int, mirror_id: Optional[int] = None
    ) -> ImageVolume:
        mirror_info = self._get_mirror_info(mirror_id)
        if scale_level > self.stack_info.num_zoom_levels:
            raise ValueError(
                f"Scale level {scale_level} does not exist "
                f"for stack {self.stack_info.sid} "
                f"with {self.stack_info.num_zoom_levels} stack levels"
            )

        if mirror_info.tile_source_type in tile_stores:
            store_class = tile_stores[mirror_info.tile_source_type]
            store = store_class(self.stack_info, mirror_info, scale_level, None)
            return self._from_array(store.to_dask(), scale_level)
        elif mirror_info.tile_source_type == 11:
            formatted = mirror_info.image_base.replace(
                "%SCALE_DATASET%", f"s{scale_level}"
            )
            *components, transpose_str = formatted.split("/")
            transpose = [int(t) for t in transpose_str.split("_")]

            store = zarr.N5FSStore("/".join(components))
            arr = zarr.open_array(store, "r")
            darr = da.from_zarr(arr).transpose(transpose)
            return self._from_array(darr, scale_level)

        raise NotImplementedError(
            f"Tile source type {mirror_info.tile_source_type} not implemented"
        )


full_orientation: dict[Orientation, Sequence[Dimension]] = {
    "xy": "xyz",
    "xz": "xzy",
    "zy": "zyx",
}


class ImageVolume:
    def __init__(self, array, offset, resolution, orientation: Orientation):
        self.array = array
        self.offset = offset
        self.resolution = resolution
        self.offset = to_array(offset, dtype="float64")
        self.resolution = to_array(resolution, dtype="float64")
        self.orientation = orientation

    @property
    def full_orientation(self):
        return full_orientation[self.orientation]

    @property
    def offset_oriented(self):
        return to_array(self.offset, "float64", self.full_orientation)

    @property
    def resolution_oriented(self):
        return to_array(self.resolution, "float64", self.full_orientation)

    def __getitem__(self, selection):
        return self.array.__getitem__(selection)

    def get_roi(
        self, offset: Coord[float], shape: Coord[float]
    ) -> tuple[Coord[float], np.ndarray]:
        order = self.full_orientation
        offset_o = to_array(offset, order=order)
        shape_o = to_array(shape, order=order)
        mins = (offset_o / self.resolution - self.offset - HALF_PX).astype("uint64")
        maxes = np.ceil(
            (offset_o + shape_o) / self.resolution - self.offset - HALF_PX
        ).astype("uint64")
        slicing = tuple(slice(mi, ma) for mi, ma in zip(mins, maxes))
        # todo: finalise orientation
        actual_offset = Coord(
            **{
                d: m
                for d, m in zip(
                    order, mins * self.resolution_oriented + self.offset_oriented
                )
            }
        )
        return actual_offset, self[slicing]
