"""Access to image data as xarray.DataArrays.

CATMAID's image source conventions are documented here
https://catmaid.readthedocs.io/en/stable/tile_sources.html
"""
from __future__ import annotations
from io import BytesIO
from typing import Any, Callable, Literal, Optional, Sequence, Type, Union, Dict
from abc import ABC
import sys

import numpy as np
from numpy.typing import DTypeLike, ArrayLike
import zarr
from dask import array as da
import xarray as xr
from zarr.storage import BaseStore
import json
import requests
import imageio.v3 as iio

from pymaid.client import CatmaidInstance

from . import utils
from .fetch.stack import (
    StackInfo,
    MirrorInfo,
    get_stacks,
    get_stack_info,
    get_mirror_info,
)

Dimension = Literal["x", "y", "z"]
# Orientation = Literal["xy", "xz", "zy"]
HALF_PX = 0.5
ENDIAN = "<" if sys.byteorder == "little" else ">"


def select_cli(prompt: str, options: Dict[int, str]) -> Optional[int]:
    out = None
    print(prompt)
    for k, v in sorted(options.items()):
        print(f"\t{k}.\t{v}")
    p = "Type number and press enter (empty to cancel): "
    while out is None:
        result_str = input(p).strip()
        if not result_str:
            break
        try:
            result = int(result_str)
        except ValueError:
            print("Not an integer, try again")
            continue
        if result not in options:
            print("Not a valid option, try again")
            continue
        out = result
    return out


def to_array(
    coord: Union[Dict[Dimension, Any], ArrayLike],
    dtype: DTypeLike = np.float64,
    order: Sequence[Dimension] = ("z", "y", "x"),
) -> np.ndarray:
    if isinstance(coord, dict):
        coord = [coord[d] for d in order]
    return np.asarray(coord, dtype=dtype)


class ImageIoStore(BaseStore, ABC):
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
        session: Union[requests.Session, CatmaidInstance, None] = None,
    ) -> None:
        if mirror_info.tile_source_type != self.tile_source_type:
            raise ValueError("Mismatched tile source type")
        self.stack_info = stack_info
        self.mirror_info = mirror_info
        self.zoom_level = zoom_level

        if isinstance(session, CatmaidInstance):
            self.session = session._session
        elif isinstance(session, requests.Session):
            self.session = session
        elif session is None:
            session = requests.Session()

        brok_sl = {int(k): int(k) + v for k, v in self.stack_info.broken_slices.items()}
        self.broken_slices = dict()
        for k, v in brok_sl.items():
            while v in brok_sl:
                v = brok_sl[v]
            self.broken_slices[k] = v

        order = self.stack_info.orientation.full_orientation(reverse=True)
        self.metadata_bytes = json.dumps(
            {
                "zarr_format": 2,
                "shape": to_array(stack_info.dimension, int, order).tolist(),
                "chunks": [1, mirror_info.tile_height, mirror_info.tile_width],
                "dtype": ENDIAN + "u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ).encode()
        self.attrs_bytes = json.dumps(
            {
                "stack_info": self.stack_info.to_jso(),
                "mirror_info": self.mirror_info.to_jso(),
                "scale_level": self.zoom_level,
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
            image_base=self.mirror_info.image_base,
            zoom_level=self.zoom_level,
            slice_idx=slice_idx,
            row=row,
            col=col,
            file_extension=self.mirror_info.file_extension,
        )

    def __delitem__(self, __key) -> None:
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __setitem__(self, __key, __value) -> None:
        raise NotImplementedError()

    def _resolve_broken_slices(self, slice_idx: int) -> int:
        return self.broken_slices.get(slice_idx, slice_idx)

    def __getitem__(self, key):
        last = key.split("/")[-1]
        if last == ".zarray":
            return self.metadata_bytes
        elif last == ".zattrs":
            return self.attrs_bytes

        # todo: check order
        slice_idx, row, col = (int(i) for i in last.split("."))
        slice_idx = self._resolve_broken_slices(slice_idx)

        url = self._format_url(row, col, slice_idx)
        response = self.session.get(url)
        if response.status_code == 404:
            return self.empty
        response.raise_for_status()
        ext = self.mirror_info.file_extension.split("?")[0]
        if not ext.startswith("."):
            ext = "." + ext
        arr = iio.imread(
            BytesIO(response.content),
            extension=ext,
            mode="L",
        )
        return arr.tobytes()

    def to_zarr_array(self) -> zarr.Array:
        return zarr.open_array(self, "r")

    def to_dask_array(self) -> xr.DataArray:
        # todo: transpose?
        as_zarr = self.to_zarr_array()
        return da.from_zarr(as_zarr)

    def to_xarray(self) -> xr.DataArray:
        as_dask = self.to_dask_array()
        return xr.DataArray(
            as_dask,
            coords=self.stack_info.get_coords(self.zoom_level),
            dims=self.stack_info.orientation.full_orientation(True),
        )


class TileStore1(ImageIoStore):
    """File-based image stack."""
    tile_source_type = 1
    fmt = "{image_base}{slice_idx}/{row}_{col}_{zoom_level}.{file_extension}"


class TileStore4(ImageIoStore):
    """File-based image stack with zoom level directories."""
    tile_source_type = 4
    fmt = "{image_base}{slice_idx}/{zoom_level}/{row}_{col}.{file_extension}"


class TileStore5(ImageIoStore):
    """Directory-based image stack with zoom, z, and row directories."""
    tile_source_type = 5
    fmt = "{image_base}{zoom_level}/{slice_idx}/{row}/{col}.{file_extension}"


# class TileStore10(ImageIoStore):
#     """H2N5 tile stack."""
#     tile_source_type = 10
#     fmt = "{image_base}.{file_extension}"

#     # todo: manually change quality?

#     def _format_url(self, row: int, col: int, slice_idx: int) -> str:
#         s = self.fmt.format(
#             image_base=self.mirror_info.image_base,
#             file_extension=self.mirror_info.file_extension,
#         )
#         s = s.replace("%SCALE_DATASET%", f"s{self.zoom_level}")
#         s = s.replace("%AXIS_0%", str(col * self.mirror_info.tile_width))
#         s = s.replace("%AXIS_1%", str(row * self.mirror_info.tile_height))
#         s = s.replace("%AXIS_2%", str(slice_idx))
#         return s


tile_stores: Dict[int, Type[ImageIoStore]] = {
    t.tile_source_type: t
    for t in [
        TileStore1,
        TileStore4,
        TileStore5,
        # TileStore10
    ]
}
supported_sources = {11}.union(tile_stores)


def select_stack(remote_instance=None) -> Optional[int]:
    """"""
    stacks = get_stacks(remote_instance)
    options = {s.id: s.title for s in stacks}
    return select_cli("Select stack:", options)


class Stack:
    """Class representing a CATMAID stack of images.

    Stacks are usually a scale pyramid.
    This class can, for certain stack mirror types,
    allow access to individual scale levels as arrays
    which can be queried in voxel or world coordinates.
    """
    def __init__(
        self,
        stack_info: StackInfo,
        mirror: Optional[Union[int, str]] = None,
    ):
        """The :func:`Stack.from_catmaid` constructor may be more convenient.

        Parameters
        ----------
        stack_info : StackInfo
        mirror_id : Optional[int], optional
        """
        self.stack_info = stack_info
        self.mirror_info: Optional[MirrorInfo] = None
        self.mirror_session_factory: Dict[int, Callable[[], Any]] = dict()

        if mirror is not None:
            self.set_mirror(mirror)

    def set_mirror_session_factory(
        self, mirror: Union[int, str, None], factory: Callable[[], Any]
    ):
        """Set functions which construct the session for fetching image data, per mirror.

        For most tile stacks, this is a
        `requests.Session <https://requests.readthedocs.io/en/latest/api/#requests.Session>`_.
        See ``get_remote_instance_session`` to use the session from a given
        ``CatmaidInstance`` (including the global).

        For N5 (tile source 11), this is a
        `aiohttp.ClientSession <https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientSession>`_.

        Parameters
        ----------
        mirror : Union[int, str, None]
            Mirror, as integer ID, string name, or None to use the one defined on the class.
        factory : Callable[[], Any]
            Function which creates a session of the appropriate type.
            For example, to re-use the ``requests.Session`` from the
            global ``CatmaidInstance`` for mirror with ID 1, use
            ``my_stack.set_mirror_instance_factory(1, get_remote_instance_session)``.
            To use HTTP basic auth for an N5 stack mirror (tile source 11) with ID 2, use
            ``my_stack.set_mirror_instance_factor(2, lambda: aiohttp.ClientSession(auth=aiohttp.BasicAuth("myusername", "mypassword")))``.
        """
        minfo = self._get_mirror_info(mirror)
        self.mirror_session_factory[minfo.id] = factory

    @classmethod
    def select_from_catmaid(cls, remote_instance=None):
        """Interactively select a stack and mirror from those available.

        Parameters
        ----------
        remote_instance : CatmaidInstance, optional
            By default global.
        """
        stacks = get_stacks(remote_instance)
        options = {s.id: s.title for s in stacks}
        sid = select_cli("Select stack:", options)
        if not sid:
            return None
        out = cls.from_catmaid(sid, remote_instance=remote_instance)
        out.select_mirror()
        return out

    @classmethod
    def from_catmaid(
        cls, stack: Union[str, int], mirror: Optional[Union[int, str]] = None, remote_instance=None
    ):
        """Fetch relevant data from CATMAID and build a Stack.

        Parameters
        ----------
        stack : Union[str, int]
            Integer stack ID or string stack title.
        mirror : Optional[int, str], optional
            Integer mirror ID or string mirror title, by default None
        remote_instance : CatmaidInstance, optional
            By default global.
        """
        sinfo = get_stack_info(stack, remote_instance)
        return cls(sinfo, mirror)

    def _get_mirror_info(self, mirror: Optional[Union[int, str]] = None) -> MirrorInfo:
        if mirror is None:
            if self.mirror_info is None:
                raise ValueError("No default mirror ID set")
            return self.mirror_info

        return get_mirror_info(self.stack_info, mirror)

    def set_mirror(self, mirror: Union[int, str]):
        """Set the mirror using its int ID or str title."""
        self.mirror_info = self._get_mirror_info(mirror)

    def select_mirror(self):
        """Interactively select a mirror from those available.
        """
        options = {
            m.id: m.title
            for m in self.stack_info.mirrors
            if m.tile_source_type in supported_sources
        }
        if not options:
            print("No mirrors with supported tile source type")
            return

        result = select_cli(
            f"Select mirror for stack '{self.stack_info.stitle}':",
            options,
        )
        if result is not None:
            self.set_mirror(result)

    def _get_session_factory(self, mirror_id: int, default: Optional[Callable[[], Any]]=None):
        try:
            return self.mirror_session_factory[mirror_id]
        except KeyError:
            if default is None:
                raise
            else:
                return default

    def get_scale(
        self, scale_level: int, mirror_id: Optional[int] = None
    ) -> xr.DataArray:
        """Get an xarray.DataArray representing th given scale level.

        Note that depending on the metadata available,
        missing scale levels may throw different errors.

        Parameters
        ----------
        scale_level : int
            0 for full resolution
        mirror_id : Optional[int], optional
            By default the one set on the class.

        Returns
        -------
        xr.DataArray
            Can be queried in voxel or world space.

        Raises
        ------
        ValueError
            Scale level does not exist, according to metadata
        NotImplementedError
            Unknown tile source type for this mirror
        """
        mirror_info = self._get_mirror_info(mirror_id)
        if (
            self.stack_info.num_zoom_levels > 0
            and scale_level > self.stack_info.num_zoom_levels
        ):
            raise ValueError(
                f"Scale level {scale_level} does not exist "
                f"for stack {self.stack_info.sid} "
                f"with {self.stack_info.num_zoom_levels} stack levels"
            )

        if mirror_info.tile_source_type in tile_stores:
            store_class = tile_stores[mirror_info.tile_source_type]
            fac = self._get_session_factory(
                mirror_info.id,
                requests.Session,
            )
            store = store_class(
                self.stack_info, mirror_info, scale_level, fac()
            )
            return store.to_xarray()
        elif mirror_info.tile_source_type == 11:
            return self._get_n5(mirror_info, scale_level)

        raise NotImplementedError(
            f"Tile source type {mirror_info.tile_source_type} not implemented"
        )

    def _get_n5(
        self,
        mirror_info: MirrorInfo,
        scale_level: int,
    ) -> xr.DataArray:
        if mirror_info.tile_source_type != 11:
            raise ValueError("Mirror info not from an N5 tile source")
        formatted = mirror_info.image_base.replace(
            "%SCALE_DATASET%", f"s{scale_level}"
        )
        *components, transpose_str = formatted.split("/")
        transpose = [int(t) for t in transpose_str.split("_")]

        container_comp = []
        arr_comp = []
        this = container_comp
        for comp in components:
            this.append(comp)
            if comp.lower().endswith(".n5"):
                this = arr_comp

        if not arr_comp:
            raise ValueError("N5 container must have '.n5' suffix")

        kwargs = dict()
        fac = self._get_session_factory(mirror_info.id, None)
        if fac is not None:
            kwargs["get_client"] = fac

        store = zarr.N5FSStore("/".join(container_comp), **kwargs)

        container = zarr.open(store, "r")
        as_zarr = container["/".join(arr_comp)]
        # todo: check this transpose
        as_dask = da.from_zarr(as_zarr).transpose(transpose)
        return xr.DataArray(
            as_dask,
            coords=self.stack_info.get_coords(scale_level),
            dims=self.stack_info.orientation.full_orientation(True),
        )


def get_remote_instance_session(remote_instance: Optional[CatmaidInstance] = None):
    """Get the ``requests.Session`` from the given ``CatmaidInstance``.

    If ``None`` is given, use the global ``CatmaidInstance``.
    """
    cm = utils._eval_remote_instance(remote_instance)
    return cm._session
