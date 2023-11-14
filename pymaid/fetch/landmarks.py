import typing as tp
from functools import cache

import numpy as np
import pandas as pd

from pymaid.client import CatmaidInstance

from ..utils import _eval_remote_instance, DataFrameBuilder, clean_points


def get_landmarks(
    with_locations=True, remote_instance=None
) -> tp.Tuple[
    pd.DataFrame,
    tp.Optional[pd.DataFrame]
]:
    """Get all landmarks from CATMAID, optionall with locations associated with them.

    Parameters
    ----------
    with_locations : bool, optional
        Whether to also return a location table,
        by default True
    remote_instance : pymaid.CatmaidInstance, optional

    Returns
    -------
    landmarks : pd.DataFrame
        DataFrame with columns
        landmark_id, name, user_id, project_id, creation_time, edition_time.
    locations : pd.DataFrame, optional (default None)
        DataFrame with columns
        location_id, x, y, z, landmark_id.

    Examples
    --------
    >>> # Join the two tables to find all landmark locations
    >>> landmarks, locations = pymaid.get_landmarks(True)
    >>> combined = landmarks.merge(locations, on="landmark_id")
    """
    cm = _eval_remote_instance(remote_instance)
    url = cm.make_url(
        cm.project_id,
        "landmarks",
        # todo: don't download locations if we're not going to use them?
        with_locations="true",
    )
    landmark_builder = DataFrameBuilder(
        ["landmark_id", "name", "user_id", "project_id", "creation_time", "edition_time"],
        ["uint64", "str", "uint64", "uint64", "datetime64[ns]", "datetime64[ns]"]
    )
    location_builder = DataFrameBuilder(
        ["location_id", "x", "y", "z", "landmark_id"],
        ["uint64", "float64", "float64", "float64", "uint64"]
    )

    for landmark in cm.fetch(url):
        landmark_builder.append_row([
            landmark["id"],
            landmark["name"],
            landmark["user"],
            landmark["project"],
            landmark["creation_time"],
            landmark["edition_time"]
        ])

        if not with_locations:
            continue

        for location in landmark["locations"]:
            location_builder.append_row(
                [
                    location["id"],
                    location["x"],
                    location["y"],
                    location["z"],
                    landmark["id"],
                ]
            )

    landmarks = landmark_builder.build()

    if with_locations:
        return landmarks, location_builder.build()
    else:
        return landmarks, None


def get_landmark_groups(
    with_locations=False, with_members=False, remote_instance=None
) -> tp.Tuple[
    pd.DataFrame,
    tp.Optional[pd.DataFrame],
    tp.Optional[tp.Dict[int, tp.List[int]]]
]:
    """Get the landmark groups, optionally with IDs of their members and locations.

    Parameters
    ----------
    with_locations : bool, optional
        Return a DataFrame of locations associated with group members,
        by default False
    with_members : bool, optional
        Return a dict of group IDs to landmark IDs,
        by default False
    remote_instance : pymaid.CatmaidInstance, optional

    Returns
    -------
    groups : pd.DataFrame
        A DataFrame with columns
        group_id, name, user_id, project_id, creation_time, edition_time.
    locations : pd.DataFrame, optional (default None)
        A DataFrame with columns
        location_id, x, y, z, group_id.
    members : dict[int, list[int]], optional (default None)
        A dict mapping group ID to a list of landmark IDs (members of that group).

    Examples
    --------
    >>> # Join the group and location tables
    >>> groups, locations, _ = pymaid.get_landmark_groups(True, False)
    >>> combined = groups.merge(locations, on="group_id")
    """
    cm = _eval_remote_instance(remote_instance)
    url = cm.make_url(
        cm.project_id,
        "landmarks",
        "groups",
        with_locations=str(with_locations).lower(),
        with_members=str(with_members).lower(),
    )

    group_builder = DataFrameBuilder(
        ["group_id", "name", "user_id", "project_id", "creation_time", "edition_time"],
        ["uint64", "str", "uint64", "uint64", "datetime64[ns]", "datetime64[ns]"]
    )
    location_builder = DataFrameBuilder(
        ["location_id", "x", "y", "z", "group_id"],
        ["uint64", "float64", "float64", "float64", "uint64"]
    )
    members = None
    if with_members:
        members = dict()

    for group in cm.fetch(url):
        group_builder.append_row([
            group["id"],
            group["name"],
            group["user"],
            group["project"],
            group["creation_time"],
            group["edition_time"]
        ])

        if members is not None:
            members[group["id"]] = group["members"]

        if not with_locations:
            continue

        for location in group["locations"]:
            location_builder.append_row(
                [
                    location["id"],
                    location["x"],
                    location["y"],
                    location["z"],
                    group["id"],
                ]
            )

    groups = group_builder.build()

    if with_locations:
        locations = location_builder.build()
    else:
        locations = None

    return groups, locations, members


class LandmarkMatcher:
    """Class for finding matching pairs of landmark locations between two groups.

    For example, find control points for transforming neurons left to right
    or between segments.
    """
    def __init__(
        self,
        landmarks: pd.DataFrame,
        landmark_locations: pd.DataFrame,
        groups: pd.DataFrame,
        group_locations: pd.DataFrame,
        group_members: dict[int, tp.Iterable[int]],
    ):
        """Prefer constructing with ``.from_catmaid()`` where possible.

        Parameters
        ----------
        landmarks : pd.DataFrame
            Landmarks dataframe:
            see first output of ``get_landmarks`` for details.
        landmark_locations : pd.DataFrame
            Landmark locations dataframe:
            see second (optional) output of ``get_landmarks``.
        groups : pd.DataFrame
            Groups dataframe:
            see first output of ``get_landmark_groups`` for details.
        group_locations : pd.DataFrame
            Group locations dataframe:
            see second (optional) output of ``get_landmark_groups`` for details.
        group_members : dict[int, tp.Iterable[int]]
            Group members:
            see third (optional) output of ``get_landmark_groups`` for details.
        """
        self.landmarks = landmarks
        self.landmark_locations = landmark_locations
        self.groups = groups
        self.group_locations = group_locations
        self.group_members: dict[int, set[int]] = {
            k: set(v) for k, v in group_members.items()
        }

    @classmethod
    def from_catmaid(cls, remote_instance=None):
        """Instantiate from a CatmaidInstance.

        Possibly the global one.

        Parameters
        ----------
        remote_instance : CatmaidInstance, optional
            If None (default) use the global instance.
        """
        cm = _eval_remote_instance(remote_instance)
        landmarks, landmark_locations = get_landmarks(True, cm)
        groups, group_locations, members = get_landmark_groups(True, True, remote_instance=cm)
        return cls(landmarks, landmark_locations, groups, group_locations, members)

    @cache
    def _group_name_to_id(self) -> dict[str, int]:
        return dict(zip(self.groups["name"], self.groups["group_id"]))

    def _locations_in_group(self, group_id: int):
        idx = self.group_locations["group_id"] == group_id
        return self.group_locations["location_id"][idx]

    def _locations_in_landmark(self, landmark_id: int):
        idx = self.landmark_locations["landmark_id"] == landmark_id
        return self.landmark_locations["location_id"][idx]

    def _unique_location(self, group_id: int, landmark_id: int) -> int:
        if landmark_id not in self.group_members[group_id]:
            return None

        landmark_ids = set(self._locations_in_group(group_id)).intersection(self._locations_in_landmark(landmark_id))
        if len(landmark_ids) == 1:
            return landmark_ids.pop()
        return None

    @cache
    def _landmark_id_to_name(self) -> dict[int, str]:
        return dict(zip(self.landmarks["landmark_id"], self.landmarks["name"]))

    @cache
    def _locations(self) -> dict[int, tuple[int, int, int]]:
        d = dict()
        for loc_id, x, y, z, _ in self.group_locations.itertuples(index=False):
            d[loc_id] = (x, y, z)

        for loc_id, x, y, z, _ in self.landmark_locations.itertuples(index=False):
            d[loc_id] = (x, y, z)

        return d

    def _as_group_id(self, group: tp.Union[str, int]) -> int:
        if isinstance(group, int):
            return group
        if isinstance(group, str):
            return self._group_name_to_id()[group]
        raise ValueError("Given group must be int ID or str name")

    def match(self, group1: tp.Union[str, int], group2: tp.Union[str, int]) -> pd.DataFrame:
        """Get matching pairs of landmarks for two groups.

        Return the paired locations of landmarks
        which are a members of both groups,
        and have a single location in each group.

        Parameters
        ----------
        group1 : tp.Union[str, int]
            First group name (as str) or ID (as int)
        group2 : tp.Union[str, int]
            Second group name (as str) or ID (as int)

        Returns
        -------
        pd.DataFrame
            Columns landmark_name, landmark_id, location_id1, x1, y1, z1, location_id2, x2, y2, z2
        """
        group_to_id = self._group_name_to_id()
        group1_id = group1 if isinstance(group1, int) else group_to_id[group1]
        group2_id = group2 if isinstance(group2, int) else group_to_id[group2]

        group1_locs = set(self._locations_in_group(group1_id))
        group2_locs = set(self._locations_in_group(group2_id))

        locs = self._locations()

        dtypes = {"landmark_name": str, "landmark_id": np.uint64}
        for g in ["1", "2"]:
            dtypes["location_id" + g] = np.uint64
            for d in "xyz":
                dtypes[d + g] = np.float64

        lm_id_to_name = self._landmark_id_to_name()

        rows = []
        for lm_id in set(self.group_members[group1_id]).intersection(self.group_members[group2_id]):
            lm_locs = self._locations_in_landmark(lm_id)
            g1_locs = group1_locs.intersection(lm_locs)
            if not len(g1_locs) == 1:
                continue
            g2_locs = group2_locs.intersection(lm_locs)
            if not len(g1_locs) == 1:
                continue

            row = [lm_id_to_name[lm_id], lm_id]
            row.append(g1_locs.pop())
            row.extend(locs[row[-1]])
            row.append(g2_locs.pop())
            row.extend(locs[row[-1]])

            rows.append(row)

        df = pd.DataFrame(rows, columns=list(dtypes), dtype=object)
        return df.astype(dtypes)


class CrossProjectLandmarkMatcher:
    """Class for finding matching pairs of landmark locations between two instances.

    For example, find control points for transforming neurons
    from one instance's space to another.
    """
    def __init__(self, this_lms: LandmarkMatcher, other_lms: LandmarkMatcher):
        """Constructing using ``.from_catmaid()`` may be more convenient.

        Parameters
        ----------
        this_lms : LandmarkMatcher
            ``LandmarkMatcher`` for landmarks in "this" space.
        other_lms : LandmarkMatcher
            ``LandmarkMatcher`` for landmarks in the "other" space.
        """
        self.this_m: LandmarkMatcher = this_lms
        self.other_m: LandmarkMatcher = other_lms

    @classmethod
    def from_catmaid(
        cls, other_remote_instance: CatmaidInstance, this_remote_instance=None
    ):
        """Instantiate from a pair of CatmaidInstances.

        Parameters
        ----------
        other_remote_instance : CatmaidInstance
            Other catmaid instance
        this_remote_instance : CatmaidInstance, optional
            This CATMAID instance.
            If None (default), use the global instance.
        """
        this_remote_instance = _eval_remote_instance(this_remote_instance)
        return cls(
            LandmarkMatcher.from_catmaid(this_remote_instance),
            LandmarkMatcher.from_catmaid(other_remote_instance),
        )

    def _member_landmarks(self, group_id: int, this=True) -> dict[str, int]:
        matcher = self.this_m if this else self.other_m
        lmid_to_name = matcher._landmark_id_to_name()
        out = dict()
        for lmid in matcher.group_members[group_id]:
            out[lmid_to_name[lmid]] = lmid
        return out

    def match(
        self,
        this_group: tp.Union[str, int],
        other_group: tp.Optional[tp.Union[str, int]] = None
    ) -> pd.DataFrame:
        """Match landmark locations between two instance of CATMAID.

        Looks through the members of the two groups to find landmarks with the same name.
        If those landmarks have one location in each group, match those.

        Parameters
        ----------
        this_group : tp.Union[str, int]
            Group name (str) or ID (int) on this instance.
        other_group : tp.Optional[tp.Union[str, int]], optional
            Group name (str) or ID (int) on the other instance.
            If None (default) and ``this_group`` is a str name, use the same name.

        Returns
        -------
        pd.DataFrame
            Columns landmark_name, x1, y1, z1, x2, y2, z2
            where 1 is "this" and 2 is "other".
        """
        this_group_id = self.this_m._as_group_id(this_group)
        if other_group is None:
            if isinstance(this_group, int):
                raise ValueError("If other_group is None, this_group must be a str name")
            other_group = this_group
        other_group_id = self.other_m._as_group_id(other_group)

        this_locs = self.this_m._locations()
        other_locs = self.other_m._locations()

        this_lms = self._member_landmarks(this_group_id)
        other_lms = self._member_landmarks(other_group_id, False)

        dtypes = {"landmark_name": str}
        for g in "12":
            for d in "xyz":
                dtypes[d + g] = np.float64

        rows = []

        for lm_name, this_lmid in this_lms.items():
            if lm_name not in other_lms:
                continue
            other_lmid = other_lms[lm_name]

            this_loc_id = self.this_m._unique_location(this_group_id, this_lmid)
            if this_loc_id is None:
                continue

            other_loc_id = self.other_m._unique_location(other_group_id, other_lmid)
            if other_loc_id is None:
                continue

            row = [lm_name]
            row.extend(this_locs[this_loc_id])
            row.extend(other_locs[other_loc_id])

            rows.append(row)

        df = pd.DataFrame(rows, columns=list(dtypes), dtype=object)
        return df.astype(dtypes)

    def match_all(self) -> pd.DataFrame:
        """Match all landmark locations between two instances of CATMAID.

        Looks for all groups which share a name,
        then all landmark members of those groups which have a single location in each group.

        Returns
        -------
        pd.DataFrame
            Columns group_name, landmark_name, x1, y1, z1, x2, y2, z2
        """
        shared_groups = set(
            self.this_m.groups["name"]
        ).intersection(self.other_m.groups["name"])
        dfs = []
        for group_name in sorted(shared_groups):
            df = self.match(group_name)
            df.insert(0, "group_name", [group_name] * len(df))
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)


def to_control_points(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Get control point arrays from a dataframe returned by a LandmarkMatcher.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by a LandmarkMatcher or CrossProjectLandmarkMatcher.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The coordinates of locations.
    """
    return clean_points(df, "{}1").to_numpy(), clean_points(df, "{}2").to_numpy()
