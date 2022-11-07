import typing as tp

import pandas as pd

from ..utils import _eval_remote_instance, DataFrameBuilder


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
    2-tuple of (DataFrame, optional DataFrame)
        The first element is a DataFrame with columns
        landmark_id, name, user_id, project_id, creation_time, edition_time.

        The second element is optionally a DataFrame with columns
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
    3-tuple of (dataframe, optional dataframe, optional dict[int, int])
        The first element is a DataFrame with columns group_id, name, user_id, project_id, creation_time, edition_time.

        The second element is optionally a DataFrame with columns location_id, x, y, z, group_id.

        The third element is optionally a dict mapping group ID to a list of landmark IDs (members of that group).

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
