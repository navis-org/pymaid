import typing as tp
import datetime as dt

from ..utils import _eval_remote_instance
from ..client import CatmaidInstance
from ..config import get_logger

logger = get_logger(__name__)


def get_skeleton_ids(
    created_by: tp.Optional[int]=None,
    reviewed_by: tp.Optional[int]=None,
    from_date: tp.Optional[dt.date]=None,
    to_date: tp.Optional[dt.date]=None,
    nodecount_gt: tp.Optional[int]=None,
    remote_instance: tp.Optional[CatmaidInstance]=None,
) -> tp.Set[int]:
    """Get all skeleton IDs from the project matching given filters.

    Parameters
    ----------
    created_by : int, optional
        Created by user with this ID.
    reviewed_by : int, optional
        Reviewed by user with this ID.
    from_date : datetime.date, optional
        Has nodes created after this date.
    to_date : datetime.date, optional
        Has nodes created before this date.
    nodecount_gt : int, optional
        Has greater than this number of nodes.
        This will remove all other criteria.
    remote_instance : pymaid.CatmaidInstance, optional
        Catmaid instance, by default None

    Returns
    -------
    set[int]
        Skeleton IDs matching the given criteria.
    """
    cm = _eval_remote_instance(remote_instance)
    params = dict()
    if created_by is not None:
        params["created_by"] = int(created_by)
    if reviewed_by is not None:
        params["reviewed_by"] = int(reviewed_by)
    if from_date is not None:
        params["from_date"] = from_date.strftime("%Y%m%d")
    if to_date is not None:
        params["to_date"] = to_date.strftime("%Y%m%d")
    if nodecount_gt is not None:
        if params:
            logger.warning("Including `nodecount_gt` in `get_skeleton_ids` will remove all other criteria: %s were given", sorted(params))
        params["nodecount_gt"] = int(nodecount_gt)

    url = cm.make_url(
        cm.project_id,
        "skeletons",
        **params,
    )

    result = cm.fetch(url)
    return set(result)
