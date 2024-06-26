import os
import pytest
import navis
from contextlib import contextmanager
import logging
import pymaid

logger = logging.getLogger(__name__)


@contextmanager
def igraph_context(use_igraph):
    orig = navis.config.use_igraph
    logger.debug(f"Setting navis.config.use_igraph = {use_igraph}")
    navis.config.use_igraph = use_igraph
    yield use_igraph
    logger.debug(f"Resetting navis.config.use_igraph = {orig}")
    navis.config.use_igraph = orig


@pytest.fixture(scope="session", params=[True, False])
def use_igraph(request):
    with igraph_context(request.param) as use:
        yield use


@pytest.fixture(scope="module")
def client():
    return pymaid.CatmaidInstance(
        os.environ.get("PYMAID_TEST_SERVER_URL", 'https://fafb.catmaid.virtualflybrain.org/'),
        os.environ.get("PYMAID_TEST_TOKEN"),
        os.environ.get("PYMAID_TEST_HTTP_USER"),
        os.environ.get("PYMAID_TEST_HTTP_PW"),
        make_global=True,
    )


@pytest.fixture(scope="module")
def skids():
    return [
        int(s) for s in os.environ.get(
            "PYMAID_TEST_SKIDS",
            "16,1299740,4744251"
        ).split(",")
    ]


@pytest.fixture(scope="module")
def annotation_names():
    return os.environ.get(
        "PYMAID_TEST_ANNOTATIONS",
        'Paper: Dolan and Belliart-Guérin et al. 2018,Paper: Wang et al 2020a'
    ).split(",")


@pytest.fixture(scope="module")
def volume_name():
    return os.environ.get("PYMAID_TEST_VOLUME", "LH_R")


@pytest.fixture(scope="module")
def stack_id():
    return int(os.environ.get("PYMAID_TEST_STACK_ID", 1))
