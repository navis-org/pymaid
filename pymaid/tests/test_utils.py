from pymaid.fetch import filter_by_query
import pandas as pd
import pytest


NAMES = pd.Series(["potato", "spade", "orange", "bears"])


@pytest.mark.parametrize("query", ["spade", "~spade", "annotation:spade"])
def test_filter_by_query_exact(query):
    out = filter_by_query(NAMES, query)
    assert list(out) == [False, True, False, False]


@pytest.mark.parametrize("query", ["ad", "~ad", "annotation:ad"])
def test_filter_by_query_partial(query):
    out = filter_by_query(NAMES, query, allow_partial=True)
    assert list(out) == [False, True, False, False, False]


@pytest.mark.parametrize("query", ["sp.*e", "~sp.*e", "annotation:sp.*e"])
def test_filter_by_query_re(query):
    out = filter_by_query(NAMES, query)
    assert list(out) == [False, True, False, False]

