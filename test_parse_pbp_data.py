import pandas as pd
import numpy as np

from parse_pbp_data import parse_cdc, parse_pmen, build_co_occurrence_graph

cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
pbp_patterns = ["a1", "b2", "x2"]


def check_data(data):
    for pbp in pbp_patterns:
        assert (
            all(
                data.groupby(data[f"{pbp}_type"]).apply(
                    lambda df: df[f"{pbp}_seq"].nunique() == 1
                )
            )
            is True
        )  # one sequence per type
        assert (
            all(
                data.groupby(data[f"{pbp}_seq"]).apply(
                    lambda df: df[f"{pbp}_type"].nunique() == 1
                )
            )
            is True
        )  # one type per sequence


def check_graph(adj, deg):
    adj = adj.todense()
    deg = deg.todense()
    assert np.array_equal(adj, adj.T), "Adjacency matrix is not symmetric"
    assert np.array_equal(
        adj, adj.astype(bool)
    ), "Adjacency matrix is not binary"
    assert (
        np.count_nonzero(deg - np.diag(np.diagonal(deg))) == 0
    ), "Degree matrix is not diagonal"


def test_parse_cdc():
    cdc_seqs = parse_cdc(cdc, pbp_patterns)
    check_data(cdc_seqs)


def test_parse_pmen():
    cdc_seqs = parse_cdc(cdc, pbp_patterns)
    pmen_seqs = parse_pmen(pmen, cdc_seqs, pbp_patterns)
    check_data(pmen_seqs)


def test_build_co_occurence_graph():
    cdc_seqs = parse_cdc(cdc, pbp_patterns)
    pmen_seqs = parse_pmen(pmen, cdc_seqs, pbp_patterns)

    cdc_graph = build_co_occurrence_graph(cdc_seqs, pbp_patterns)
    pmen_graph = build_co_occurrence_graph(pmen_seqs, pbp_patterns)

    check_graph(*cdc_graph)
    check_graph(*pmen_graph)
