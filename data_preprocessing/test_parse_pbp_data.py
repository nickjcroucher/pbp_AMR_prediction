from itertools import combinations

import pandas as pd
import numpy as np

from parse_pbp_data import (
    parse_cdc,
    parse_pmen,
    build_co_occurrence_graph,
    pairwise_blast_comparisons,
)

cdc = pd.read_csv("./data/pneumo_pbp/cdc_seqs_df.csv")
pmen = pd.read_csv("./data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
pbp_patterns = ["a1", "b2", "x2"]
cdc_seqs = parse_cdc(cdc, pbp_patterns)
pmen_seqs = parse_pmen(pmen, cdc_seqs, pbp_patterns)


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


def check_pairwise_blast_comparisons(df, pbp_patterns):
    for pbp in pbp_patterns:
        e_values_dict = pairwise_blast_comparisons(df, pbp)
        dict_keys = list(e_values_dict.keys())
        dict_keys.sort()

        pbp_types = list(df[f"{pbp}_type"].drop_duplicates())
        pbp_types.sort()
        assert (
            pbp_types == dict_keys
        ), f"PBP types in data and blast search results do not match, pbp: {pbp}"

        assert all(
            [
                dict_keys == sorted(list(e_values_dict[i].keys()))
                for i in e_values_dict
            ]
        ), f"Blast comparisons are incomplete, pbp: {pbp}"

        assert all(
            [
                all([type(i) == float for i in e_values_dict[j].values()])
                for j in dict_keys
            ]
        ), f"Unexpected data type encountered, pbp: {pbp}"

        for i in combinations(dict_keys, 2):
            assert (
                e_values_dict[i[0]][i[1]] == e_values_dict[i[1]][i[0]]
            ), f"Order of comparison affects result, pbp: {pbp}"


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
    check_data(cdc_seqs)


def test_parse_pmen():
    check_data(pmen_seqs)


def test_build_co_occurence_graph():
    cdc_graph = build_co_occurrence_graph(cdc_seqs, pbp_patterns)
    pmen_graph = build_co_occurrence_graph(pmen_seqs, pbp_patterns)

    check_graph(*cdc_graph)
    check_graph(*pmen_graph)


def test_pairwise_blast_comparisons():
    check_pairwise_blast_comparisons(cdc_seqs, pbp_patterns)
    check_pairwise_blast_comparisons(pmen_seqs, pbp_patterns)
