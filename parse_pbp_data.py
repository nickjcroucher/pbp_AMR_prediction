import subprocess
from math import log2
from itertools import combinations

import pandas as pd
from scipy import sparse
from typing import List


def get_pbp_sequence(
    pbp_pattern: str, df: pd.DataFrame, cols: pd.Series
) -> pd.Series:
    pbp_cols = cols[cols.str.startswith(pbp_pattern)]
    return df[pbp_cols].sum(axis=1)


def parse_pmen(
    pmen: pd.DataFrame, cdc: pd.DataFrame, pbp_patterns: List[str]
) -> pd.DataFrame:
    cols = pmen.columns.to_series()
    pbp_seqs = {pbp: get_pbp_sequence(pbp, pmen, cols) for pbp in pbp_patterns}
    df = pd.DataFrame(pbp_seqs)

    df["id"] = pmen.id
    df["mic"] = pmen.mic
    df["log2_mic"] = pmen.mic.apply(log2)

    df = df.loc[~pd.isna(df.mic)]  # drop samples with missing mic

    def match_pbp_types(df, pbp, cdc=cdc):
        df = df.merge(
            cdc[[f"{pbp}_type", f"{pbp}_seq"]].drop_duplicates(),
            left_on=pbp,
            right_on=f"{pbp}_seq",
            how="left",
        )

        # are there pbp types in pmen that aren't in cdc?
        if any(pd.isna(df[f"{pbp}_type"])):
            last_pbp_type = cdc[f"{pbp}_type"].astype(int).max()

            # dataframe for naming novel pbp types in pmen data
            additional_pbps = pd.DataFrame(
                df.loc[pd.isna(df[f"{pbp}_type"])][f"{pbp}"].unique()
            )
            additional_pbps[1] = list(
                range(
                    last_pbp_type + 1, last_pbp_type + len(additional_pbps) + 1
                )
            )
            additional_pbps.rename(
                columns={0: f"{pbp}_seq", 1: f"{pbp}_type"}, inplace=True
            )
            additional_pbps[f"{pbp}_type"] = additional_pbps[
                f"{pbp}_type"
            ].astype(str)

            # add newly named pbp sequences back to dataframe and format
            df = df.merge(
                additional_pbps,
                left_on=pbp,
                right_on=f"{pbp}_seq",
                how="outer",
            )
            df[f"{pbp}_seq_x"].fillna(df[f"{pbp}_seq_y"], inplace=True)
            df[f"{pbp}_type_x"].fillna(df[f"{pbp}_type_y"], inplace=True)
            df.drop(columns=[f"{pbp}_seq_y", f"{pbp}_type_y"], inplace=True)
            df.rename(
                columns={
                    f"{pbp}_seq_x": f"{pbp}_seq",
                    f"{pbp}_type_x": f"{pbp}_type",
                },
                inplace=True,
            )
        return df

    for pbp in pbp_patterns:
        df = match_pbp_types(df, pbp)

    df["isolates"] = df.a1_type + "-" + df.b2_type + "-" + df.x2_type
    df = df[cdc.columns.to_list()]

    return df


def parse_cdc(cdc: pd.DataFrame, pbp_patterns: List[str]) -> pd.DataFrame:
    cols = cdc.columns.to_series()
    pbp_seqs = {pbp: get_pbp_sequence(pbp, cdc, cols) for pbp in pbp_patterns}
    df = pd.DataFrame(pbp_seqs)

    df["isolate"] = cdc.isolate
    df["mic"] = cdc.mic
    df = df.loc[~pd.isna(df.mic)]  # drop samples with missing mic
    df.reindex()
    df["id"] = "cdc_" + df.index.astype(str)

    cdc_isolates = df.isolate.str.split("_", expand=True)[1]
    cdc_a1 = cdc_isolates.str.split("-", expand=True)[0]
    cdc_b2 = cdc_isolates.str.split("-", expand=True)[1]
    cdc_x2 = cdc_isolates.str.split("-", expand=True)[2]

    cdc_seqs = pd.DataFrame(
        {
            "id": df.id,
            "isolates": cdc_isolates,
            "mic": df.mic,
            "log2_mic": df.mic.apply(log2),
            "a1_type": cdc_a1,
            "b2_type": cdc_b2,
            "x2_type": cdc_x2,
            "a1_seq": df.a1,
            "b2_seq": df.b2,
            "x2_seq": df.x2,
        }
    )

    return cdc_seqs


def pairwise_blast_comparisons(data, pbp):
    combos = list(combinations(data[f"{pbp}_seq"], 2))

    e_values = {i: None for i in combos}
    n = 0
    n_max = len(combos)
    for pair in combos:
        e_value = float(
            subprocess.check_output(
                f"bash pairwise_blast.sh {pair[0]} {pair[1]}",
                shell=True,
                stderr=subprocess.DEVNULL,
            )
        )
        e_values[pair] = e_value
        print(f"\r{n}/{n_max}", end="")
        n += 1

    return e_values


def build_co_occurrence_graph(
    df: pd.DataFrame, pbp_patterns: List[str]
) -> sparse.coo_matrix:
    matches = [
        df[f"{pbp}_type"].apply(lambda x: df[f"{pbp}_type"] == x).astype(int)
        for pbp in pbp_patterns
    ]
    weighted_adj = sum(matches)
    return sparse.coo_matrix(weighted_adj)


def main():
    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")

    pbp_patterns = ["a1", "b2", "x2"]

    cdc = parse_cdc(cdc, pbp_patterns)
    pmen = parse_pmen(pmen, cdc, pbp_patterns)
