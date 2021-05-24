import subprocess
import logging
from math import log2
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from scipy.stats.distributions import norm


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
    df = df.loc[
        (df.mic >= cdc.mic.min()) & (df.mic <= cdc.mic.max())
    ]  # drop samples outside the range of the training data

    def match_pbp_types(df, pbp, cdc=cdc):
        pbp_type = f"{pbp}_type"
        pbp_seq = f"{pbp}_seq"

        df = df.merge(
            cdc[[pbp_type, pbp_seq]].drop_duplicates(),
            left_on=pbp,
            right_on=pbp_seq,
            how="left",
        )

        # are there pbp types in pmen that aren't in cdc?
        if any(pd.isna(df[pbp_type])):
            last_pbp_type = cdc[pbp_type].astype(int).max()

            # dataframe for naming novel pbp types in pmen data
            additional_pbps = pd.DataFrame(
                df.loc[pd.isna(df[pbp_type])][pbp].unique()
            )
            additional_pbps[1] = list(
                range(
                    last_pbp_type + 1, last_pbp_type + len(additional_pbps) + 1
                )
            )
            additional_pbps.rename(
                columns={0: pbp_seq, 1: pbp_type}, inplace=True
            )
            additional_pbps[pbp_type] = additional_pbps[pbp_type].astype(str)

            # add newly named pbp sequences back to dataframe and format
            df = df.merge(
                additional_pbps,
                left_on=pbp,
                right_on=pbp_seq,
                how="outer",
            )
            df[f"{pbp_seq}_x"].fillna(df[f"{pbp_seq}_y"], inplace=True)
            df[f"{pbp_type}_x"].fillna(df[f"{pbp_type}_y"], inplace=True)
            df.drop(columns=[f"{pbp_seq}_y", f"{pbp_type}_y"], inplace=True)
            df.rename(
                columns={
                    f"{pbp_seq}_x": pbp_seq,
                    f"{pbp_type}_x": pbp_type,
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


def standardise_MICs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Where there is more than one MIC per isolate, fit a normal distribution
    and to the MIC values for each isolate and report the mean of the
    distribution to each
    """

    def standardise_MICs_(data):
        mean_log_mic = norm.fit(data.log2_mic, scale=1)[0]
        data.log2_mic = [mean_log_mic] * len(data)
        return data

    df = df.groupby("isolates").apply(standardise_MICs_)
    return df


def pairwise_blast_comparisons(
    df: pd.DataFrame, pbp: str
) -> Dict[str, Dict[str, float]]:
    pbp_type = f"{pbp}_type"
    pbp_seq = f"{pbp}_seq"

    df = df[[pbp_seq, pbp_type]].drop_duplicates()
    combos = pd.DataFrame(
        {
            i: df[pbp_seq] + "_" + df.loc[df[pbp_type] == i][pbp_seq].iloc[0]
            for i in df[pbp_type]
        }
    )

    # lower triangular to remove inverse matches for which e score will
    # be equal same
    combos_tril = pd.DataFrame(
        np.tril(combos.to_numpy()),
        columns=df[pbp_type].tolist(),
        index=df[pbp_type].tolist(),
    )
    np.fill_diagonal(combos_tril.values, 0)  # assign 0 to the diagonal

    def blast(sequences):
        if sequences == 0:
            return 0.0
        else:
            seq_1, seq_2 = sequences.split("_")
            try:
                e_value = float(
                    subprocess.check_output(
                        f"bash pairwise_blast.sh {seq_1} {seq_2}", shell=True
                    )
                )
            except Exception as ex:
                logging.warning(ex)
                e_value = -1.0
            return e_value

    e_values_tril = combos_tril.applymap(blast)
    e_values_triu = e_values_tril.T
    e_values = (
        e_values_tril + e_values_triu
    )  # combine so pbps can be searched in either order

    return e_values.to_dict()  # dictionary is quicker to search


def encode_sequences(
    data: pd.DataFrame, pbp_patterns: List[str]
) -> sparse.csr_matrix:
    """
    One hot encoding of sequences
    """
    amino_acids = [
        "A",
        "C",
        "G",
        "H",
        "I",
        "L",
        "M",
        "P",
        "S",
        "T",
        "V",
        "D",
        "E",
        "F",
        "K",
        "N",
        "Q",
        "R",
        "W",
        "Y",
    ]

    pbp_seqs = [i + "_seq" for i in pbp_patterns]
    # have to check all sequences of each type are same length because they are
    # combined into one before being encoded
    for pbp in pbp_seqs:
        assert (
            data[pbp].apply(len).nunique() == 1
        ), f"More than one length sequence found in column {pbp} cannot encode"

    encoded_sequences = []
    for pbp in pbp_seqs:
        sequences = data[pbp]
        n_var = len(sequences.iloc[0])

        # format as array for encoder
        sequences = np.array(sequences.apply(list).to_list()).astype(np.object)
        enc = OneHotEncoder(
            handle_unknown="error",
            categories=[amino_acids for i in range(n_var)],
            sparse=False,
            dtype=int,
        )
        enc.fit(sequences)

        encoded_sequences.append(enc.transform(sequences))

    data_encoded = np.concatenate(encoded_sequences, axis=1)

    return sparse.csr_matrix(data_encoded)


def build_co_occurrence_graph(
    df: pd.DataFrame, pbp_patterns: List[str]
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    matches = [
        df[f"{pbp}_type"].apply(lambda x: df[f"{pbp}_type"] == x).astype(int)
        for pbp in pbp_patterns
    ]  # adjacency matrix built from this will have self loops

    # add together matches to get adjacency matrix using all sequences
    weighted_adj = matches[0]
    if len(matches) > 1:
        for i in matches[1:]:
            weighted_adj += i

    adj = weighted_adj.applymap(
        lambda x: min(x, 1)
    ).to_numpy()  # adjacency matrix
    deg = np.diag(np.apply_along_axis(sum, 0, adj))  # degree matrix
    return sparse.csr_matrix(adj), sparse.csr_matrix(deg)


def main():
    cdc_raw = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen_raw = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
    maela_raw = pd.read_csv("../data/pneumo_pbp/maela_aa_df.csv")

    pbp_patterns = ["a1", "b2", "x2"]

    cdc = parse_cdc(cdc_raw, pbp_patterns)
    pmen = parse_pmen(pmen_raw, cdc, pbp_patterns)
    maela = parse_pmen(
        maela_raw, cdc, pbp_patterns
    )  # same format as raw pmen data

    cdc_encoded_sequences = encode_sequences(cdc, pbp_patterns)  # noqa: F841
    pmen_encoded_sequences = encode_sequences(pmen, pbp_patterns)  # noqa: F841
    maela_encoded_sequences = encode_sequences(  # noqa: F841
        maela, pbp_patterns
    )

    cdc_adj, cdc_deg = build_co_occurrence_graph(
        cdc, pbp_patterns
    )  # noqa: F841
    pmen_adj, pmen_deg = build_co_occurrence_graph(
        pmen, pbp_patterns
    )  # noqa: F841


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
