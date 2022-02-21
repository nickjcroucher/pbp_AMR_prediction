"""
An implementation of the penicillin resistance linear regression model developed by
https://journals.asm.org/doi/10.1128/AAC.01370-21#T1
"""
from typing import Dict, Tuple, Union
from numpy import NaN

import pandas as pd

from ..utils import parse_extended_sequences


INTERCEPT = -4.61
A1_FEATURE_COEFFS = {(370, 374, "STMK", ""): 1.547, (574, 578, "TSQF", ""): 0.949}
B2_FEATURE_COEFFS = {(443, 447, "SSNT", ""): 1.202, (564, 569, "QLQPT", ""): 0.356}
X2_FEATURE_COEFFS = {
    (337, 341, "STMK", "SAFK"): 1.626,
    (505, 508, "KDA", "EDT"): 1.548,
    (505, 508, "KDA", "KEA"): 0.680,
    (546, 550, "LKSG", "VKSG"): 0.753,
}
# key is tuple (start, stop, WT seq, alternative seq), value is linear coeff


def _single_feature(motif: str, feature_def: Tuple) -> Union[bool, float]:
    wt_match = motif == feature_def[2]
    if feature_def[-1] == "":
        return wt_match
    elif wt_match:
        return True
    elif motif == feature_def[3]:
        return False
    else:
        return NaN


def sequence_features(sequences: pd.Series) -> pd.DataFrame:
    seq_name = sequences.name.split("_")[0]
    seq_loci = eval(f"{seq_name.upper()}_FEATURE_COEFFS")
    features_df = pd.concat(
        [
            sequences.str[slice(*a[:2])].apply(_single_feature, feature_def=a)
            for a in seq_loci
        ],
        axis=1,
    )
    features_df.columns = [f"{seq_name}_{i}" for i, j in enumerate(seq_loci)]
    return features_df.astype(float)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    pbp_patterns = ["a1", "b2", "x2"]
    df = parse_extended_sequences(df, pbp_patterns)
    return pd.concat(
        [sequence_features(df[f"{pbp}_seq"]) for pbp in pbp_patterns] + [df.log2_mic],
        axis=1,
    )


def load_data() -> Dict[str, pd.DateOffset]:
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_full_pbp_seqs_mic.csv")
    maela = pd.read_csv("../data/pneumo_pbp/maela_full_pbp_mic.csv")
    features = [generate_features(df) for df in [pmen, maela]]
    return pd.concat(
        [df.assign(population=pop) for df, pop in zip(features, ["pmen", "maela"])]
    )


class penicillin_model:
    def __init__(self, intercept, feature_coefficients):
        ...

    def predict(self, a1_features, b2_features, x2_features):
        ...


if __name__ == "__main__":
    features = load_data()
