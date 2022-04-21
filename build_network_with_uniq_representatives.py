import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from utils import load_data
from data_preprocessing.parse_pbp_data import encode_sequences, standardise_MICs


def get_data(n_representatives: int) -> pd.DataFrame:
    """
    Same way data is loaded in fit_models.py
    """
    train, test_1, test_2, val = load_data(
        train_data_population="cdc",
        test_data_population_1="pmen",
        test_data_population_2="maela",
        blosum_inference=False,
        HMM_inference=False,
        HMM_MIC_inference=False,
        filter_unseen=False,
        standardise_training_MIC=False,
        standardise_test_and_val_MIC=False,
    )
    original_datasets = {
        "cdc": pd.concat((train, val)),
        "pmen": test_1,
        "maela": test_2,
    }

    df = pd.concat([v.assign(pop=k) for k, v in original_datasets.items()])
    df = standardise_MICs(df)

    # sort using sample names to ensure same order with diff train test split
    sorted_ids = df.id.sort_values(ignore_index=True)
    sorted_ids = sorted_ids.sample(frac=1, random_state=0)
    df = df.assign(id=pd.Categorical(df.id, categories=sorted_ids, ordered=True))
    df = df.sort_values("id", ignore_index=True)
    df = df.assign(id=df.id.astype("str"))  # return to original data type

    df.loc[df["pop"] == "pmen", "id"] = "pmen_" + df.loc[df["pop"] == "pmen", "id"]
    df.loc[df["pop"] == "maela", "id"] = "maela_" + df.loc[df["pop"] == "maela", "id"]
    df = df.assign(id=df.id.str.replace("#", "_"))

    df = df.sample(frac=1, random_state=0)  # shuffle rows
    return (
        df.groupby(["isolates", "log2_mic"])
        .apply(lambda x: x.head(n_representatives))
        .reset_index(drop=True)
    )


def encoding(df: pd.DataFrame) -> pd.DataFrame:
    pbp_patterns = ["a1", "b2", "x2"]
    sparse_encoding = encode_sequences(df, pbp_patterns)
    dense_encoding = np.array(sparse_encoding.todense())
    informative_sites = dense_encoding[:, np.var(dense_encoding, 0) != 0]
    return pd.DataFrame(informative_sites, index=df.id)


def hamming_distance_matrix(encoded_seqs: pd.DataFrame) -> pd.DataFrame:
    dists = pairwise_distances(encoded_seqs, metric="hamming", n_jobs=-1)
    return pd.DataFrame(dists, columns=encoded_seqs.index, index=encoded_seqs.index)


if __name__ == "__main__":
    n = 4
    df = get_data(n)
    encoded_seqs = encoding(df)
    hamming_dists = hamming_distance_matrix(encoded_seqs)
    hamming_dists.to_parquet(
        f"hamming_distance_network/{n}_duplicates_hamming_dists.parquet"
    )
