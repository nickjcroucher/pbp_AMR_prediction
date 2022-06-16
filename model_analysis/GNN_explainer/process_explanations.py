import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_explanations(
    train_population: str = "cdc",
    test_population_1: str = "pmen",
    directory: str = "results/phylogeny_GNN_model/hamming_dist_tree/feature_importance/",
) -> Dict[str, pd.DataFrame]:
    train_file = f"important_features_{train_population}_{test_population_1}.csv"
    test_1_file = f"test1_important_features_{train_population}_{test_population_1}.csv"
    test_2_file = f"test2_important_features_{train_population}_{test_population_1}.csv"
    val_file = f"val_important_features_{train_population}_{test_population_1}.csv"
    explanation_files = {
        train_file: "train",
        test_1_file: "test_1",
        test_2_file: "test_2",
        val_file: "val",
    }
    all_explanations = {
        filepath: pd.read_csv(filepath, index_col=0)
        for filepath in [
            os.path.join(directory, filename) for filename in explanation_files.keys()
        ]
    }
    return {
        explanation_files[os.path.split(k)[1]]: v for k, v in all_explanations.items()
    }


def standardise_explanations(
    explanations: pd.DataFrame, log: bool = True
) -> pd.DataFrame:
    df = explanations / explanations.max().max()
    df = df.apply(np.reciprocal)
    if log:
        return df.apply(np.log10)
    else:
        return df


def mask_importances(features: pd.Series) -> pd.Series:
    features.name = None
    sorted_features = pd.DataFrame(features).sort_values(0).reset_index()
    max_idx = sorted_features[0].diff().idxmax()
    sorted_features = sorted_features.assign(
        index=sorted_features["index"].astype(int),
        masked_features=sorted_features[0].mask(
            pd.Series([True] * max_idx + [False] * (len(sorted_features) - max_idx)),
            other=0,
        ),
    )
    return sorted_features.sort_values("index", ignore_index=True).masked_features


def cluster_explanations(df: pd.DataFrame):
    pca = PCA(random_state=0).fit(df.values)


if __name__ == "__main__":
    explanations = load_explanations()
    explanations = pd.concat(explanations)
