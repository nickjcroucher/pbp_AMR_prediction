import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple

import pandas as pd
import seaborn as sns

from utils import ResultsContainer


MODELS = ["random_forest", "DBSCAN", "DBSCAN_with_UMAP"]
POPULATIONS = ["cdc", "pmen", "maela"]


def plot_metrics(all_metrics: Dict[str, pd.DataFrame], train_pop: str):
    for metric, metric_data in all_metrics.items():
        per_test_pop = list(all_metrics[metric].groupby("test_pop_1"))


def process_data(data: List[ResultsContainer]) -> Dict[str, pd.DataFrame]:

    all_metrics = {}  # type: ignore
    for metric in ["accuracy", "MSE", "mean_acc_per_bin"]:
        all_metrics[metric] = {
            "train_score": [],
            "val_score": [],
            "test_1_score": [],
            "test_2_score": [],
            "train_pop": [],
            "test_pop_1": [],
            "test_pop_2": [],
            "model": [],
        }
        for results in data:
            all_metrics[metric]["train_score"].append(
                results.__getattribute__(f"training_{metric}")
            )
            all_metrics[metric]["val_score"].append(
                results.__getattribute__(f"validation_{metric}")
            )
            all_metrics[metric]["test_1_score"].append(
                results.__getattribute__(f"testing_{metric}_1")
            )
            all_metrics[metric]["test_2_score"].append(
                results.__getattribute__(f"testing_{metric}_2")
            )
            all_metrics[metric]["train_pop"].append(
                results.config["train_val_population"]
            )
            all_metrics[metric]["test_pop_1"].append(
                results.config["test_1_population"]
            )
            all_metrics[metric]["test_pop_2"].append(
                results.config["test_2_population"]
            )
            all_metrics[metric]["model"].append(results.model_type)

        all_metrics[metric] = pd.DataFrame(all_metrics[metric])

    return all_metrics


def load_data(train_pop: str) -> List[ResultsContainer]:
    all_data = []
    for model in MODELS:
        file_path_1 = f"results/{model}/train_pop_{train_pop}_results_blosum_inferred_pbp_types.pkl"  # noqa: E501
        file_path_2 = f"results/{model}/train_pop_{train_pop}_results_blosum_inferred_pbp_types(1).pkl"  # noqa: E501
        for fp in [file_path_1, file_path_2]:
            with open(fp, "rb") as a:
                all_data.append(pickle.load(a))

    return all_data


def main():
    for train_pop in POPULATIONS:
        data = load_data(train_pop)
        all_metrics = process_data(data, train_pop)


if __name__ == "__main__":
    main()
