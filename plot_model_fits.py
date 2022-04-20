import os
import pickle
from itertools import product
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import ResultsContainer


def plot_metrics(all_metrics: Dict[str, pd.DataFrame], train_pop: str, output_dir: str):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for metric, metric_data in all_metrics.items():
        plt.clf()
        df = all_metrics[metric]
        df.rename(columns={"score": metric}, inplace=True)
        g = sns.FacetGrid(
            df,
            col="Test Population 1",
            row="Model",
            margin_titles=True,
        )
        g.map(
            sns.barplot,
            "Population",
            metric,
            order=["Train", "Validate", "Test1", "Test2"],
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Models Trained on {train_pop}")
        g.fig.tight_layout()
        plt.savefig(f"{output_dir}/train_pop_{train_pop}_{metric}.png")
        plt.clf()


def process_data(data: List[ResultsContainer]) -> Dict[str, pd.DataFrame]:

    all_metrics = {}  # type: ignore
    for metric in ["accuracy", "MSE", "mean_acc_per_bin"]:
        all_metrics[metric] = {
            "score": [],
            "Population": [],
            "Train Population": [],
            "Test Population 1": [],
            "Test Population 2": [],
            "Model": [],
        }
        for results in data:
            all_metrics[metric]["score"].append(
                results.__getattribute__(f"training_{metric}")
            )
            all_metrics[metric]["score"].append(
                results.__getattribute__(f"validation_{metric}")
            )
            all_metrics[metric]["score"].append(
                results.__getattribute__(f"testing_{metric}_1")
            )
            all_metrics[metric]["score"].append(
                results.__getattribute__(f"testing_{metric}_2")
            )
            all_metrics[metric]["Population"].extend(
                ["Train", "Validate", "Test1", "Test2"]
            )
            all_metrics[metric]["Train Population"].extend(
                [results.config["train_val_population"]] * 4
            )
            all_metrics[metric]["Test Population 1"].extend(
                [results.config["test_1_population"]] * 4
            )
            all_metrics[metric]["Test Population 2"].extend(
                [results.config["test_2_population"]] * 4
            )
            all_metrics[metric]["Model"].extend([results.model_type] * 4)

        df = pd.DataFrame(all_metrics[metric])
        df.sort_values(by="Test Population 2", inplace=True)
        all_metrics[metric] = df

    return all_metrics


def load_data(
    train_pop: str,
    inference_method: str,
    just_hmm_scores: bool = False,
    models=["random_forest", "DBSCAN", "DBSCAN_with_UMAP", "elastic_net"],
) -> List[ResultsContainer]:
    all_data = []
    for model in models:
        if just_hmm_scores:
            file_path_1 = f"results/{model}/just_HMM_scores/train_pop_{train_pop}_results_{inference_method}_inferred_pbp_types.pkl"  # noqa: E501
            file_path_2 = f"results/{model}/just_HMM_scores/train_pop_{train_pop}_results_{inference_method}_inferred_pbp_types(1).pkl"  # noqa: E501
        else:
            file_path_1 = f"results/{model}/train_pop_{train_pop}_results_{inference_method}_inferred_pbp_types.pkl"  # noqa: E501
            file_path_2 = f"results/{model}/train_pop_{train_pop}_results_{inference_method}_inferred_pbp_types(1).pkl"  # noqa: E501

        for fp in [file_path_1, file_path_2]:
            with open(fp, "rb") as a:
                all_data.append(pickle.load(a))

    return all_data


def main():
    populations = ["cdc", "pmen", "maela"]
    inference_methods = ["blosum", "HMM", "HMM_MIC"]

    for train_pop, inference_method in product(populations, inference_methods):
        data = load_data(train_pop, inference_method)
        all_metrics = process_data(data)
        plot_metrics(
            all_metrics,
            train_pop,
            output_dir=f"figures/model_and_pop_permutations/{inference_method}_inferred",
        )


if __name__ == "__main__":
    main()
