import matplotlib.pyplot as plt
import pickle
from typing import Dict, List

import pandas as pd
import seaborn as sns

from utils import ResultsContainer


MODELS = ["random_forest", "DBSCAN", "DBSCAN_with_UMAP"]
POPULATIONS = ["cdc", "pmen", "maela"]


def plot_metrics(all_metrics: Dict[str, pd.DataFrame], train_pop: str):
    for metric, metric_data in all_metrics.items():
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
        plt.savefig(
            f"figures/model_and_pop_permutations/train_pop_{train_pop}_{metric}.png"  # noqa: E501
        )


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
        all_metrics = process_data(data)
        plot_metrics(all_metrics, train_pop)


if __name__ == "__main__":
    main()
