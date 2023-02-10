import os
import pickle
from itertools import product
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import ResultsContainer


def plot_metrics(all_metrics: Dict[str, pd.DataFrame], train_pop: str, output_dir: str):
    """
    DEPRECATED
    """
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


def chapter_figure_model_comparison(
    metrics: pd.DataFrame,
    models: str,
    output_dir: str,
    inference_method: str,
    train_pop: str,
):
    metrics = metrics.copy()
    metrics.loc[metrics.Model == "lasso", "Model"] = "LASSO Interaction Model"
    metrics.loc[metrics.Model == "random_forest", "Model"] = "Random Forest"
    metrics = metrics.loc[metrics.Model.isin(models)]
    plt.clf()
    plt.rcParams.update({"font.size": 16})
    metrics = metrics.rename(columns={"score": "Mean Accuracy per Bin"})
    g = sns.catplot(
        data=metrics,
        x="Population",
        y="Mean Accuracy per Bin",
        hue="Model",
        col="Test Population 1",
        kind="bar",
        order=["Train", "Validate", "Test1", "Test2"],
    )
    g.set(ylim=[0, 100])
    fname = f"{'-'.join(models)}_{inference_method}_{train_pop}.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, fname))
    plt.clf()


def chapter_figure_inference_comparison(
    metrics: pd.DataFrame,
    model: str,
    output_dir: str,
    inference_method: str,
    train_pop: str,
    comparison_inference_methods: Optional[List[str]] = None,
    maela_correction: bool = False,
):
    metrics = metrics.assign(**{"Inference Method": inference_method})
    inference_method_renaming_dict = {
        "blosum_inferred": "BLOSUM inferred",
        "HMM_MIC_inferred": "HMM MIC inferred",
        "no_inference": "No Inference",
        "blosum_inferred_strictly_non_negative": "NN BLOSUM inferred",
    }
    if comparison_inference_methods is not None:
        for comparison_inference_method in comparison_inference_methods:
            data = load_data(
                train_pop,
                comparison_inference_method,
                models=[model],
                maela_correction=maela_correction,
            )
            comparison_metrics = process_data(data)["mean_acc_per_bin"]
            comparison_metrics = comparison_metrics.assign(
                **{"Inference Method": comparison_inference_method}
            )
            metrics = pd.concat((metrics, comparison_metrics), ignore_index=True)
        metrics = metrics.assign(
            **{
                "Inference Method": metrics["Inference Method"].apply(
                    lambda x: inference_method_renaming_dict[x]
                )
            }
        )

    plt.clf()
    plt.rcParams.update({"font.size": 16})
    metrics = metrics.rename(columns={"score": "Mean Accuracy per Bin"})
    g = sns.catplot(
        data=metrics,
        x="Population",
        y="Mean Accuracy per Bin",
        hue="Inference Method" if comparison_inference_method is not None else None,
        col="Test Population 1",
        kind="bar",
        color="#2b7bba" if comparison_inference_method is None else None,
        order=["Train", "Validate", "Test1", "Test2"],
    )
    g.set(ylim=[0, 100])
    # if comparison_inference_method is not None:
    # g.legend.remove()
    # g.fig.subplots_adjust(bottom=0.05, left=0.3)
    # g.fig.tight_layout()

    if comparison_inference_method is None:
        middle = inference_method
    else:
        middle = f"{inference_method}_{'-'.join(comparison_inference_methods)}"
    fname = f"{model}_{middle}_{train_pop}.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, fname))
    plt.clf()


def process_data(
    data: List[ResultsContainer], annotations: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    if annotations is not None:
        assert len(data) == len(annotations)

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
        if annotations is not None:
            all_metrics[metric]["annotation"] = []

        for n, results in enumerate(data):
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
            if annotations is not None:
                all_metrics[metric]["annotation"].extend([annotations[n]] * 4)

        df = pd.DataFrame(all_metrics[metric])
        df.sort_values(by="Test Population 2", inplace=True)
        all_metrics[metric] = df

    return all_metrics


def load_data(
    train_pop: str,
    inference_method: str,
    root_dir: str = "results",
    just_hmm_scores: bool = False,
    models=[
        "random_forest",
        "DBSCAN",
        "DBSCAN_with_UMAP",
        "elastic_net",
        "interaction_models",
    ],
    maela_correction: bool = False,
) -> List[ResultsContainer]:
    if maela_correction:
        root_dir = os.path.join(root_dir, "maela_updated_mic_rerun")

    all_data = []
    for model in models:
        if just_hmm_scores:
            file_path_1 = f"{model}/just_HMM_scores/train_pop_{train_pop}_results_{inference_method}_pbp_types.pkl"  # noqa: E501
            file_path_2 = f"{model}/just_HMM_scores/train_pop_{train_pop}_results_{inference_method}_pbp_types(1).pkl"  # noqa: E501
        else:
            file_path_1 = f"{model}/train_pop_{train_pop}_results_{inference_method}_pbp_types.pkl"  # noqa: E501
            file_path_2 = f"{model}/train_pop_{train_pop}_results_{inference_method}_pbp_types(1).pkl"  # noqa: E501

        file_path_1 = os.path.join(root_dir, file_path_1)
        file_path_2 = os.path.join(root_dir, file_path_2)

        for fp in [file_path_1, file_path_2]:
            with open(fp, "rb") as a:
                all_data.append(pickle.load(a))

    return all_data


def main():
    """
    DEPRECATED
    """
    populations = ["cdc", "pmen", "maela"]
    inference_methods = ["blosum_inferred", "HMM_inferred", "HMM_MIC_inferred"]

    for train_pop, inference_method in product(populations, inference_methods):
        data = load_data(train_pop, inference_method)
        all_metrics = process_data(data)
        plot_metrics(
            all_metrics,
            train_pop,
            output_dir=f"figures/model_and_pop_permutations/{inference_method}_inferred",
        )


if __name__ == "__main__":
    train_pop = "cdc"
    inference_method = "blosum_inferred"
    model = "elastic_net"

    data = load_data(train_pop, inference_method, models=[model], maela_correction=True)
    all_metrics = process_data(data, annotations=None)
    chapter_figure_inference_comparison(
        all_metrics["mean_acc_per_bin"],
        model,
        "chapter_figures_maela_correction/",
        inference_method,
        train_pop,
        ["no_inference", "HMM_MIC_inferred", "blosum_inferred_strictly_non_negative"],
        maela_correction=True,
    )
    chapter_figure_model_comparison(
        all_metrics["mean_acc_per_bin"],
        ["Random Forest", "LASSO Interaction Model"],
        "chapter_figures_maela_correction/",
        inference_method,
        train_pop,
    )
