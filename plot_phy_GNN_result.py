import os
import pickle
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plot_model_fits import load_data as load_data_, process_data


def load_GNN_results(results_dir: str) -> pd.DataFrame:
    results_files = os.listdir(results_dir)
    all_results = {
        "train_pop": [],
        "test_pop_1": [],
        "test_pop_2": [],
        "metric": [],
        "score": [],
        "pop": [],
        "test_split": [],
    }
    metrics = ["mean_bin_acc", "acc", "loss"]
    populations = ["train_", "test_1_", "test_2_", "val_"]
    all_pops = ["cdc", "pmen", "maela"]
    for f in results_files:
        f = os.path.join(results_dir, f)
        with open(f, "rb") as a:
            result = pickle.load(a)
            metrics_df = result["metrics_df"]
            train_pop = result["train_population"]
            test_pop_1 = result["test_population_1"]
            test_pop_2 = list(set([train_pop, test_pop_1]) ^ set(all_pops))[0]
        for metric, pop in product(metrics, populations):
            all_results["train_pop"].append(train_pop)
            all_results["test_pop_1"].append(test_pop_1)
            all_results["test_pop_2"].append(test_pop_2)
            all_results["metric"].append(metric)
            all_results["score"].append(metrics_df[f"{pop}{metric}"].iloc[-1])
            if pop in ["train_", "val_"]:
                population = train_pop
            elif pop == "test_1_":
                population = test_pop_1
            else:
                population = test_pop_2
            all_results["pop"].append(population)
            all_results["test_split"].append(pop.replace("_", "").capitalize())
    all_results = pd.DataFrame(all_results)
    all_results.loc[all_results.metric == "mean_bin_acc", "metric"] = "mean_acc_per_bin"
    all_results.loc[all_results.metric == "acc", "metric"] = "accuracy"
    all_results.loc[all_results.metric == "loss", "metric"] = "MSE"
    all_results.loc[all_results["test_split"] == "Val", "test_split"] = "Validate"
    return all_results.rename(
        columns={
            "train_pop": "Train Population",
            "test_pop_1": "Test Population 1",
            "test_pop_2": "Test Population 2",
            "test_split": "Population",
        }
    ).assign(Model="GNN")[
        [
            "score",
            "Population",
            "Train Population",
            "Test Population 1",
            "Test Population 2",
            "Model",
            "metric",
        ]
    ]


def load_other_results(inference_method: str = "HMM_MIC") -> pd.DataFrame:
    populations = ["cdc", "pmen", "maela"]
    all_metrics = [
        process_data(load_data_(train_pop, inference_method, models=["random_forest"]))
        for train_pop in populations
    ]
    return pd.concat(
        [pd.concat([v.assign(metric=k) for k, v in i.items()]) for i in all_metrics]
    )


def plot_metric(all_results: pd.DataFrame, metric: str = "mean_acc_per_bin"):
    results = all_results.loc[all_results.metric == metric]
    plt.clf()
    fig = plt.figure()
    for i, (_, df) in enumerate(
        results.groupby(["Train Population", "Test Population 1"])
    ):
        ax = fig.add_subplot(3, 2, i + 1)
        sns.barplot(
            data=df,
            x="Population",
            y="score",
            hue="Model",
            ax=ax,
        )
        if i % 2 == 1:
            plt.ylabel("")
        if i < 4:
            plt.xlabel("")
        ax.get_legend().remove()


def single_pop_plot_metric(
    all_results: pd.DataFrame, metric: str = "mean_acc_per_bin", pop: str = "cdc"
):
    results = all_results.loc[
        (all_results.metric == metric) & (all_results["Train Population"] == pop)
    ]
    plt.clf()
    g = sns.FacetGrid(
        results,
        col="Test Population 1",
    )
    g.map(
        sns.barplot,
        "Population",
        "score",
        "Model",
        hue_order=["GNN", "random_forest"],
        order=["Train", "Validate", "Test1", "Test2"],
        palette=sns.color_palette(["tab:blue", "tab:orange"]),
    )
    g.add_legend()
    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle(f"Models Trained on {pop.upper()}")


if __name__ == "__main__":
    GNN_results = load_GNN_results("results/phylogeny_GNN_model/hamming_dist_tree")
    other_model_results = load_other_results()
    all_results = pd.concat([GNN_results, other_model_results])
    for train_pop in all_results["Train Population"].drop_duplicates():
        single_pop_plot_metric(all_results, pop=train_pop)
        plt.savefig(f"GNN_vs_RF_train_pop_{train_pop}.png")
