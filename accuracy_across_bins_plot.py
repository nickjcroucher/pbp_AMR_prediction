import pickle
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fit_models import load_and_format_data
from models.phylogeny_GNN_model import GCN
from utils import accuracy, bin_labels, ResultsContainer


INPUT_DATA = load_and_format_data(
    "cdc",
    "pmen",
    "maela",
    blosum_inference=False,
    HMM_inference=False,
    HMM_MIC_inference=False,
    filter_unseen=False,
    include_HMM_scores=False,
    just_HMM_scores=False,
    standardise_training_MIC=True,
    standardise_test_and_val_MIC=False,
    maela_correction=True,
)
TRAIN = INPUT_DATA["train"][1].values
VAL = INPUT_DATA["val"][1].values
PMEN = INPUT_DATA["test_1"][1].values
MAELA = INPUT_DATA["test_2"][1].values


files = {
    "Random Forest": {
        "pmen": "results/maela_updated_mic_rerun/random_forest/train_pop_cdc_results_no_inference_pbp_types.pkl",
        "maela": "results/maela_updated_mic_rerun/random_forest/train_pop_cdc_results_no_inference_pbp_types(1).pkl",
    },
    "GNN": {
        "pmen": "results/maela_updated_mic_rerun/phylogeny_GNN_model/hamming_dist_tree/GNN_cdc_pmen.pkl",
        "maela": "results/maela_updated_mic_rerun/phylogeny_GNN_model/hamming_dist_tree/GNN_cdc_maela.pkl",
    },
    "LIM": {
        "pmen": "results/maela_updated_mic_rerun/interaction_models/train_pop_cdc_results_no_inference_pbp_types.pkl",
        "maela": "results/maela_updated_mic_rerun/interaction_models/train_pop_cdc_results_no_inference_pbp_types(1).pkl",
    },
    "DBSCAN-UMAP": {
        "pmen": "results/maela_updated_mic_rerun/DBSCAN_with_UMAP/train_pop_cdc_results_no_inference_pbp_types.pkl",
        "maela": "results/maela_updated_mic_rerun/DBSCAN_with_UMAP/train_pop_cdc_results_no_inference_pbp_types(1).pkl",
    }
    # "LIM (just interactions)": {
    #     "pmen": "results/interaction_models/train_pop_cdc_results_no_inference_pbp_types.pkl",
    #     "maela": "results/interaction_models/train_pop_cdc_results_no_inference_pbp_types(1).pkl",
    # },
    # "LIM (full sequences & interactions)": {
    #     "pmen": "results/interaction_models/including_all_sequences/train_pop_cdc_results_no_inference_pbp_types.pkl",
    #     "maela": "results/interaction_models/including_all_sequences/train_pop_cdc_results_no_inference_pbp_types(1).pkl",
    # },
}


def open_file(fpath: str) -> Union[Dict, ResultsContainer]:
    with open(fpath, "rb") as a:
        return pickle.load(a)


def per_bin_accuracy(labels: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    binned_labels, median_bin_values = bin_labels(labels)

    df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": predictions,
            "binned_labels": binned_labels,
        }
    )  # to allow quick searches across bins

    # percentage accuracy per bin
    def _get_accuracy(d):
        acc = accuracy(
            d.labels.to_numpy(),
            d.predictions.to_numpy(),
        )
        return acc

    bin_accuracies = df.groupby(df.binned_labels).apply(_get_accuracy)
    bin_accuracies = bin_accuracies.reset_index()

    # necessary to insert a bin if there were no labels in that bin
    # will make it easier to plot
    all_bin_labels = np.array(range(1, len(median_bin_values) + 1))
    all_bin_labels_df = pd.DataFrame(
        {"bin_labels": all_bin_labels, "median_bin_values": median_bin_values}
    )
    bin_accuracies = all_bin_labels_df.merge(
        bin_accuracies, left_on="bin_labels", right_on="binned_labels", how="left"
    )
    bin_accuracies = bin_accuracies.rename(columns={0: "accuracy"})[
        ["median_bin_values", "accuracy"]
    ]
    return bin_accuracies.fillna(0)


def _parse_GNN(
    results: Dict,
) -> List[pd.DataFrame]:
    data_indices = ["idx_train", "idx_val", "idx_test_1", "idx_test_2"]
    model = GCN(
        results["X"].shape[1],
        results["X"].shape[1],
        **results["model_params"],
        nclass=1,
        sparse_adj=False,
    )
    model.load_state_dict(results["model_state_dict"])
    predictions = model(results["X"], results["adj"]).squeeze().detach()
    return [
        per_bin_accuracy(
            results["y"][results[data_split]].squeeze(),
            np.array(predictions[results[data_split]]),
        )
        for data_split in data_indices
    ]


def _parse_other_models(
    results: ResultsContainer,
    test_population_1: str,
) -> List[pd.DataFrame]:
    data_splits = [
        "training_predictions",
        "validation_predictions",
        "testing_predictions_1",
        "testing_predictions_2",
    ]
    true_labels = [
        TRAIN,
        VAL,
        PMEN if test_population_1 == "pmen" else MAELA,
        MAELA if test_population_1 == "pmen" else PMEN,
    ]
    return [
        per_bin_accuracy(labels, results.__getattribute__(data_split))
        for data_split, labels in zip(data_splits, true_labels)
    ]


def parse_predictions(
    results_object: Union[Dict, ResultsContainer],
    model_name: str,
    test_population_1: str,
) -> pd.DataFrame:
    if model_name == "GNN":
        binned_accuracy = _parse_GNN(results_object)
    else:
        binned_accuracy = _parse_other_models(results_object, test_population_1)

    data_splits = ["Train", "Validate", "Test1", "Test2"]
    return pd.concat(
        [
            df.assign(
                data_split=i,
                model_type=model_name,
                test_population_1=test_population_1,
            )
            for df, i in zip(binned_accuracy, data_splits)
        ]
    )


if __name__ == "__main__":
    model_results = {
        model: {k: open_file(v) for k, v in model_dict.items()}
        for model, model_dict in files.items()
    }
    accuracy_bins = [
        [parse_predictions(v, model_name, k) for k, v in model_dict.items()]
        for model_name, model_dict in model_results.items()
    ]
    df = pd.concat([x for l in accuracy_bins for x in l], ignore_index=True)

    df = df.rename(
        columns={
            "accuracy": "Accuracy",
            "test_population_1": "Test Pop. 1",
            "data_split": "Population",
            "median_bin_values": "Bin Median MIC",
        }
    )
    df.loc[df["Test Pop. 1"] == "pmen", "Test Pop. 1"] = "PMEN"
    df.loc[df["Test Pop. 1"] == "maela", "Test Pop. 1"] = "Maela"

    df = df.loc[df.Population != "Train"]
    df = df.assign(**{"Model Type": df.model_type})

    plt.clf()
    plt.rcParams.update({"font.size": 16})
    sns.catplot(
        data=df,
        x="Bin Median MIC",
        y="Accuracy",
        col="Population",
        row="Model Type",
        hue="Test Pop. 1",
        kind="point",
        linestyles="--",
        margin_titles=True,
    )
    plt.savefig("temp.png")

    # for model_type in df.model_type.drop_duplicates():
    #     plt.clf()
    #     plt.rcParams.update({"font.size": 12})
    #     sns.catplot(
    #         data=df.loc[df.model_type == model_type],
    #         x="Bin Median MIC",
    #         y="Accuracy",
    #         row="Population",
    #         col="Test Pop. 1",
    #         kind="bar",
    #         color="#1f77b4",
    #     )
    #     plt.title(model_type)
    #     plt.subplots_adjust(bottom=0.1)
    #     plt.savefig(f"{model_type}.png")
