import logging
import os
import pickle
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from models import load_data
from utils import bin_labels


def plot_binned_predictions(**kwargs):
    model = load_model(
        "random_forest", "pmen"
    )  # second arg is irrelavant here

    def percent_correct_predictions(df):
        return len(df[df.accurate_prediction]) / len(df) * 100

    def count_correct_predictions(df):
        return len(df[df.accurate_prediction])

    for dataname, dataset in kwargs.items():
        predictions = model.predict(dataset[0])
        binned_labels, bin_values = bin_labels(dataset[1])
        df = pd.DataFrame(
            {
                "predictions": predictions,
                "labels": [bin_values[i - 1] for i in binned_labels],
            }
        )
        df["accurate_prediction"] = abs(df.predictions - df.labels) < 1
        percent_accuracy_by_bin = df.groupby(df.labels).apply(
            percent_correct_predictions
        )
        count_accuracy_by_bin = df.groupby(df.labels).apply(
            count_correct_predictions
        )


def plot_mics_by_type(df: pd.DataFrame, figname: str):
    df.sort_values(["mic", "isolates"], inplace=True)

    plt.clf()
    vp = sns.violinplot(x="isolates", y="mic", data=df)
    vp.set(xticklabels=[])  # remove tick labels
    vp.tick_params(bottom=False)  # remove the ticks
    vp.set(ylabel="Log2(MIC)")
    vp.set(xlabel="PBP Profile")
    vp.set(title="Distribution of MICs by PBP Profiles")
    vp
    plt.tight_layout()
    plt.savefig(figname)


def load_model(
    model_type: str,
    validation_data: str = None,
    *,
    blosum_inferred: bool = True,
    filtered: bool = False,
) -> Union[RandomForestRegressor, ElasticNet]:
    if blosum_inferred == filtered:
        raise ValueError("One of blosum_inferred or filtered must be true")

    if blosum_inferred:
        results_file = "results_blosum_inferred_pbp_types.pkl"
    elif filtered:
        results_file = "results_filtered_pbp_types.pkl"

    if validation_data is not None:
        results_file = f"{validation_data}_{results_file}"

    results_dir = f"results/{model_type}"

    with open(os.path.join(results_dir, results_file), "rb") as a:
        results = pickle.load(a)

    return results.model


def load_raw_data():
    cdc_raw = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen_raw = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
    maela_raw = pd.read_csv("../data/pneumo_pbp/maela_aa_df.csv")

    return cdc_raw, pmen_raw, maela_raw


def main():
    cdc_raw, pmen_raw, maela_raw = load_raw_data()
    plot_mics_by_type(cdc_raw, "cdc_mic_by_pbp_profile.png")
    plot_mics_by_type(pmen_raw, "pmen_mic_by_pbp_profile.png")
    plot_mics_by_type(maela_raw, "maela_mic_by_pbp_profile.png")


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
