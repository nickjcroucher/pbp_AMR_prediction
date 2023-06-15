import logging
import os
import pickle
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from fit_models import perform_blosum_inference
from data_preprocessing.parse_pbp_data import parse_cdc, parse_pmen
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


def plot_mics_by_type(
    cdc: pd.DataFrame,
    maela: pd.DataFrame,
    pmen: pd.DataFrame,
    blosum_inference: bool,
    figname: str,
):
    if blosum_inference:
        for pbp in ["a1", "b2", "x2"]:
            pbp_type = f"{pbp}_type"
            train_types = set(cdc[pbp_type])

            pmen = perform_blosum_inference(
                pbp_type, pbp, train_types, cdc, pmen
            )
            maela = perform_blosum_inference(
                pbp_type, pbp, train_types, cdc, maela
            )
        pmen["isolates"] = (
            pmen["a1_type"] + "-" + pmen["b2_type"] + "-" + pmen["x2_type"]
        )
        maela["isolates"] = (
            maela["a1_type"] + "-" + maela["b2_type"] + "-" + maela["x2_type"]
        )

    cdc_types = pd.DataFrame(cdc.isolates.unique())
    pmen_types = pd.DataFrame(pmen.isolates.unique())
    maela_types = pd.DataFrame(maela.isolates.unique())

    df = cdc_types.merge(pmen_types).merge(maela_types)
    if len(df) == 0:
        raise ValueError("No shared pbp types between all three datasets")

    pop_mic_by_type = {"PBP_type": [], "MIC": [], "Population": []}
    for i in df[0]:
        for population, data in {
            "CDC": cdc,
            "PMEN": pmen,
            "Maela": maela,
        }.items():
            mics = data.loc[data.isolates == i].mic.tolist()
            n_mics = len(mics)
            pop_mic_by_type["PBP_type"] += [i] * n_mics
            pop_mic_by_type["MIC"] += mics
            pop_mic_by_type["Population"] += [population] * n_mics

    pop_mic_type_df = pd.DataFrame(pop_mic_by_type)

    plt.clf()
    sns.boxplot(x="PBP_type", y="MIC", hue="Population", data=pop_mic_type_df)
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
    cdc_raw = pd.read_csv("./data/pneumo_pbp/cdc_seqs_df.csv")
    pmen_raw = pd.read_csv("./data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
    maela_raw = pd.read_csv("./data/pneumo_pbp/maela_aa_df.csv")

    return cdc_raw, pmen_raw, maela_raw


def main():
    cdc_raw, pmen_raw, maela_raw = load_raw_data()

    pbp_patterns = ["a1", "b2", "x2"]

    cdc = parse_cdc(cdc_raw, pbp_patterns)
    pmen = parse_pmen(pmen_raw, cdc, pbp_patterns)
    maela = parse_pmen(
        maela_raw, cdc, pbp_patterns
    )  # same format as raw pmen data

    plot_mics_by_type(cdc, maela, pmen, True, "mic_range_per_population.png")


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
