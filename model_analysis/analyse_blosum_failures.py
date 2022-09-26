import pickle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_preprocessing.parse_pbp_data import encode_sequences, standardise_MICs
from utils import load_data, mean_acc_per_bin, parse_blosum_matrix, ResultsContainer


# def load_results(model: str = "elastic_net") -> Tuple[ResultsContainer]:
#     with open(
#         f"results/{model}/train_pop_cdc_results_blosum_inferred_pbp_types.pkl", "rb"
#     ) as a:
#         results_1 = pickle.load(a)
#     with open(
#         f"results/{model}/train_pop_cdc_results_HMM_MIC_inferred_pbp_types.pkl", "rb"
#     ) as a:
#         results_2 = pickle.load(a)
#     with open(
#         f"results/{model}/train_pop_cdc_results_blosum_inferred_pbp_types(1).pkl",
#         "rb",
#     ) as a:
#         results_3 = pickle.load(a)
#     with open(
#         f"results/{model}/train_pop_cdc_results_HMM_MIC_inferred_pbp_types(1).pkl",
#         "rb",
#     ) as a:
#         results_4 = pickle.load(a)

#     return results_1, results_2, results_3, results_4


def pbp_representatives(original_datasets, n_PBP_representatives=-1):
    df = pd.concat([v.assign(pop=k) for k, v in original_datasets.items()])
    df = standardise_MICs(df)

    # sort using sample names to ensure same order with diff train test split
    sorted_ids = df.id.sort_values(ignore_index=True)
    sorted_ids = sorted_ids.sample(frac=1, random_state=0)
    df = df.assign(id=pd.Categorical(df.id, categories=sorted_ids, ordered=True))
    df = df.sort_values("id", ignore_index=True)
    df = df.assign(id=df.id.astype("str"))  # return to original data type

    df = df.sample(frac=1, random_state=0)  # shuffle rows
    df_uniq = (
        df.groupby(["isolates", "log2_mic"])
        .apply(lambda x: x.head(n_PBP_representatives))
        .reset_index(drop=True)
    )
    standardised_datasets = {
        k: v.reset_index(drop=True).drop(columns="pop")
        for k, v in df_uniq.groupby("pop")
    }
    # same order
    return {k: standardised_datasets[k] for k in original_datasets.keys()}


def get_data(
    test_data_population_1: str = "pmen",
    test_data_population_2: str = "maela",
    blosum_inference: bool = True,
    HMM_MIC_inference: bool = False,
    blosum_strictly_non_negative: bool = False,
    pbp_representatives: bool = False,
) -> Tuple[Dict[str, pd.DataFrame]]:
    train, test_1, test_2, val = load_data(
        train_data_population="cdc",
        test_data_population_1=test_data_population_1,
        test_data_population_2=test_data_population_2,
        blosum_inference=blosum_inference,
        filter_unseen=False,
        standardise_training_MIC=True,
        HMM_MIC_inference=HMM_MIC_inference,
        blosum_strictly_non_negative=blosum_strictly_non_negative,
    )
    blosum_inferred_sequences = {
        "train": train,
        "test_1": test_1,
        "test_2": test_2,
        "val": val,
    }
    train, test_1, test_2, val = load_data(
        train_data_population="cdc",
        test_data_population_1=test_data_population_1,
        test_data_population_2=test_data_population_2,
        blosum_inference=False,
        filter_unseen=False,
        standardise_training_MIC=True,
    )
    original_sequences = {
        "train": train,
        "test_1": test_1,
        "test_2": test_2,
        "val": val,
    }
    if pbp_representatives:
        blosum_inferred_sequences = pbp_representatives(blosum_inferred_sequences)
        original_sequences = pbp_representatives(original_sequences)
    return blosum_inferred_sequences, original_sequences


def _combine_preds_and_sequences(
    seuqences_df: pd.DataFrame, predictions: np.array
) -> pd.DataFrame:
    return seuqences_df.assign(predictions=predictions)


# def _get_AA_sub(AA_1: str, AA_2: str) -> str:
#     if AA_1 == AA_2:
#         return "-"
#     else:
#         return f"{AA_1}-{AA_2}"


# def _get_unique_AA_substitutions(
#     blosum_seqs: pd.DataFrame,
#     original_seqs: pd.DataFrame,
#     pbp: str,
# ):
#     blosum_seqs = blosum_seqs.reset_index(drop=True)
#     original_seqs = original_seqs.reset_index(drop=True)

#     df = pd.DataFrame(
#         {
#             "blosum_inferred": blosum_seqs[pbp].apply(lambda x: [i for i in x]),
#             "original": original_seqs[pbp].apply(lambda x: [i for i in x]),
#         }
#     )
#     all_substitutions = df.apply(
#         lambda a: np.apply_along_axis(
#             lambda x: _get_AA_sub(*x), axis=0, arr=np.stack(a.values)
#         ),
#         axis=1,
#     )


def _compute_distance(
    blosum_seqs: pd.DataFrame,
    original_seqs: pd.DataFrame,
    pbp: str,
    blosum: bool = True,
) -> pd.DataFrame:
    blosum_seqs = blosum_seqs.reset_index(drop=True)
    original_seqs = original_seqs.reset_index(drop=True)

    df = pd.DataFrame(
        {"blosum_inferred": blosum_seqs[pbp], "original": original_seqs[pbp]}
    )

    if blosum:
        blosum_scores = parse_blosum_matrix()

        def compute_distance(seq_1, seq_2):
            return sum([blosum_scores[i][j] for i, j in zip(seq_1, seq_2)])

    else:

        def compute_distance(seq_1, seq_2):
            if seq_1 == seq_2:
                return 0
            else:
                return sum(seq_1[i] != seq_2[i] for i in range(len(seq_1)))

    return blosum_seqs.assign(
        **{f"{pbp}_distance": df.apply(lambda x: compute_distance(*x), axis=1)}
    )


def combine_preds_and_sequences(
    results: ResultsContainer,
    blosum_inferred_sequences: Dict[str, pd.DataFrame],
    original_sequences: Dict[str, pd.DataFrame],
    blosum: bool = True,
) -> Dict[str, pd.DataFrame]:
    blosum_inferred_sequences = {
        k: _combine_preds_and_sequences(v, results.__getattribute__(i))
        for (k, v), i in zip(
            blosum_inferred_sequences.items(),
            [
                "training_predictions",
                "testing_predictions_1",
                "testing_predictions_2",
                "validation_predictions",
            ],
        )
    }
    blosum_inferred_sequences = {
        k: df.assign(mic_difference=abs(df.log2_mic - df.predictions))
        for k, df in blosum_inferred_sequences.items()
    }

    for pbp in ["a1_seq", "b2_seq", "x2_seq"]:
        blosum_inferred_sequences = {
            k: _compute_distance(df, original_sequences[k], pbp, blosum=blosum)
            for k, df in blosum_inferred_sequences.items()
        }
    return {
        k: v.assign(total_pbp_distance=v[v.columns[-3:]].sum(axis=1))
        for k, v in blosum_inferred_sequences.items()
    }


def compare_imputed(df):
    df = df.assign(imputed=df.total_pbp_distance >= 1)
    return df.groupby("imputed").apply(
        lambda x: mean_acc_per_bin(x.predictions.values, x.log2_mic.values)
    )


def compare_used_features(
    inferred_sequences: Dict[pd.DataFrame],
    original_sequences: Dict[pd.DataFrame],
    results: ResultsContainer,
):
    indices_of_used = results.model.feature_importances_ != 0
    pbps = ["a1", "b2", "x2"]
    output = {}
    for data in ["test_1", "test_2", "val"]:
        inferred = encode_sequences(inferred_sequences[data], pbps).toarray()
        original = encode_sequences(original_sequences[data], pbps).toarray()
        comparison = inferred[:, indices_of_used] != original[:, indices_of_used]
        output[data] = comparison
    return output


def main():
    with open(
        "results/random_forest/train_pop_cdc_results_blosum_inferred_strictly_non_negative_pbp_types.pkl",
        "rb",
    ) as a:
        results = pickle.load(a)
    blosum_inferred_sequences, original_sequences = get_data(
        "pmen", "maela", blosum_strictly_non_negative=True
    )
    for pbp in ["a1_seq", "b2_seq", "x2_seq"]:
        blosum_inferred_sequences = {
            k: _compute_distance(df, original_sequences[k], pbp, blosum=False)
            for k, df in blosum_inferred_sequences.items()
        }
    blosum_inferred_sequences = {
        k: v.assign(total_pbp_distance=v[v.columns[-3:]].sum(axis=1))
        for k, v in blosum_inferred_sequences.items()
    }
    {
        k: v.total_pbp_distance.value_counts()
        for k, v in blosum_inferred_sequences.items()
    }

    # blosum_inferred_sequences = combine_preds_and_sequences(
    #     results, blosum_inferred_sequences, original_sequences, blosum=False
    # )
    # {k: compare_imputed(v) for k, v in blosum_inferred_sequences.items()}


def name_models(results_object: ResultsContainer) -> str:
    if results_object.config["blosum_inference"]:
        inference = "BLOSUM Inference"
    elif results_object.config["hmm_mic_inference"]:
        inference = "HMM-MIC Inference"
    else:
        raise ValueError()

    model = " ".join([i.capitalize() for i in results_object.model_type.split("_")])

    return f"{model} {inference}"


def summarise_performance(results_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    out = {}
    for i, j in zip(["val", "test_1", "test_2"], ["Validate", "Test1", "Test2"]):
        df = results_dict[i]
        df_inferred = df.loc[df.total_pbp_distance.astype(bool)]
        df_non_inferred = df.loc[~df.total_pbp_distance.astype(bool)]
        inferred_mab = mean_acc_per_bin(
            df_inferred.predictions.values, df_inferred.log2_mic.values
        )
        non_inferred_mab = mean_acc_per_bin(
            df_non_inferred.predictions.values, df_non_inferred.log2_mic.values
        )
        out[f"{j} Imputed"] = inferred_mab
        out[f"{j} Not Imputed"] = non_inferred_mab
    return out


def plot_imputed_vs_non():
    rf_results = load_results("random_forest")
    en_results = load_results("elastic_net")

    # order so pmen test pop 1 is first
    results = rf_results[:2] + en_results[:2] + rf_results[2:] + en_results[2:]

    blosum_inferred_sequences, original_sequences = get_data("pmen", "maela")
    results1 = {
        name_models(r): combine_preds_and_sequences(
            r, blosum_inferred_sequences, original_sequences, False
        )
        for r in results[:4]
    }

    blosum_inferred_sequences, original_sequences = get_data("maela", "pmen")
    results2 = {
        name_models(r): combine_preds_and_sequences(
            r, blosum_inferred_sequences, original_sequences, False
        )
        for r in results[4:]
    }
    results1 = {k: summarise_performance(v) for k, v in results1.items()}
    results2 = {k: summarise_performance(v) for k, v in results2.items()}

    df1 = pd.concat(
        [pd.DataFrame(v, index=[0]).assign(model=k) for k, v in results1.items()]
    ).assign(test_pop_1="pmen")
    df2 = pd.concat(
        [pd.DataFrame(v, index=[0]).assign(model=k) for k, v in results2.items()]
    ).assign(test_pop_1="maela")
    df = pd.concat([df1, df2])

    df = df.melt(id_vars=["model", "test_pop_1"], value_name="Mean Accuracy per Bin")
    df = df.assign(population=df.variable.str.split(" ", expand=True)[0])
    df = df.assign(imputed=df.variable.str.split(" ", expand=True)[1])
    df = df.drop(columns="variable")
    df = df.assign(Imputed=df.imputed == "Imputed").drop(columns="imputed")

    df = df.assign(
        Model=df.model.apply(
            lambda x: {
                "Random Forest BLOSUM Inference": "RF BLOSUM",
                "Random Forest HMM-MIC Inference": "RF HMM-MIC",
                "Elastic Net BLOSUM Inference": "EN BLOSUM",
                "Elastic Net HMM-MIC Inference": "EN HMM-MIC",
            }[x]
        )
    ).drop(columns="model")
    df = df.assign(
        **{
            "Test Population 1": df.test_pop_1.apply(
                lambda x: {
                    "pmen": "PMEN",
                    "maela": "Maela",
                }[x]
            )
        }
    ).drop(columns="test_pop_1")
    df = df.rename(columns={"population": "Population"})
    df = df.sort_values(["Model", "Imputed"], ascending=[True, False])

    plt.rcParams.update({"font.size": 12})
    sns.catplot(
        data=df.loc[df.Model.str.startswith("RF")],
        x="Population",
        y="Mean Accuracy per Bin",
        hue="Imputed",
        kind="bar",
        col="Test Population 1",
        row="Model",
        col_order=["PMEN", "Maela"],
        order=["Validate", "Test1", "Test2"],
        hue_order=[True, False],
    )
    plt.subplots_adjust(bottom=0.1)