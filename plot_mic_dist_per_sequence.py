from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import load_data, mean_acc_per_bin


def plot_variation(df: pd.DataFrame):
    df = df.rename(columns={"log2_mic": "log2(MIC)"})
    plt.clf()
    plt.rcParams.update({"font.size": 12})
    g = sns.catplot(
        data=df,
        x="Sequence Type",
        y="log2(MIC)",
        col="Population",
        col_wrap=2,
        kind="violin",
        sharex=False,
        sharey=True,
        cut=0,
    )
    g.set(xticks=[])
    g.savefig("temp.png")


def rename_sequence_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    To make plots look better rename all the sequence types within each population
    starting from 1 and in order of median mic
    """
    df = df.groupby(["Population", "Sequence Type"], as_index=False).apply(
        lambda x: x.assign(median_mic=x.mic.median())
    )

    def rename_sequence_types(data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values(["median_mic", "Sequence Type"])
        reindexed_seq_types = (
            data[["Sequence Type"]].drop_duplicates(ignore_index=True).reset_index()
        )
        data = data.merge(reindexed_seq_types, on="Sequence Type")
        return data.assign(**{"Sequence Type": data["index"]}).drop(columns="index")

    return df.groupby("Population").apply(rename_sequence_types)


def add_all_populations_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes it easier to plot if we add copies for the all population variation
    """

    def process_sequences(data: pd.DataFrame) -> pd.DataFrame:
        if data.Population.nunique() > 1:
            return data.assign(Population="All Populations")
        else:
            return data.iloc[0:0]

    df2 = df.groupby("Sequence Type", as_index=False).apply(process_sequences)
    return pd.concat((df, df2), ignore_index=True)


def type_sequences(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(
        [df.assign(Population=pop) for pop, df in all_data.items()], ignore_index=True
    )
    df = df.assign(
        combined_sequence=df[["a1_seq", "b2_seq", "x2_seq"]].agg("".join, axis=1)
    )
    uniq_sequences = (
        df[["combined_sequence"]].drop_duplicates(ignore_index=True).reset_index()
    )
    uniq_sequences = uniq_sequences.rename(columns={"index": "Sequence Type"})
    return df.merge(uniq_sequences, on="combined_sequence")


def compute_max_expected_MAB(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        combined_sequence=df[["a1_seq", "b2_seq", "x2_seq"]].agg("".join, axis=1)
    )
    df = (
        df.groupby("combined_sequence")
        .apply(lambda x: x.assign(mean_log_mic=x.log2_mic.mean()))
        .reset_index(drop=True)
    )
    return mean_acc_per_bin(df.mean_log_mic.values, df.log2_mic.values)


def max_expected_mab(df, population):
    df = df.loc[df.population == population]
    return mean_acc_per_bin(df.mean_log_mic.values, df.log2_mic.values)


def main(
    blosum_inference: bool = False,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = False,
    standardise_training_MIC: bool = True,
    standardise_test_and_val_MIC: bool = False,
    blosum_strictly_non_negative: bool = False,
    maela_correction: bool = True,
):

    train_data_population = "cdc"
    test_data_population_1 = "pmen"
    test_data_population_2 = "maela"

    cdc_train, pmen, maela, cdc_val = load_data(
        train_data_population=train_data_population,
        test_data_population_1=test_data_population_1,
        test_data_population_2=test_data_population_2,
        blosum_inference=blosum_inference,
        HMM_inference=HMM_inference,
        HMM_MIC_inference=HMM_MIC_inference,
        filter_unseen=filter_unseen,
        standardise_training_MIC=standardise_training_MIC,
        standardise_test_and_val_MIC=standardise_test_and_val_MIC,
        blosum_strictly_non_negative=blosum_strictly_non_negative,
        maela_correction=maela_correction,
    )
    cdc = pd.concat((cdc_train, cdc_val))
    all_data = {"CDC": cdc, "PMEN": pmen, "Maela": maela}

    df = type_sequences(all_data)
    df = add_all_populations_partition(df)
    df = rename_sequence_types(df)
