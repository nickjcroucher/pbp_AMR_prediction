import logging
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib_venn
import pandas as pd

from fit_models import fit_model, load_and_format_data, optimise_hps


def venn_diagram(
    fname: str, features_cuttoff_dict: Dict, p_values: List[float] = [0.05, 0.1, 0.15]
):
    feature_sets = [{i[0] for i in features_cuttoff_dict[p]} for p in p_values]

    plt.clf()
    matplotlib_venn.venn3_unweighted(
        feature_sets, set_labels=[f"p value cuttoff = {i}" for i in p_values]
    )
    plt.savefig(fname)


def parallel_coords_plot(
    fname: str, features_cuttoff_dict: Dict, p_values: List[float] = [0.05, 0.1, 0.15]
):
    feature_intersection = set.intersection(
        *[{i[0] for i in features_cuttoff_dict[p]} for p in p_values]
    )
    feature_dicts = [
        {x[0]: x[1] for x in features_cuttoff_dict[p] if x[0] in feature_intersection}
        for p in p_values
    ]
    feature_dicts = [
        {"-".join([str(i) for i in k]): v for k, v in f_dict.items()}
        for f_dict in feature_dicts
    ]
    df = pd.concat(
        [pd.DataFrame(feature_dicts[i], index=[j]) for i, j in enumerate(p_values)]
    )
    df = df.sort_values(min(p_values), axis=1, ascending=False)
    df = df.rename(columns={j: i + 1 for i, j in enumerate(df.columns)})
    df = df.reset_index().rename(columns={"index": "p-value"})

    plt.clf()
    pd.plotting.parallel_coordinates(df, "p-value")
    plt.legend(title="p-value")
    plt.xlabel("Feature Pair")
    plt.ylabel("Linear Coefficient")
    plt.savefig(fname)


def p_value_cuttoff_tests(
    p_value: float,
    test_data_population_1: str,
    test_data_population_2: str,
    paired_sf_p_values: List,
):
    interactions = [i[0] for i in paired_sf_p_values if i[1] < p_value]

    data = load_and_format_data(
        "cdc",
        test_data_population_1,
        test_data_population_2,
        interactions=interactions,
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

    train = data["train"]
    test = data["test_1"]

    model_type = "lasso"
    pbounds = {"alpha": [0.05, 0.8]}

    logging.info("Optimising the model for the test data accuracy")
    optimizer = optimise_hps(train, test, pbounds, model_type)  # select hps using GP

    logging.info(
        f"Fitting model with optimal hyperparameters: {optimizer.max['params']}"  # noqa: E501
    )
    model = fit_model(train, model_type=model_type, **optimizer.max["params"])
    return [i for i in zip(interactions, model.coef_) if i[1] != 0]


def main(
    test_data_population_1: str = "pmen",
    test_data_population_2: str = "maela",
):
    logging.info("Loading inferred interaction data")
    with open(
        f"results/intermediates/maela_updated_mic_rerun/cdc/no_inference_paired_sf_p_values_test1_{test_data_population_1}.pkl",
        "rb",
    ) as a:
        paired_sf_p_values = pickle.load(a)

    p_values = [0.1, 0.2, 0.3]
    features_cuttoff_dict = {
        p_value: p_value_cuttoff_tests(
            p_value,
            test_data_population_1,
            test_data_population_2,
            paired_sf_p_values,
        )
        for p_value in p_values
    }
    with open(
        f"results/intermediates/maela_updated_mic_rerun/cdc/sensitivity_analysis/inc_features_p_value_cuttoff_{test_data_population_1}.pkl",
        "wb",
    ) as a:
        pickle.dump(features_cuttoff_dict, a)

    venn_diagram(
        f"results/intermediates/maela_updated_mic_rerun/cdc/sensitivity_analysis/{test_data_population_1}_venn_diagram2.png",
        features_cuttoff_dict,
        p_values=p_values,
    )
    parallel_coords_plot(
        f"results/intermediates/maela_updated_mic_rerun/cdc/sensitivity_analysis/{test_data_population_1}_parcoords_plot2.png",
        features_cuttoff_dict,
        p_values=p_values,
    )
