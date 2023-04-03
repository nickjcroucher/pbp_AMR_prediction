from typing import List

import pandas as pd

from plot_model_fits import load_data, process_data


def _get_data(models: List[str], inference_method: str) -> pd.DataFrame:
    train_pop = "cdc"
    data = load_data(
        train_pop,
        inference_method=inference_method,
        models=models,
        maela_correction=True,
    )
    data = process_data(data)["mean_acc_per_bin"]
    return data.assign(inference_method=inference_method)


def _per_model_data(model: str, inference_methods: str) -> pd.DataFrame:
    return pd.concat(
        [_get_data([model], inference_method) for inference_method in inference_methods]
    )


def elastic_net_data() -> pd.DataFrame:
    inference_methods = [
        "no_inference",
        "blosum_inferred",
    ]
    return _per_model_data("elastic_net", inference_methods)


def format_data(data: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat(
        [
            df.pivot(
                values="score",
                columns=["Test Population 1", "Population"],
                index=["Model", "inference_method"],
            )
            for _, df in data.groupby(
                ["Test Population 1", "Model", "inference_method"]
            )
        ]
    )
    df = df.apply(round)
    return df.reset_index()


def random_forest_data() -> pd.DataFrame:
    inference_methods = [
        "no_inference",
        "blosum_inferred",
        "blosum_inferred_strictly_non_negative",
        "HMM_MIC_inferred",
    ]
    return _per_model_data("random_forest", inference_methods)


def main():
    table_1_data = pd.concat([elastic_net_data(), random_forest_data()])
    table_1 = format_data(table_1_data)
    table_1.to_clipboard()
