import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_clusters(fpath: str) -> pd.DataFrame:
    with open(fpath, "rb") as a:
        results = pickle.load(a)
    clusters_df = pd.DataFrame(
        {
            "MIC": results.training_predictions,
            "cluster_labels": results.model.clustering.labels_,
        }
    )
    clusters_df = clusters_df.loc[clusters_df.cluster_labels != -1]  # drop unassigned
    median_cluster_MIC = clusters_df.groupby("cluster_labels").apply(
        lambda x: x.median()
    )
    df = clusters_df.join(median_cluster_MIC, on="cluster_labels", rsuffix="_median")
    df = df.sort_values("MIC_median")
    df = df.assign(
        **{
            "cluster_labels": pd.factorize(df.cluster_labels_median)[0].astype(str),
            "Test Population 1": results.config["test_1_population"],
        }
    )
    return df[["MIC", "cluster_labels", "Test Population 1"]].reset_index(drop=True)


def plot_MIC_cluster_distribution(df: pd.DataFrame, fname: str):
    plt.clf()
    plt.rcParams.update({"font.size": 12})
    g = sns.catplot(
        data=df,
        x="cluster_labels",
        y="MIC",
        hue=None,
        col="Test Population 1",
        row="Model",
        kind="box",
        sharex=False,
        color="white",
    )
    g.set_xticklabels([])
    g.set_ylabels("log2(MIC)")
    g.set_xlabels("Cluster")
    g.savefig(fname)


def main():
    result_files = [
        [
            "results/maela_updated_mic_rerun/DBSCAN/train_pop_cdc_results_no_inference_pbp_types.pkl",
            "DBSCAN",
        ],
        [
            "results/maela_updated_mic_rerun/DBSCAN/train_pop_cdc_results_no_inference_pbp_types(1).pkl",
            "DBSCAN",
        ],
        [
            "results/maela_updated_mic_rerun/DBSCAN_with_UMAP/train_pop_cdc_results_no_inference_pbp_types.pkl",
            "DBSCAN-UMAP",
        ],
        [
            "results/maela_updated_mic_rerun/DBSCAN_with_UMAP/train_pop_cdc_results_no_inference_pbp_types(1).pkl",
            "DBSCAN-UMAP",
        ],
    ]
    df = pd.concat([get_clusters(i[0]).assign(Model=i[1]) for i in result_files])
    df.loc[df["Test Population 1"] == "pmen", "Test Population 1"] = "PMEN"
    df.loc[df["Test Population 1"] == "maela", "Test Population 1"] = "Maela"
    plot_MIC_cluster_distribution(df, "temp.png")


if __name__ == "__main__":
    main()
