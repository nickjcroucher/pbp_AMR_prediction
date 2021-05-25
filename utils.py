import datetime
from functools import lru_cache
from typing import Dict, Any

import pandas as pd
import numpy as np
import nptyping
from dataclasses import dataclass


def accuracy(
    predictions: nptyping.NDArray[nptyping.Float],
    labels: nptyping.NDArray[nptyping.Float],
) -> float:
    """
    Prediction accuracy defined as percentage of predictions within 1 twofold
    dilution of true value
    """
    diff = abs(predictions - labels)
    correct = diff[[i < 1 for i in diff]]
    return len(correct) / len(predictions) * 100


def bin_labels(
    labels: nptyping.NDArray[nptyping.Float],
) -> nptyping.NDArray[nptyping.Float]:

    # apply Freedman-Diaconis rule to get optimal bin size
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    IQR = np.subtract(*np.percentile(labels, [75, 25]))
    bin_size = 2 * IQR / (len(labels) ** (1 / 3))
    bin_size = int(
        np.ceil(bin_size)
    )  # round up cause if less than 1 will not work with accuracy function

    min_value = int(np.floor(min(labels)))
    max_value = int(np.floor(max(labels)))
    bins = list(range(min_value, max_value + bin_size, bin_size))
    binned_labels = np.digitize(labels, bins)

    median_bin_values = np.array(bins) + bin_size / 2

    return binned_labels, median_bin_values


def mean_acc_per_bin(
    predictions: nptyping.NDArray[nptyping.Float],
    labels: nptyping.NDArray[nptyping.Float],
) -> float:
    """
    Splits labels into bins of size = bin_size, and calculates the prediction
    accuracy in each bin.
    Returns the mean accuracy across all bins
    """
    assert len(predictions) == len(labels)

    binned_labels = bin_labels(labels)[0]

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

    return bin_accuracies.mean()


@lru_cache(maxsize=1)
def parse_blosum_matrix() -> Dict[str, Dict[str, int]]:
    # return as dict cause quicker to search
    df = pd.read_csv("blosum62.csv", index_col=0)
    return {i: df[i].to_dict() for i in df.columns}


def closest_blosum_sequence(
    pbp_data: pd.Series,
    pbp: str,
    training_sequence_array: nptyping.NDArray,
    blosum_scores: pd.DataFrame,
):
    """
    pbp: the pbp to match
    pbp_data: series with type and sequence of pbp not in training_data
    training_sequence_array: array of aligned sequences in the training data
    """
    pbp_seq = f"{pbp}_seq"
    pbp_type = f"{pbp}_type"

    pbp_sequence = pbp_data[pbp_seq]

    def check_amino_acid(AA, pos):
        """
        If amino acid AA at is not seen at position pos in the training data
        will return the closest which is based on the blosum matrix
        """
        # all amino acids at that position in the train set
        training_AAs = training_sequence_array[:, pos]

        if AA in training_AAs:
            return AA

        # if AA not seen at that position computes the blosum scores for all
        # that were seen and returns the AA with highest
        AA_blosum_comparisons = {}
        for i in np.unique(training_AAs):
            AA_blosum_comparisons[blosum_scores[AA][i]] = i

        # if multiple AAs had the highest score the one which is returned first
        # will be the one which it found last in the training set. This is
        # effectively random.
        return AA_blosum_comparisons[max(AA_blosum_comparisons.keys())]

    new_pbp_sequence = [
        check_amino_acid(j, i) for i, j in enumerate(pbp_sequence)
    ]
    new_pbp_sequence = "".join(new_pbp_sequence)  # type: ignore

    return pbp_data[pbp_type], new_pbp_sequence, "inferred_type"


@dataclass(unsafe_hash=True)
class ResultsContainer:
    training_accuracy: float
    testing_accuracy: float
    validation_accuracy: float

    training_MSE: float
    testing_MSE: float
    validation_MSE: float

    training_mean_acc_per_bin: float
    testing_mean_acc_per_bin: float
    validation_mean_acc_per_bin: float

    training_predictions: nptyping.NDArray[nptyping.Float]
    testing_predictions: nptyping.NDArray[nptyping.Float]
    validation_predictions: nptyping.NDArray[nptyping.Float]

    hyperparameters: Dict[str, float]

    model_type: str

    model: Any

    date_time = datetime.datetime.now()

    config: Dict

    def __repr__(self):
        return (
            f"model: {self.model_type}\n"
            + f"model_hyperparameters: {self.hyperparameters},\n"
            + "\n"
            + "ACCURACY\n"
            + f"Training Data Accuracy = {self.training_accuracy}\n"
            + f"Testing Data Accuracy = {self.testing_accuracy}\n"
            + f"Validation Data Accuracy = {self.validation_accuracy}\n"
            + "\n"
            + "MEAN ACCURACY PER BIN\n"
            + f"Training Data Mean Accuracy = {self.training_mean_acc_per_bin}\n"  # noqa: E501
            + f"Testing Data Mean Accuracy = {self.testing_mean_acc_per_bin}\n"
            + f"Validation Data Mean Accuracy = {self.validation_mean_acc_per_bin}\n"  # noqa: E501
            + "\n"
            + "MSE\n"
            + f"Training Data MSE = {self.training_MSE}\n"
            + f"Testing Data MSE = {self.testing_MSE}\n"
            + f"Validation Data MSE = {self.validation_MSE}\n"
        )
