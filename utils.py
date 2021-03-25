import pandas as pd
import numpy as np
import nptyping
from typing import Dict, Any
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


def mean_acc_per_bin(
    predictions: nptyping.NDArray[float], labels: nptyping.NDArray[float]
) -> float:
    """
    Splits labels into bins of size = bin_size, and calculates the prediction
    accuracy in each bin.
    Returns the mean accuracy across all bins
    """
    assert len(predictions) == len(labels)

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


@dataclass(unsafe_hash=True)
class ResultsContainer:
    training_accuracy: float
    testing_accuracy: float
    validation_accuracy: float

    training_MSE: float
    testing_MSE: float
    validation_MSE: float

    training_predictions: nptyping.NDArray[nptyping.Float]
    testing_predictions: nptyping.NDArray[nptyping.Float]
    validation_predictions: nptyping.NDArray[nptyping.Float]

    hyperparameters: Dict[str, float]

    model_type: str

    model: Any
