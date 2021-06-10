import datetime
from functools import lru_cache
from typing import Any, Dict, Set, Tuple

import pandas as pd
import numpy as np
import nptyping
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from data_preprocessing.parse_pbp_data import (
    parse_cdc,
    parse_pmen_and_maela,
    standardise_MICs,
)
from models.HMM_model import ProfileHMMPredictor


def _filter_data(data, train_types, pbp_type, invert=False):
    """
    filters data by pbp types which appear in training data
    """
    inc_types = set(data[pbp_type])
    inc_types = filter(lambda x: x in train_types, inc_types)  # type: ignore
    if invert:
        return data.loc[~data[pbp_type].isin(list(inc_types))]
    else:
        return data.loc[data[pbp_type].isin(list(inc_types))]


@lru_cache(maxsize=1)
def parse_blosum_matrix() -> Dict[str, Dict[str, int]]:
    # return as dict cause quicker to search
    df = pd.read_csv("blosum62.csv", index_col=0)
    return {i: df[i].to_dict() for i in df.columns}


def closest_sequence(
    pbp_data: pd.Series,
    pbp: str,
    training_sequence_array: nptyping.NDArray,
    method: str,
    blosum_scores: Dict = None,
    hmm_scores: Dict = None,
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

        AA_comparisons = {}
        if method == "blosum":
            # if AA not seen at that position computes the blosum scores for
            # all that were seen and returns the AA with highest
            for i in np.unique(training_AAs):
                AA_comparisons[blosum_scores[AA][i]] = i

        elif method == "hmm":
            for i in np.unique(training_AAs):
                AA_comparisons[hmm_scores[pos][i]] = i
        else:
            raise ValueError(f"Unknown inference method: {method}")

        # if multiple AAs had the highest score the one which is returned first
        # will be the one which it found last in the training set. This is
        # effectively random.
        return AA_comparisons[max(AA_comparisons.keys())]

    new_pbp_sequence = [
        check_amino_acid(j, i) for i, j in enumerate(pbp_sequence)
    ]
    new_pbp_sequence = "".join(new_pbp_sequence)  # type: ignore

    return pbp_data[pbp_type], new_pbp_sequence, "inferred_type"


def HMM_position_scores(training_data: pd.DataFrame, pbp_seq: str) -> Dict:
    hmm_predictor = ProfileHMMPredictor(
        training_data, [pbp_seq], HMM_per_phenotype=False
    )
    alphabet = hmm_predictor.alphabet.symbols

    # convert to list from pyhmmer object and remove entry probabilities
    match_emissions = [list(i) for i in hmm_predictor.hmm.match_emissions][1:]

    # nested dictionary with emission prob of each amino acid at each position
    # in the sequence
    return {
        pos: {alphabet[n]: prob for n, prob in enumerate(emissions)}
        for pos, emissions in enumerate(match_emissions)
    }


def infer_sequences(
    pbp_type: str,
    pbp: str,
    train_types: Set,
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    method: str,
) -> pd.DataFrame:

    pbp_seq = f"{pbp}_seq"

    missing_types_and_sequences = _filter_data(
        testing_data, train_types, pbp_type, invert=True
    )[[pbp_type, pbp_seq]].drop_duplicates()

    training_types_and_sequences = training_data[
        [pbp_type, pbp_seq]
    ].drop_duplicates()

    training_sequence_array = np.vstack(
        training_types_and_sequences[pbp_seq].apply(
            lambda x: np.array(list(x))
        )
    )  # stack sequences in the training data as array of characters

    if method == "blosum":
        inferred_sequences = missing_types_and_sequences.apply(
            closest_sequence,
            axis=1,
            pbp=pbp,
            training_sequence_array=training_sequence_array,
            method=method,
            blosum_scores=parse_blosum_matrix(),
        )

    elif method == "hmm":
        inferred_sequences = missing_types_and_sequences.apply(
            closest_sequence,
            axis=1,
            pbp=pbp,
            training_sequence_array=training_sequence_array,
            method=method,
            hmm_scores=HMM_position_scores(training_data, pbp_seq),
        )

    inferred_sequences = inferred_sequences.apply(pd.Series)
    inferred_sequences.rename(
        columns={
            0: "original_type",
            1: "inferred_seq",
            2: "inferred_type",
        },
        inplace=True,
    )

    testing_data = testing_data.merge(
        inferred_sequences,
        left_on=pbp_type,
        right_on="original_type",
        how="left",
    )
    testing_data[pbp_type].mask(
        ~testing_data.inferred_type.isna(),
        testing_data.inferred_type,
        inplace=True,
    )
    testing_data[pbp_seq].mask(
        ~testing_data.inferred_seq.isna(),
        testing_data.inferred_seq,
        inplace=True,
    )

    return testing_data[training_data.columns]


def perform_HMM_inference(
    train: pd.DataFrame,
    test_1: pd.DataFrame,
    test_2: pd.DataFrame,
    val: pd.DataFrame,
) -> Tuple:

    pbp_seqs = ["a1_seq", "b2_seq", "x2_seq"]

    # one model trained on each pbp type
    hmm_predictors = {
        pbp: ProfileHMMPredictor(train, pbp_seqs=[pbp]) for pbp in pbp_seqs
    }

    for pbp in pbp_seqs:
        test_1[pbp] = hmm_predictors[pbp].closest_HMM_sequence(test_1[pbp])
        test_2[pbp] = hmm_predictors[pbp].closest_HMM_sequence(test_2[pbp])
        val[pbp] = hmm_predictors[pbp].closest_HMM_sequence(val[pbp])

    return test_1, test_2, val


def load_data(
    train_data_population,
    test_data_population_1,
    test_data_population_2,
    blosum_inference: bool = False,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = True,
    standardise_training_MIC: bool = False,
    standardise_test_and_val_MIC: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if sorted(
        [train_data_population, test_data_population_1, test_data_population_2]
    ) != ["cdc", "maela", "pmen"]:
        raise ValueError(
            "train_data_population, test_data_population_1, and test_data_\
population_2 should be unique and should be either of cdc, maela, or pmen"
        )

    if (
        len([i for i in [blosum_inference, HMM_inference, filter_unseen] if i])
        > 1
    ):
        raise ValueError(
            "At most one of blosum inference, HMM_inference, or filter_unseen can be true"  # noqa: E501
        )
    pbp_patterns = ["a1", "b2", "x2"]

    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
    maela = pd.read_csv("../data/pneumo_pbp/maela_aa_df.csv")

    cdc = parse_cdc(cdc, pbp_patterns)
    pmen = parse_pmen_and_maela(pmen, cdc, pbp_patterns)
    maela = parse_pmen_and_maela(maela, cdc, pbp_patterns)

    if train_data_population == "cdc":
        train, val = train_test_split(cdc, test_size=0.33, random_state=0)
        if test_data_population_1 == "pmen":
            test_1 = pmen
            test_2 = maela
        else:
            test_1 = maela
            test_2 = pmen
    elif train_data_population == "pmen":
        train, val = train_test_split(pmen, test_size=0.33, random_state=0)
        if test_data_population_1 == "cdc":
            test_1 = cdc
            test_2 = maela
        else:
            test_1 = maela
            test_2 = cdc
    elif train_data_population == "maela":
        train, val = train_test_split(maela, test_size=0.33, random_state=0)
        if test_data_population_1 == "cdc":
            test_1 = cdc
            test_2 = pmen
        else:
            test_1 = pmen
            test_2 = cdc
    else:
        raise ValueError(f"train_data_population = {train_data_population}")

    if standardise_training_MIC:
        train = standardise_MICs(train)
    if standardise_test_and_val_MIC:
        test_1 = standardise_MICs(test_1)
        test_2 = standardise_MICs(test_2)
        val = standardise_MICs(val)

    for pbp in pbp_patterns:
        pbp_type = f"{pbp}_type"
        train_types = set(train[pbp_type])

        # get closest type to all missing in the training data
        if blosum_inference:
            test_1 = infer_sequences(
                pbp_type, pbp, train_types, train, test_1, method="blosum"
            )
            test_2 = infer_sequences(
                pbp_type, pbp, train_types, train, test_2, method="blosum"
            )
            val = infer_sequences(
                pbp_type, pbp, train_types, train, val, method="blosum"
            )

        elif HMM_inference:
            test_1 = infer_sequences(
                pbp_type, pbp, train_types, train, test_1, method="hmm"
            )
            test_2 = infer_sequences(
                pbp_type, pbp, train_types, train, test_2, method="hmm"
            )
            val = infer_sequences(
                pbp_type, pbp, train_types, train, val, method="hmm"
            )

        # filter out everything which isnt in the training data
        elif filter_unseen:
            test_1 = _filter_data(test_1, train_types, pbp_type)
            test_2 = _filter_data(test_2, train_types, pbp_type)
            val = _filter_data(val, train_types, pbp_type)

    # replace each sequence in test sets with consensus sequence from HMM
    # fitted to training sequences of each MIC
    if HMM_MIC_inference:
        test_1, test_2, val = perform_HMM_inference(train, test_1, test_2, val)

    return train, test_1, test_2, val


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
    bin_size = (
        bin_size if bin_size >= 1 else 1
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

    # if there is only one label then mean_acc_per_bin is the same as accuracy
    if np.unique(labels).shape[0] == 1:
        return accuracy(predictions, labels)

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


@dataclass(unsafe_hash=True)
class ResultsContainer:
    training_accuracy: float
    validation_accuracy: float
    testing_accuracy_1: float
    testing_accuracy_2: float

    training_MSE: float
    validation_MSE: float
    testing_MSE_1: float
    testing_MSE_2: float

    training_mean_acc_per_bin: float
    validation_mean_acc_per_bin: float
    testing_mean_acc_per_bin_1: float
    testing_mean_acc_per_bin_2: float

    training_predictions: nptyping.NDArray[nptyping.Float]
    validation_predictions: nptyping.NDArray[nptyping.Float]
    testing_predictions_1: nptyping.NDArray[nptyping.Float]
    testing_predictions_2: nptyping.NDArray[nptyping.Float]

    hyperparameters: Dict[str, float]

    model_type: str

    model: Any

    config: Dict

    date_time = datetime.datetime.now()

    def __repr__(self):
        return (
            f"model: {self.model_type}\n"
            + f"model_hyperparameters: {self.hyperparameters},\n"
            + f"model fit config: {self.config}, \n"
            + "\n"
            + "ACCURACY\n"
            + f"Training Data Accuracy = {self.training_accuracy}\n"
            + f"Validation Data Accuracy = {self.validation_accuracy}\n"
            + f"Testing Data 1 Accuracy = {self.testing_accuracy_1}\n"
            + f"Testing Data 2 Accuracy = {self.testing_accuracy_2}\n"
            + "\n"
            + "MEAN ACCURACY PER BIN\n"
            + f"Training Data Mean Accuracy = {self.training_mean_acc_per_bin}\n"  # noqa: E501
            + f"Validation Data Mean Accuracy = {self.validation_mean_acc_per_bin}\n"  # noqa: E501
            + f"Testing Data 1 Mean Accuracy = {self.testing_mean_acc_per_bin_1}\n"  # noqa: E501
            + f"Testing Data 2 Mean Accuracy = {self.testing_mean_acc_per_bin_2}\n"  # noqa: E501
            + "\n"
            + "MSE\n"
            + f"Training Data MSE = {self.training_MSE}\n"
            + f"Validation Data MSE = {self.validation_MSE}\n"
            + f"Testing Data 1 MSE = {self.testing_MSE_1}\n"
            + f"Testing Data 2 MSE = {self.testing_MSE_2}\n"
        )
