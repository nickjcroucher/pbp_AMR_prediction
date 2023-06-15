import datetime
import math
from functools import lru_cache
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import nptyping
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from data_preprocessing.parse_pbp_data import (
    parse_cdc,
    parse_pmen_and_maela,
    parse_extended_sequences,
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
    hmm_predictor: ProfileHMMPredictor = None,
    blosum_strictly_non_negative: bool = False,
):
    """
    pbp: the pbp to match
    pbp_data: series with type and sequence of pbp not in training_data
    training_sequence_array: array of aligned sequences in the training data
    """
    pbp_seq = f"{pbp}_seq"
    pbp_type = f"{pbp}_type"

    pbp_sequence = pbp_data[pbp_seq]

    if method == "hmm_mic":
        closest_hmm = hmm_predictor.closest_HMM([pbp_sequence])[  # type: ignore # noqa: E501
            0
        ]
        match_emissions = [list(i) for i in closest_hmm.match_emissions][1:]
        closest_hmm_scores = {
            pos: {
                closest_hmm.alphabet.symbols[n]: prob
                for n, prob in enumerate(emissions)
            }
            for pos, emissions in enumerate(match_emissions)
        }

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

        elif method == "hmm_mic":
            hmm_scores
            for i in np.unique(training_AAs):
                AA_comparisons[closest_hmm_scores[pos][i]] = i

        else:
            raise ValueError(f"Unknown inference method: {method}")

        closest_AA = AA_comparisons[max(AA_comparisons.keys())]
        if (
            method == "blosum"
            and blosum_strictly_non_negative
            and blosum_scores[AA][closest_AA] < 0
        ):
            return AA

        # if multiple AAs had the highest score the one which is returned first
        # will be the one which it found last in the training set. This is
        # effectively random.
        return closest_AA

    new_pbp_sequence = [check_amino_acid(j, i) for i, j in enumerate(pbp_sequence)]
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
    method: str,
    training_data: pd.DataFrame,
    blosum_strictly_non_negative: bool = False,
    **all_testing_data: Dict[str, pd.DataFrame],
) -> Dict:

    pbp_seq = f"{pbp}_seq"

    if method == "blosum":
        blosum_scores = parse_blosum_matrix()
    elif method == "hmm":
        hmm_scores = HMM_position_scores(training_data, pbp_seq)
    elif method == "hmm_mic":
        hmm_predictor = ProfileHMMPredictor(training_data, [pbp_seq])

    for name, testing_data in all_testing_data.items():
        missing_types_and_sequences = _filter_data(
            testing_data, train_types, pbp_type, invert=True
        )[[pbp_type, pbp_seq]].drop_duplicates()

        training_types_and_sequences = training_data[
            [pbp_type, pbp_seq]
        ].drop_duplicates()

        training_sequence_array = np.vstack(
            training_types_and_sequences[pbp_seq].apply(lambda x: np.array(list(x)))
        )  # stack sequences in the training data as array of characters

        if method == "blosum":
            inferred_sequences = missing_types_and_sequences.apply(
                closest_sequence,
                axis=1,
                pbp=pbp,
                training_sequence_array=training_sequence_array,
                method=method,
                blosum_scores=blosum_scores,
                blosum_strictly_non_negative=blosum_strictly_non_negative,
            )

        elif method == "hmm":
            inferred_sequences = missing_types_and_sequences.apply(
                closest_sequence,
                axis=1,
                pbp=pbp,
                training_sequence_array=training_sequence_array,
                method=method,
                hmm_scores=hmm_scores,
            )

        elif method == "hmm_mic":
            inferred_sequences = missing_types_and_sequences.apply(
                closest_sequence,
                axis=1,
                pbp=pbp,
                training_sequence_array=training_sequence_array,
                method=method,
                hmm_predictor=hmm_predictor,
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

        all_testing_data[name] = testing_data[training_data.columns]

    return all_testing_data


def _data_processing(
    pbp_patterns: List[str],
    standardise_training_MIC: bool,
    standardise_test_and_val_MIC: bool,
    blosum_inference: bool,
    HMM_inference: bool,
    HMM_MIC_inference: bool,
    filter_unseen: bool,
    train: pd.DataFrame,
    blosum_strictly_non_negative: bool = False,
    **test_datasets: Dict[str, pd.DataFrame],
):
    if standardise_training_MIC:
        train = standardise_MICs(train)
    if standardise_test_and_val_MIC:
        test_datasets = {k: standardise_MICs(v) for k, v in test_datasets.items()}

    for pbp in pbp_patterns:

        pbp_type = f"{pbp}_type"
        train_types = set(train[pbp_type])

        # get closest type to all missing in the training data using blosum score # noqa: E501
        if blosum_inference:
            test_datasets = infer_sequences(
                pbp_type,
                pbp,
                train_types,
                method="blosum",
                blosum_strictly_non_negative=blosum_strictly_non_negative,
                training_data=train,
                **test_datasets,
            )
        # get closest type to all missing in the training data using hmm
        # trained on all the training data
        elif HMM_inference:
            test_datasets = infer_sequences(
                pbp_type,
                pbp,
                train_types,
                method="hmm",
                training_data=train,
                blosum_strictly_non_negative=blosum_strictly_non_negative,
                **test_datasets,
            )
        # get closest type to all missing in the training data using closest
        # hmm of hmms trained on each MIC in the training data
        elif HMM_MIC_inference:
            test_datasets = infer_sequences(
                pbp_type,
                pbp,
                train_types,
                method="hmm_mic",
                training_data=train,
                blosum_strictly_non_negative=blosum_strictly_non_negative,
                **test_datasets,
            )

        # filter out everything which isnt in the training data
        elif filter_unseen:
            test_datasets = {
                k: _filter_data(v, train_types, pbp_type)
                for k, v in test_datasets.items()
            }

    return tuple([train] + list(test_datasets.values()))


def load_data(
    train_data_population: str,
    test_data_population_1: str,
    test_data_population_2: str,
    blosum_inference: bool = False,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = True,
    standardise_training_MIC: bool = False,
    standardise_test_and_val_MIC: bool = False,
    blosum_strictly_non_negative: bool = False,
    maela_correction: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if sorted(
        [train_data_population, test_data_population_1, test_data_population_2]
    ) != ["cdc", "maela", "pmen"]:
        raise ValueError(
            "train_data_population, test_data_population_1, and test_data_\
population_2 should be unique and should be either of cdc, maela, or pmen"
        )

    if len([i for i in [blosum_inference, HMM_inference, filter_unseen] if i]) > 1:
        raise ValueError(
            "At most one of blosum inference, HMM_inference, or filter_unseen can be true"  # noqa: E501
        )
    pbp_patterns = ["a1", "b2", "x2"]

    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
    maela = pd.read_csv("../data/pneumo_pbp/maela_aa_df.csv")

    if maela_correction:
        maela.loc[maela.mic == 0.060, 'mic'] = 0.03

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

    # will throw error due to scikit-learn bug when changing columns in
    # original val object (https://stackoverflow.com/questions/45090639/pandas-shows-settingwithcopywarning-after-train-test-split) # noqa: E501
    val = val.copy(deep=False)
    train = train.copy(deep=False)

    return _data_processing(
        pbp_patterns,
        standardise_training_MIC,
        standardise_test_and_val_MIC,
        blosum_inference,
        HMM_inference,
        HMM_MIC_inference,
        filter_unseen,
        blosum_strictly_non_negative=blosum_strictly_non_negative,
        train=train,
        test_1=test_1,
        test_2=test_2,
        val=val,
    )


def load_extended_sequence_data(
    train_data_population: str,
    blosum_inference: bool = False,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = True,
    standardise_training_MIC: bool = False,
    standardise_test_and_val_MIC: bool = False,
    blosum_strictly_non_negative: bool = False,
    maela_correction: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if len([i for i in [blosum_inference, HMM_inference, filter_unseen] if i]) > 1:
        raise ValueError(
            "At most one of blosum inference, HMM_inference, or filter_unseen can be true"  # noqa: E501
        )
    pbp_patterns = ["a1", "b2", "x2"]

    pmen = pd.read_csv("../data/pneumo_pbp/pmen_full_pbp_seqs_mic.csv")
    maela = pd.read_csv("../data/pneumo_pbp/maela_full_pbp_mic.csv")

    if maela_correction:
        maela.loc[maela.mic == 0.060, 'mic'] = 0.03

    if train_data_population == "maela":
        maela = parse_extended_sequences(maela, pbp_patterns)
        test = parse_pmen_and_maela(pmen, maela, pbp_patterns)
        train, val = train_test_split(maela, test_size=0.33, random_state=0)
    elif train_data_population == "pmen":
        pmen = parse_extended_sequences(pmen, pbp_patterns)
        test = parse_pmen_and_maela(maela, pmen, pbp_patterns)
        train, val = train_test_split(pmen, test_size=0.33, random_state=0)
    else:
        raise ValueError(f"train_data_population = {train_data_population}")

    # will throw error due to scikit-learn bug when changing columns in
    # original val object (https://stackoverflow.com/questions/45090639/pandas-shows-settingwithcopywarning-after-train-test-split) # noqa: E501
    val = val.copy(deep=False)
    train = train.copy(deep=False)

    return _data_processing(
        pbp_patterns,
        standardise_training_MIC,
        standardise_test_and_val_MIC,
        blosum_inference,
        HMM_inference,
        HMM_MIC_inference,
        filter_unseen,
        blosum_strictly_non_negative=blosum_strictly_non_negative,
        train=train,
        test_1=test,
        val=val,
    )


def accuracy(
    predictions: nptyping.NDArray[Any,nptyping.Float],
    labels: nptyping.NDArray[Any,nptyping.Float]
) -> float:
    """
    Prediction accuracy defined as percentage of predictions within 1 twofold
    dilution of true value
    """
    diff = abs(predictions - labels)
    correct = diff[[i < 1 for i in diff]]
    return len(correct) / len(predictions) * 100


def bin_labels(
    labels: nptyping.NDArray[Any,nptyping.Float],
) -> nptyping.NDArray[Any,nptyping.Float]:

    # apply Freedman-Diaconis rule to get optimal bin size
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    IQR = np.subtract(*np.percentile(labels, [75, 25]))
    bin_size = 2 * IQR / (len(labels) ** (1 / 3))
    bin_size = (
        bin_size if bin_size >= 1 else 1
    )  # round up cause if less than 1 will not work with accuracy function
    bin_size = round(bin_size)  # ensure is integer

    min_value = int(np.floor(min(labels)))
    max_value = int(np.floor(max(labels)))
    bins = list(range(min_value, max_value + bin_size, bin_size))
    binned_labels = np.digitize(labels, bins)

    median_bin_values = np.array(bins) + bin_size / 2

    return binned_labels, median_bin_values


def mean_acc_per_bin(
    predictions: nptyping.NDArray[Any,nptyping.Float],
    labels: nptyping.NDArray[Any,nptyping.Float],
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

    training_predictions: nptyping.NDArray[Any,nptyping.Float]
    validation_predictions: nptyping.NDArray[Any,nptyping.Float]
    testing_predictions_1: nptyping.NDArray[Any,nptyping.Float]
    testing_predictions_2: nptyping.NDArray[Any,nptyping.Float]

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


def ordinal_regression_format(data: Dict, minor_variant_frequency: float) -> Dict:
    x = np.array(data["train"][0].todense())
    y = data["train"][1].apply(math.floor)
    y_adjusted = y + abs(y.min())  # modelling is easier if all >= 0

    lower_freq_threshold = round(minor_variant_frequency * len(x))
    upper_freq_threshold = len(x) - lower_freq_threshold

    y_train = pd.Series(
        pd.Categorical(y_adjusted, sorted(y_adjusted.unique()), ordered=True)
    )
    data["train"][1] = y_train

    idx = (lower_freq_threshold <= np.count_nonzero(x, axis=0)) & (
        upper_freq_threshold >= np.count_nonzero(x, axis=0)
    )
    x_train = x[:, idx]
    data["train"][0] = x_train

    for i in ["val", "test_1", "test_2"]:
        x = np.array(data[i][0].todense())
        y_ = data[i][1].apply(math.floor)
        training_phenotype_idx = y_.isin(y)
        y_ = y_.loc[training_phenotype_idx]  # remove phenotypes not in training set
        x = x[training_phenotype_idx, :]
        y_ = y_ + abs(y_.min())

        y = pd.Series(pd.Categorical(y_, sorted(y_train.unique()), ordered=True))

        data[i][0] = x[:, idx]
        data[i][1] = y[np.isin(y, y_train)]

    return data
