import logging
import os
from typing import List, Tuple
from random import shuffle

import torch
from pandas import Series
from scipy.sparse import csr_matrix


class DataGenerator:
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        auto_reset_generator=True,
    ):
        assert len(features) == len(
            labels
        ), "Features and labels are of different length"

        self.features = features
        self.labels = labels
        self.n_features = self.features[0].shape[0]
        self.n_samples = len(labels)
        self.n = 0
        self._index = list(range(self.n_samples))
        self.auto_reset_generator = auto_reset_generator

    def _parse_features(self, features: torch.Tensor) -> List:
        features_list = []
        for i in range(len(features)):
            features_list.append(features[i].unsqueeze(1))
        return features_list

    def _iterate(self):
        self.n += 1
        if self.n == self.n_samples and self.auto_reset_generator:
            sample = self.features[self.n - 1], self.labels[self.n - 1]
            self.reset_generator()
            yield sample
        else:
            yield self.features[self.n - 1], self.labels[self.n - 1]

    def next_sample(self):
        return next(self._iterate())

    def reset_generator(self):
        self.n = 0

    def shuffle_samples(self):
        shuffle(self._index)
        self.labels = torch.as_tensor(
            list({i: self.labels[i] for i in self._index}.values())
        )
        self.features = list(
            {i: self.features[i] for i in self._index}.values()
        )


class MetricAccumulator:
    def __init__(self, gradient_batch=10):
        self.training_data_loss = []
        self.training_data_acc = []
        self.testing_data_loss = []
        self.testing_data_acc = []
        self.validation_data_loss = []
        self.validation_data_acc = []
        self.training_data_loss_grads = []
        self.training_data_acc_grads = []
        self.testing_data_loss_grads = []
        self.testing_data_acc_grads = []
        self.validation_data_loss_grads = []
        self.validation_data_acc_grads = []
        self.gradient_batch = gradient_batch

    def add(self, epoch_results):
        self.training_data_loss.append(epoch_results[0])
        self.training_data_acc.append(epoch_results[1])
        self.testing_data_loss.append(epoch_results[2])
        self.testing_data_acc.append(epoch_results[3])
        self.validation_data_loss.append(epoch_results[4])
        self.validation_data_acc.append(epoch_results[5])
        self._all_grads()

    def _all_grads(self):
        self.training_data_loss_grads.append(
            self.metric_gradient(self.training_data_loss)
        )
        self.training_data_acc_grads.append(
            self.metric_gradient(self.training_data_acc)
        )
        self.testing_data_loss_grads.append(
            self.metric_gradient(self.testing_data_loss)
        )
        self.testing_data_acc_grads.append(
            self.metric_gradient(self.testing_data_acc)
        )
        self.validation_data_loss_grads.append(
            self.metric_gradient(self.validation_data_loss)
        )
        self.validation_data_acc_grads.append(
            self.metric_gradient(self.validation_data_acc)
        )

    def metric_gradient(self, x: list):
        batch = self.gradient_batch
        data = x[-batch:]
        return round((data[-1] - data[0]) / len(data), 2)

    def avg_gradient(self, x: list):
        batch = self.gradient_batch
        data = x[-batch:]
        return sum(data) / len(data)

    def log_gradients(self, epoch):
        avg_grads = [
            self.avg_gradient(self.training_data_loss_grads),
            self.avg_gradient(self.training_data_acc_grads),
            self.avg_gradient(self.testing_data_loss_grads),
            self.avg_gradient(self.testing_data_acc_grads),
            self.avg_gradient(self.validation_data_loss_grads),
            self.avg_gradient(self.validation_data_acc_grads),
        ]
        last_epoch = max(0, epoch - self.gradient_batch)
        logging.info(
            f"Average Gradient Between Epoch {epoch} and {last_epoch}:\n \
            Training Data Loss Gradient = {avg_grads[0]}\n \
            Training Data Accuracy Gradient = {avg_grads[1]}\n \
            Testing Data Loss Gradient = {avg_grads[2]}\n \
            Testing Data Accuracy Gradient = {avg_grads[3]}\n \
            Validation Data Loss Gradient = {avg_grads[4]}\n \
            Validation Data Accuracy Gradient = {avg_grads[5]}\n"
        )


def data_to_tensor(data: csr_matrix) -> torch.Tensor:
    return torch.as_tensor(data.todense(), dtype=torch.float64)


def format_data(
    train: Tuple[csr_matrix, Series],
    test: Tuple[csr_matrix, Series],
    validate: Tuple[csr_matrix, Series],
) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
    train_generator = DataGenerator(
        data_to_tensor(train[0]), torch.as_tensor(train[1].values)
    )
    test_generator = DataGenerator(
        data_to_tensor(test[0]), torch.as_tensor(test[1].values)
    )
    validate_generator = DataGenerator(
        data_to_tensor(validate[0]), torch.as_tensor(validate[1].values)
    )

    return train_generator, test_generator, validate_generator


def write_epoch_results(epoch, epoch_results, summary_file):
    if not os.path.isfile(summary_file):
        with open(summary_file, "w") as a:
            a.write(
                "epoch\ttraining_data_loss\ttraining_data_acc\ttesting_data_loss\ttesting_data_acc\tvalidation_data_loss\tvalidation_data_acc\n"  # noqa: E501
            )

    with open(summary_file, "a") as a:
        line = str(epoch) + "\t" + "\t".join([str(i) for i in epoch_results])
        a.write(line + "\n")
