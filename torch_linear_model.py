import logging
import time
import os
import json
from functools import partial
from typing import Callable, Tuple, Union

import torch
import torch.optim as optim
from torch import nn
from bayes_opt import BayesianOptimization

from linear_models import load_data
from utils import accuracy
from torch_model_utils import (
    DataGenerator,
    MetricAccumulator,
    format_data,
    write_epoch_results,
)

logging.basicConfig()
logging.root.setLevel(logging.INFO)


class LinearRegressor(nn.Module):
    def __init__(self, input_dimensions: int):
        super(LinearRegressor, self).__init__()
        self.layer1 = nn.Linear(input_dimensions, 1).double()

    def forward(self, model_input):
        return self.layer1(model_input)


def epoch_(model: LinearRegressor, data: DataGenerator) -> torch.Tensor:

    data.reset_generator()

    outputs = []
    for i in range(data.n_samples):
        features = data.next_sample()[0]
        outputs.append(model(features))

    return torch.cat(outputs, dim=0)


def test(
    data: DataGenerator,
    model: LinearRegressor,
    loss_function: Callable,
    accuracy: Callable,
) -> Tuple[float, float]:

    data.reset_generator()
    model.train(False)

    with torch.no_grad():
        output = epoch_(model, data)
    loss = float(loss_function(output, data.labels))
    acc = accuracy(output, data.labels)

    return loss, acc


def train(
    data: DataGenerator,
    model: LinearRegressor,
    optimizer,
    epoch: int,
    loss_function: Callable,
    accuracy: Callable,
    l1_alpha: float = None,
    testing_data: DataGenerator = None,
    validation_data: DataGenerator = None,
    verbose: bool = True,
) -> Tuple[
    LinearRegressor,
    Tuple[
        float,
        float,
        Union[float, str],
        Union[float, str],
        Union[float, str],
        Union[float, str],
    ],
]:
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = epoch_(model, data)
    loss_train = loss_function(output, data.labels)
    if l1_alpha is not None:  # apply l1 regularisation
        L1_reg = torch.tensor(  # pylint: disable=not-callable
            0.0, requires_grad=True
        )
        for name, param in model.named_parameters():
            if name.endswith("weight"):
                L1_reg = L1_reg + torch.norm(param, 1)
        regularised_loss_train = loss_train + l1_alpha * L1_reg
    else:
        regularised_loss_train = loss_train

    acc_train = accuracy(output, data.labels)

    if testing_data:
        loss_test, acc_test = test(
            testing_data, model, loss_function, accuracy
        )
    else:
        loss_test = "N/A"  # type: ignore
        acc_test = "N/A"  # type: ignore
    if validation_data:
        loss_val, acc_val = test(
            validation_data, model, loss_function, accuracy
        )
    else:
        loss_val = "N/A"  # type: ignore
        acc_val = "N/A"  # type: ignore

    regularised_loss_train.backward()
    optimizer.step()
    loss_train = float(loss_train)  # to write it to file

    if verbose:
        logging.info(
            f"Epoch {epoch} complete\n"
            + f"\tTime taken = {time.time() - t}\n"
            + f"\tTraining Data Loss = {loss_train}\n"
            + f"\tTraining Data Accuracy = {acc_train}\n"
            f"\tTesting Data Loss = {loss_test}\n"
            + f"\tTesting Data Accuracy = {acc_test}\n"
            f"\tValidation Data Loss = {loss_val}\n"
            + f"\tValidation Data Accuracy = {acc_val}\n"
        )

    return model, (
        loss_train,
        acc_train,
        loss_test,
        acc_test,
        loss_val,
        acc_val,
    )


def fit_model(
    return_model: bool,
    log_outputs: bool,
    verbose: bool,
    l2_alpha: float,
    l1_alpha: float = None,
    lr: float = 0.001,
    stopping_criterion: str = "",
):

    if verbose:
        logging.info("Loading data")

    training, testing, validation = load_data(
        adj_convolution=False, laplacian_convolution=False, interactions=False
    )
    training_data, testing_data, validation_data = format_data(
        training, testing, validation
    )

    if verbose:
        logging.info("Initialising Model")

    torch.manual_seed(0)
    model = LinearRegressor(training_data.n_features)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_alpha)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", cooldown=5, verbose=verbose
    )
    loss_function = nn.MSELoss()

    # records training metrics and logs the gradient after each epoch
    training_metrics = MetricAccumulator()
    n = 1
    summary_file = f"torch_model_fitting_metrics_{n}.tsv"
    while os.path.isfile(summary_file):
        summary_file = f"torch_model_fitting_metrics_{n}.tsv"
        n += 1
    config_file = f"torch_model_{n}_config.txt"
    if log_outputs:
        with open(config_file, "w") as a:
            a.write(
                json.dumps(
                    {"lr": lr, "l1_alpha": l1_alpha, "l2_alpha": l2_alpha}
                )
            )

    start_time = time.time()
    for epoch in range(300):
        epoch += 1

        model, epoch_results = train(
            training_data,
            model,
            optimizer,
            epoch,
            loss_function,
            accuracy,
            l1_alpha=l1_alpha,
            testing_data=testing_data,
            validation_data=validation_data,
            verbose=verbose,
        )

        training_metrics.add(epoch_results)
        if log_outputs:
            write_epoch_results(epoch, epoch_results, summary_file)

        if stopping_criterion == "testing_accuracy" and (
            all(
                [
                    i < 0.1
                    for i in training_metrics.testing_data_acc_grads[-20:]
                ]
            )
            and epoch > 50
        ):

            logging.info(
                "Gradient of testing data accuracy appears to have plateaued,"
                + " terminating early"
            )
            break

        scheduler.step(epoch_results[2])  # testing data loss

    if verbose:
        logging.info(
            f"Model Fitting Complete. Time elapsed {time.time() - start_time}"
        )

    if return_model:
        return model
    else:
        return -(
            min(training_metrics.validation_data_loss)
        )  # return lowest validation data MSE encountered in training


def main():
    partial_fitting_function = partial(
        fit_model, return_model=False, log_outputs=False, verbose=False
    )

    pbounds = {"l1_alpha": [0.01, 0.5], "l2_alpha": [0.01, 0.5]}

    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(n_iter=20)

    _ = fit_model(
        return_model=True,
        log_outputs=True,
        verbose=True,
        **optimizer.max["params"],
    )
