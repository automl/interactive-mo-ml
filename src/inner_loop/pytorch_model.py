from __future__ import annotations


import numpy as np


from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

import torch
import torch.nn as nn
from collections import OrderedDict

# from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from inner_loop.exec_model import ExecModel


from utils.input import ConfDict


class PytorchModel(ExecModel):
    def __init__(self) -> None:
        torch.manual_seed(ConfDict()["seed"])

    def build_model(self, config, budget):
        def get_activation_function(activation):
            if activation == "relu":
                return nn.ReLU()
            elif activation == "logistic":
                return nn.Sigmoid()
            elif activation == "tanh":
                return nn.Tanh()
            else:
                raise Exception("Wrong activation function keyword")

        def get_solver(solver, learning_rate_init, alpha):
            if solver == "adam":
                return torch.optim.Adam(
                    self.model.parameters(), lr=learning_rate_init, weight_decay=alpha
                )
            elif solver == "sgd":
                return torch.optim.SGD(
                    self.model.parameters(), lr=learning_rate_init, weight_decay=alpha
                )
            else:
                raise Exception("Wrong solver keyword")

        input_layer = [
            ("input", nn.Linear(ConfDict()["X"].shape[1], config["n_neurons"])),
            ("inact", get_activation_function(config["activation"])),
        ]
        hidden_layers = [
            [
                (f"hidden{i}", nn.Linear(config["n_neurons"], config["n_neurons"])),
                (f"hiddenact{i}", get_activation_function(config["activation"])),
            ]
            for i in range(config["n_layer"])
        ]
        output_layer = [
            ("output", nn.Linear(config["n_neurons"], len(np.unique(ConfDict()["y"])))),
            ("outact", nn.Softmax()),
        ]
        self.model = nn.Sequential(
            OrderedDict(
                input_layer
                + [item for layer in hidden_layers for item in layer]
                + output_layer
            )
        )
        self.solver = get_solver(
            config["solver"], config["learning_rate_init"], config["alpha"]
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.budget = config["n_epochs"]

    def train_model(self, X_train, y_train):
        self.model.train()

        X_train_normalized = MinMaxScaler().fit_transform(X_train)

        batches = list(
            zip(
                torch.split(torch.tensor(X_train_normalized), 200),
                torch.split(torch.tensor(y_train), 200),
            )
        )
        for epoch in range(self.budget):
            # enumerate mini batches
            for inputs, y_true in batches:
                # clear the gradients
                self.solver.zero_grad()
                # compute the model output
                y_pred = self.model(inputs)
                # calculate loss
                loss = self.loss_fn(
                    y_pred,
                    nn.functional.one_hot(
                        y_true, len(np.unique(ConfDict()["y"]))
                    ).float(),
                )
                # credit assignment
                loss.backward()
                # update model weights
                self.solver.step()

    def evaluate_model(self, X_test):
        X_test_normalized = MinMaxScaler().fit_transform(X_test)
        self.model.eval()
        return (
            torch.argmax(self.model(torch.tensor(X_test_normalized)), 1)
            .detach()
            .numpy()
        )
