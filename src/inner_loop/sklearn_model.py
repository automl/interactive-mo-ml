from __future__ import annotations


import numpy as np


from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

import sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from inner_loop.exec_model import ExecModel


from utils.input import ConfDict


class SklearnModel(ExecModel):
    def build_model(self, config, budget):
        numeric_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer()),
                ("scaler", MinMaxScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    [
                        idx
                        for idx, elem in enumerate(ConfDict()["categorical_indicator"])
                        if not elem
                    ],
                ),
                (
                    "cat",
                    categorical_transformer,
                    [
                        idx
                        for idx, elem in enumerate(ConfDict()["categorical_indicator"])
                        if elem
                    ],
                ),
            ]
        )

        classifier = MLPClassifier(
            hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
            solver=config["solver"],
            activation=config["activation"],
            learning_rate_init=config["learning_rate_init"],
            alpha=config["alpha"],
            max_iter=int(np.ceil(config["n_epochs"])),
            random_state=ConfDict()["seed"],
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def train_model(self, X_train, y_train):
        self.model = self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test):
        return self.model.predict(X_test)
