from __future__ import annotations

import warnings

import numpy as np

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio,
    equalized_odds_ratio,
)


from codecarbon import EmissionsTracker
from inner_loop.lcbench_model import LCBenchModel
from inner_loop.pytorch_model import PytorchModel
from inner_loop.sklearn_model import SklearnModel

from utils.input import ConfDict
from utils.output import adapt_to_mode


class MLP:
    def __init__(self, implementation="sklearn"):
        if implementation == "sklearn":
            self.implementation = SklearnModel()
        elif implementation == "pytorch":
            self.implementation = PytorchModel()
        elif implementation == "lcbench":
            self.implementation = LCBenchModel()
        else:
            raise Exception("Wrong implementation model keyword")

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.implementation.configspace

    def train(
        self,
        config: Configuration,
        seed: int = 0,
        budget: int = 52,
    ) -> dict[str, float]:
        return self.implementation.train(config, seed, budget)
