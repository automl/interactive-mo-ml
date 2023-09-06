from __future__ import annotations
import copy
import os

import warnings
import random

import numpy as np

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

from ConfigSpace import Configuration

from smac.facade.abstract_facade import AbstractFacade

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from inner_loop.pareto_mlp import ParetoMLP
from my_model.my_grid_model import MyGridModel
from ranker.my_rank_svc import MyRankSVM
from utils.input import ConfDict
from utils.output import (
    adapt_paretos,
    check_preferences,
    load_json_file,
    update_config,
    adapt_encoded,
)
from utils.pareto import encode_pareto, get_pareto_indicators
from utils.preference_learning import create_preference_dataset

from utils.sample import grid_search

from inner_loop.mlp import MLP


class UtilityParetoMLP(ParetoMLP):
    def __init__(
        self,
        implementation="sklearn",
        main_indicator="hv",
        mode="preferences",
        preference_budget=None,
    ):
        super().__init__(implementation)

        preference_path = os.path.join(
            os.path.dirname(ConfDict()["output_folder"]), "preliminar_sampling"
        )
        if check_preferences(os.path.join(preference_path, "preferences.csv")):
            config_dict = load_json_file(
                os.path.join(
                    "/",
                    "home",
                    "interactive-mo-ml",
                    "output",
                    "preference",
                    "incumbent.json",
                )
            )
            self.preference_models_ = {
                key: MyRankSVM(**value, random_state=ConfDict()["seed"])
                for key, value in config_dict.items()
            }

            random.seed(ConfDict()["seed"])
            for indicator, model in self.preference_models_.items():
                X, y, _ = create_preference_dataset(
                    preference_path=preference_path, indicator=indicator
                )
                if preference_budget == None:
                    indeces = range(len(y))
                else:
                    indeces = ConfDict()[f"indeces_{preference_budget}"]
                model.fit(X[indeces].copy(), y[indeces].copy())
        else:
            raise Exception("No preferences found")
        self.indicators_ = get_pareto_indicators()
        self.main_indicator = main_indicator
        self.mode = mode

    def get_indicators(self, flatten):
        new_flatten = np.array(
            adapt_encoded({idx: elem for idx, elem in enumerate(flatten)})[0]
        )
        return {
            acronym: indicator["indicator"](new_flatten)
            for acronym, indicator in self.indicators_.items()
        }

    def get_preferences(self, encoded):
        return {
            acronym: model.predict_scores(np.array([[np.array(encoded).flatten()]]))[0][
                0
            ]
            for acronym, model in self.preference_models_.items()
        }

    def get_scores_from_encoded(self, flatten, encoded):
        return {
            "indicators": self.get_indicators(copy.deepcopy(flatten)),
            "preferences": self.get_preferences(copy.deepcopy(encoded)),
        }

    def get_scores(self, pareto):
        flatten, encoded = encode_pareto(pareto)
        return self.get_scores_from_encoded(flatten, encoded)

    def train(
        self,
        random_config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            pareto = [super().train(random_config, seed, budget)]
            adapt_paretos(pareto)

            scores = self.get_scores(pareto)

            ConfDict({"paretos": ConfDict()["paretos"] + pareto})
            ConfDict({"scores": ConfDict()["scores"] + [scores]})

            score = scores[self.mode][self.main_indicator]
            return (
                score
                if (self.mode == "indicators") and (self.main_indicator in ["sp", "r2"])
                else score * -1
            )
