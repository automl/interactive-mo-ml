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

from ConfigSpace import Configuration

from smac.facade.abstract_facade import AbstractFacade

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from my_model.my_grid_model import MyGridModel
from utils.input import ConfDict

from utils.sample import grid_search

from inner_loop.mlp import MLP


class ParetoMLP(MLP):
    def __init__(self, implementation="sklearn"):
        super().__init__(implementation)
        self.p_star = super().configspace.get_hyperparameter(
            "alpha" if ConfDict()["use_case"] == "fairness" else "epoch"
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        return ConfigurationSpace(
            {
                k: v
                for k, v in super().configspace.get_hyperparameters_dict().items()
                if k != self.p_star.name
            },
            seed=ConfDict()["seed"],
        )

    @property
    def grid_configspace(self) -> ConfigurationSpace:
        return ConfigurationSpace(
            {self.p_star.name: self.p_star},
            seed=ConfDict()["seed"],
        )

    def __union_configs(
        self, random_config: Configuration, grid_config: Configuration
    ) -> Configuration:
        temp_config = super().configspace.sample_configuration()
        for config in [random_config, grid_config]:
            for k, v in config.get_dictionary().items():
                temp_config[k] = v
        return temp_config

    def train(
        self,
        random_config: Configuration,
        seed: int = 0,
        budget: int = 10,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # return [
            #     (
            #         self.__union_configs(random_config, grid_config),
            #         self.train(
            #             self.__union_configs(random_config, grid_config), seed, budget
            #         ),
            #     )
            #     for grid_config in grid_search(
            #         self.grid_configspace, ConfDict()["grid_samples"]
            #     )
            # ]

            result = []
            for idx, grid_config in enumerate(
                grid_search(self.grid_configspace, ConfDict()["grid_samples"])
            ):
                if (
                    grid_config.get_dictionary()["epoch"] > 1
                    and grid_config.get_dictionary()["epoch"] < 52
                ):
                    # print(f"    {idx}th conf of grid sampling")
                    result.append(
                        {
                            "config": self.__union_configs(
                                random_config, grid_config
                            ).get_dictionary(),
                            "evaluation": super().train(
                                self.__union_configs(random_config, grid_config),
                                seed,
                                budget,
                            ),
                        }
                    )
            return result
