from __future__ import annotations

import warnings

import numpy as np


from ConfigSpace import ConfigurationSpace, Configuration, Integer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from yahpo_gym import local_config
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from utils.input import ConfDict
from utils.output import adapt_to_mode


class LCBenchModel:
    def __init__(self):
        local_config.init_config()
        local_config.set_data_path("/home/yahpo_data-1.0")
        self.bench = benchmark_set.BenchmarkSet("lcbench")
        self.bench.set_instance(ConfDict()["task"])

    @property
    def configspace(self) -> ConfigurationSpace:
        default_space = self.bench.get_opt_space(
            # drop_fidelity_params=True,
            seed=ConfDict()["seed"]
        ).get_hyperparameters_dict()

        return ConfigurationSpace(
            {
                k: v
                if k != "epoch"
                else Integer("epoch", (2, 51), default=26, log=False)
                for k, v in default_space.items()
            },
            seed=ConfDict()["seed"],
        )

    def get_energy_consumption(self, computation_time):
        # Power consumption in watts (W) for Intel Xeon Gold 6242
        power_consumption = 150

        # Energy consumption in watt-hours (Wh)
        energy_consumption = power_consumption * (computation_time / 3600)

        # # Carbon intensity in kilograms of CO2 per kilowatt-hour (kgCO2/kWh) in Germany in 2021
        # carbon_intensity = 0.347

        # # Convert energy consumption to kilowatt-hours (kWh)
        # energy_consumption_kwh = energy_consumption / 1000

        # # Calculate carbon emissions in kilograms of CO2 (kgCO2)
        # carbon_emissions = energy_consumption_kwh * carbon_intensity

        # return carbon_emissions

        return energy_consumption

    def train(
        self,
        config: Configuration,
        seed: int = 0,
        budget: int = 52,
    ) -> dict[str, float]:
        with warnings.catch_warnings():
            value = config.get_dictionary()
            # value["epoch"] = int(np.ceil(budget))
            results_list = self.bench.objective_function(value, seed=seed)
            if ConfDict()["use_case"] == "green_automl":
                use_case_dict = {
                    f"""{ConfDict()["use_case_objective"]["metric"]}""": adapt_to_mode(
                        self.get_energy_consumption(results_list[0]["time"]),
                        ConfDict()["use_case_objective"]["mode"],
                    )
                }
            else:
                raise Exception(
                    f"""LCBenchModel does not apply to {ConfDict()["use_case"]}"""
                )

            performance_dict = {
                f"""{ConfDict()["performance_objective"]["metric"]}""": adapt_to_mode(
                    results_list[0]["val_accuracy"] / 100,
                    ConfDict()["performance_objective"]["mode"],
                )
            }

            return {**performance_dict, **use_case_dict}
