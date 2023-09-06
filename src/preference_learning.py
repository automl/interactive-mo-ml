import json
import random
import warnings
import logging

import os

import numpy as np
import pandas as pd

from smac import HyperparameterOptimizationFacade, Scenario

from utils.argparse import parse_args
from utils.common import get_tuning_datasets, make_dir
from utils.input import ConfDict, create_configuration
from utils.output import check_preferences
from utils.pareto import get_pareto_indicators

from utils.preference_learning import configspace, create_preference_dataset, objective


if __name__ == "__main__":
    datasets = get_tuning_datasets()
    create_configuration(datasets)
    ConfDict({"datasets": datasets, "indicators": {}})

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    incumbents = {}
    for indicator in ["hv", "sp", "ms", "r2"]:
        for dataset in datasets:
            if check_preferences(
                os.path.join(ConfDict()[dataset]["output_folder"], "preferences.csv")
            ):
                X, y, preferences = create_preference_dataset(
                    preference_path=os.path.join(ConfDict()[dataset]["output_folder"]),
                    indicator=indicator,
                )

                ConfDict()[dataset]["X"] = X
                ConfDict()[dataset]["Y"] = y
                ConfDict()[dataset]["preferences"] = preferences[
                    ["pair_0", "pair_1"]
                ].to_numpy()
                ConfDict()["current_indicator"] = indicator
                ConfDict()["indicators"][indicator] = {
                    "iteration": 0,
                    "summary": pd.DataFrame(),
                }

            else:
                raise Exception(f"No preference file found for {dataset}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Next, we create an object, holding general information about the run
            scenario = Scenario(
                configspace(),
                n_trials=ConfDict()["preference_samples"],
                seed=ConfDict()["seed"],
                n_workers=1,
            )

            # We want to run the facade's default initial design, but we want to change the number
            # of initial configs to 5.
            initial_design = HyperparameterOptimizationFacade.get_initial_design(
                scenario, n_configs=50
            )
            intensifier = HyperparameterOptimizationFacade.get_intensifier(
                scenario, max_config_calls=3
            )

            # Now we use SMAC to find the best hyperparameters
            smac = HyperparameterOptimizationFacade(
                scenario,
                objective,
                initial_design=initial_design,
                intensifier=intensifier,
                overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
            )

            incumbent = smac.optimize()

            # Get cost of default configuration
            default_cost = smac.validate(configspace().get_default_configuration())
            print(f"Default cost: {default_cost}")

            # Let's calculate the cost of the incumbent
            incumbent_cost = smac.validate(incumbent)
            print(f"Incumbent cost: {incumbent_cost}")

            incumbents[indicator] = incumbent.get_dictionary()

            ConfDict()["indicators"][indicator]["summary"].to_csv(
                os.path.join(
                    make_dir(
                        os.path.join(
                            "/", "home", "interactive-mo-ml", "output", "preference"
                        ),
                    ),
                    f"{indicator}.csv",
                ),
                index=False,
            )

    with open(
        os.path.join(
            "/",
            "home",
            "interactive-mo-ml",
            "output",
            "preference",
            "incumbent.json",
        ),
        "w",
    ) as f:
        json.dump(incumbents, f)


# %%
