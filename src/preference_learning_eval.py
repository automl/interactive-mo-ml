import json
import random
import warnings
import logging

import os

import numpy as np
import pandas as pd

from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.model_selection import KFold

from sklearn.metrics import silhouette_samples, silhouette_score

from jenkspy import JenksNaturalBreaks

from utils.argparse import parse_args
from utils.common import get_tuning_datasets, make_dir
from utils.input import ConfDict, create_configuration
from utils.output import adapt_encoded, check_preferences, load_encoded, load_json_file
from utils.pareto import get_pareto_indicators

from utils.preference_learning import (
    configspace,
    create_preference_dataset,
    objective,
    objective_kendall,
    evaluate_model,
)

from ranker.my_rank_svc import MyRankSVM

if __name__ == "__main__":
    datasets = get_tuning_datasets()
    create_configuration(datasets)
    ConfDict({"datasets": datasets, "indicators": {}})

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    incumbents = {}
    indicators = ["hv", "sp", "ms", "r2"]
    # ConfDict()["config_dicts"] = load_json_file(
    #     os.path.join("/", "home", "interactive-mo-ml", "output", "preference", "incumbent.json")
    # )
    for indicator in indicators:
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
                ConfDict()[dataset]["flatten_encoded"] = {
                    key: np.array(value).flatten()
                    for key, value in load_encoded(
                        os.path.join(ConfDict()[dataset]["output_folder"])
                    ).items()
                }
                ConfDict()[dataset]["scores"] = load_json_file(
                    os.path.join(ConfDict()[dataset]["output_folder"], "scores.json")
                )
                ConfDict()[dataset]["test_folds"] = np.array(
                    [
                        fold
                        for _, fold in KFold(n_splits=5).split(
                            range(
                                int(
                                    len(ConfDict()[dataset]["scores"].keys())
                                    / len(indicators)
                                )
                            )
                            # np.unique([int(elem.split("_")[1]) for elem in ConfDict()[dataset]["scores"].keys()])
                        )
                    ]
                )
                ConfDict()[dataset]["preferences"] = preferences[
                    ["pair_0", "pair_1"]
                ].to_numpy()
                ConfDict()["current_indicator"] = indicator
                ConfDict()["indicators"][indicator] = {
                    "iteration": 0,
                    "summary": pd.DataFrame(),
                    "binnerizer": None,
                }

                min_clusters = 30

                indicator_scores = np.array(
                    [
                        value
                        for key, value in ConfDict()[dataset]["scores"].items()
                        if indicator in key
                    ]
                )
                range_n_clusters = range(min_clusters, len(indicator_scores))
                max_silhouette = -1
                final_clusters = min_clusters
                for n_clusters in range_n_clusters:
                    jnb = JenksNaturalBreaks(n_clusters)
                    jnb.fit(indicator_scores)
                    new_silhouette = silhouette_score(
                        indicator_scores.reshape(-1, 1), jnb.labels_
                    )
                    if new_silhouette >= max_silhouette:
                        ConfDict()["indicators"][indicator]["binnerizer"] = jnb
                        max_silhouette = new_silhouette
                        final_clusters = n_clusters

            else:
                raise Exception(f"No preference file found for {dataset}")

        # config_dict = ConfDict()["config_dicts"][ConfDict()["current_indicator"]]
        # objective_kendall(config_dict)

        # ConfDict()["indicators"][indicator]["summary"].to_csv(
        #     os.path.join(
        #         make_dir(
        #             os.path.join("/", "home", "interactive-mo-ml", "output", "preference"),
        #         ),
        #         f"{indicator}_mv.csv",
        #     ),
        #     index=False,
        # )

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
                objective_kendall,
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
