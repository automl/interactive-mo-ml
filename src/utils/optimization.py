from __future__ import annotations
import os
import time
import random

import numpy as np
import pandas as pd

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel
from inner_loop.mlp import MLP

from inner_loop.utility_pareto_mlp import UtilityParetoMLP

from utils.argparse import parse_args
from utils.common import make_dir
from utils.dataset import load_dataset_from_openml
from utils.pareto import (
    encode_pareto,
    get_pareto_from_smac,
    get_pareto_from_history,
    plot_pareto_from_history,
    plot_pareto_from_smac,
    plot_pareto,
)
from utils.sample import grid_search, random_search
from utils.input import ConfDict, create_configuration
from utils.output import (
    adapt_paretos,
    check_pictures,
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)


def multi_objective(mode="fair", preference_budget=None):
    new_output_path, is_dump = restore_results(
        mode=mode, main_indicator=None, preference_budget=preference_budget
    )
    if not is_dump:
        mlp = MLP("lcbench")

        ConfDict({"paretos": []})
        ConfDict({"scores": []})

        n_trials = ConfDict()["optimization_samples"] * (
            ConfDict()["grid_samples"] if mode == "fair" else 1
        )

        # Define our environment variables
        scenario = Scenario(
            mlp.configspace,
            objectives=ConfDict()["obj_metrics"],
            n_trials=n_trials,
            seed=ConfDict()["seed"],
            n_workers=1,
        )

        # We want to run five random configurations before starting the optimization.
        initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
        intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

        # Create our SMAC object and pass the scenario and the train method
        smac = HPOFacade(
            scenario,
            mlp.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

        # Let's optimize
        incumbents = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(mlp.configspace.get_default_configuration())
        print(f"Validated costs from default config: \n--- {default_cost}\n")

        print("Validated costs from the Pareto front (incumbents):")
        for incumbent in incumbents:
            print("---", incumbent)
            cost = smac.validate(incumbent)
            print("---", cost)

        history = get_pareto_from_smac(smac, incumbents)

        mlp = UtilityParetoMLP(
            implementation="lcbench", preference_budget=preference_budget
        )
        pareto_costs = history["pareto_costs"]
        if len(pareto_costs) < 10:
            for i in range(len(pareto_costs), 10):
                pareto_costs.append([np.nan, np.nan])
        pareto_costs = pd.DataFrame(pareto_costs).ffill().bfill().values.tolist()
        pareto_scores = mlp.get_scores_from_encoded([pareto_costs], [pareto_costs])

        ConfDict({"paretos": ConfDict()["paretos"] + [pareto_costs]})
        ConfDict({"scores": ConfDict()["scores"] + [pareto_scores]})

        save_paretos(
            np.array(ConfDict()["scores"]).flatten(),
            new_output_path,
            "scores",
        )
        save_paretos(
            ConfDict()["paretos"],
            new_output_path,
            "encoded",
        )

        file_name = "best"
        if not check_pictures(output_path=new_output_path, file_name=file_name):
            plot_pareto(
                history,
                os.path.join(
                    new_output_path,
                    file_name,
                ),
                title=f"""Sample Multi-objective  w/ {n_trials} samples""",
            )


def restore_results(mode, main_indicator, preference_budget, seed):
    new_output_path = make_dir(
        os.path.join(
            ConfDict()["output_folder"],
            *(
                [mode, main_indicator, str(seed)]
                if main_indicator != None
                else [f"multi_objective_{mode}"]
            ),
            str(preference_budget),
        )
    )
    dump_file = "dump" if main_indicator != None else "encoded"
    is_dump = check_dump(file_name=os.path.join(new_output_path, f"{dump_file}.json"))
    if is_dump:
        ConfDict(
            {
                "paretos": list(
                    load_dump(
                        file_name=os.path.join(new_output_path, f"{dump_file}.json")
                    )
                ),
                "scores": list(
                    load_dump(file_name=os.path.join(new_output_path, "scores.json"))
                ),
            },
        )
    return new_output_path, is_dump


def process_results(results, main_indicator, mode, preference_budget, seed):
    return pd.concat(
        [
            results,
            pd.DataFrame(
                ConfDict()["scores"][-1]
                | {
                    "main_indicator": main_indicator,
                    "mode": mode,
                    "preference_budget": preference_budget,
                    "seed": seed,
                }
            ),
        ]
    )


def single_objective(
    main_indicator="hv", mode="preferences", preference_budget=None, seed=0
):
    new_output_path, is_dump = restore_results(
        mode=mode,
        main_indicator=main_indicator,
        preference_budget=preference_budget,
        seed=seed,
    )
    if not is_dump:
        mlp = UtilityParetoMLP(
            implementation="lcbench",
            main_indicator=main_indicator,
            mode=mode,
            preference_budget=preference_budget,
        )

        ConfDict({"paretos": []})
        ConfDict({"scores": []})

        # Define our environment variables
        scenario = Scenario(
            mlp.configspace,
            n_trials=ConfDict()["optimization_samples"],
            seed=seed,
            n_workers=1,
        )

        # We want to run five random configurations before starting the optimization.
        initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
        intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=3)

        # Create our SMAC object and pass the scenario and the train method
        smac = HPOFacade(
            scenario,
            mlp.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

        # Let's optimize
        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(mlp.configspace.get_default_configuration())
        print(f"Validated costs from default config: \n--- {default_cost}\n")

        print("Validated costs from the Pareto front (incumbents):")
        cost = smac.validate(incumbent)
        print("---", cost)

        flatten, encoded = encode_pareto(ConfDict()["paretos"])
        save_paretos(
            ConfDict()["paretos"],
            new_output_path,
            "dump",
        )
        save_paretos(
            np.array(ConfDict()["scores"]).flatten(),
            new_output_path,
            "scores",
        )
        save_paretos(
            encoded,
            new_output_path,
            "encoded",
        )
        save_paretos(
            flatten,
            new_output_path,
            "flatten",
        )

    # print(f"Optimization time: {time.time() - start_time}")

    update_config(ConfDict()["paretos"])

    if not check_pictures(output_path=new_output_path):
        for idx, history in enumerate(ConfDict()["paretos"]):
            plot_pareto_from_history(
                history,
                os.path.join(
                    new_output_path,
                    str(idx),
                ),
                title=f"""{mode.capitalize()} w/ {main_indicator.capitalize()} ({ConfDict()["optimization_samples"]} samples)""",
            )
