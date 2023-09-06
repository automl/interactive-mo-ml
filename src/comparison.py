# %%
from __future__ import annotations
from itertools import combinations
import logging
import os
import time
import random

import numpy as np
import pandas as pd

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from utils.argparse import parse_args
from utils.common import make_dir
from utils.comparison import get_cell_value
from utils.dataset import load_dataset_from_openml
from utils.optimization import (
    multi_objective,
    single_objective,
    restore_results,
    process_results,
)
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
    plot_pareto_from_smac,
    get_pareto_indicators,
)
from utils.sample import grid_search, random_search
from utils.input import ConfDict, create_configuration
from utils.preference_learning import get_preference_budgets
from utils.output import (
    adapt_paretos,
    check_pictures,
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)


logger = logging.getLogger()
logger.disabled = True


if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(
        file_name=args.conf_file,
        origin="optimization",
    )

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    preference_budgets = get_preference_budgets()

    indicators = get_pareto_indicators().keys()
    results = pd.DataFrame()
    for main_indicator in indicators:
        for mode in ["indicators", "preferences"]:
            for preference_budget in preference_budgets:
                for seed in [0, 1, 42]:
                    new_output_path, is_dump = restore_results(
                        main_indicator=main_indicator,
                        mode=mode,
                        preference_budget=preference_budget,
                        seed=seed,
                    )
                    results = process_results(
                        results,
                        main_indicator=main_indicator,
                        mode=mode,
                        preference_budget=preference_budget,
                        seed=seed,
                    )
    # for mode in ["fair", "unfair"]:
    #     for preference_budget in preference_budgets:
    #         new_output_path, is_dump = restore_results(
    #             main_indicator=None,
    #             mode=mode,
    #             preference_budget=preference_budget,
    #         )
    #         results = process_results(
    #             results,
    #             main_indicator=main_indicator,
    #             mode=mode,
    #             preference_budget=preference_budget,
    #         )
    results = results.reset_index(inplace=False).rename(
        columns={"index": "second_indicator"}
    )
    results.to_csv(
        os.path.join(ConfDict()["output_folder"], "results_raw.csv"), index=False
    )
    results = (
        results.groupby(
            ["main_indicator", "second_indicator", "preference_budget", "mode"]
        )
        .agg(
            {
                "indicators": ["mean", "std"],
                "preferences": ["mean", "std"],
                "seed": lambda x: " ".join(str(x.values)),
            }
        )
        .reset_index()
    )
    results.columns = [" ".join(col).strip() for col in results.columns.values]
    results.to_csv(
        os.path.join(ConfDict()["output_folder"], "results.csv"), index=False
    )

    per_budget_results = {
        preference_budget: pd.concat(
            [
                pd.DataFrame({"indicators\preferences": indicators}),
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                column: [
                                    f"{get_cell_value(results, preference_budget, column, row)}"
                                    for row in indicators
                                ]
                            }
                        )
                        for column in indicators
                    ],
                    axis=1,
                ),
            ],
            axis=1,
        )
        for preference_budget in preference_budgets
    }
    for k, v in per_budget_results.items():
        v.to_csv(
            os.path.join(ConfDict()["output_folder"], f"budget_{k}.csv"), index=False
        )
# %%
