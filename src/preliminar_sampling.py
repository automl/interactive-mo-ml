# %%
from __future__ import annotations
import logging
import os
import time
import random

import numpy as np

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from inner_loop.pareto_mlp import ParetoMLP

from utils.argparse import parse_args
from utils.dataset import load_dataset_from_openml
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
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
    save_config,
)


if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(args.conf_file)

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    # start_time = time.time()

    if check_dump():
        paretos = load_dump()
    else:
        mlp = ParetoMLP("lcbench")
        # grid_samples = grid_search(configspace=mlp.configspace, num_steps=2)
        random_samples = random_search(
            configspace=mlp.configspace, num_samples=ConfDict()["random_samples"]
        )

        paretos = []
        for idx, sample in enumerate(random_samples):
            # print(f"{idx}th conf of random sampling")
            paretos += [mlp.train(sample)]

        adapt_paretos(paretos)
        save_paretos(paretos, ConfDict()["output_folder"], "dump")

    # print(f"Optimization time: {time.time() - start_time}")

    update_config(paretos)

    if not check_pictures():
        for idx, history in enumerate(paretos):
            plot_pareto_from_history(
                history,
                os.path.join(ConfDict()["output_folder"], str(idx)),
                title=ConfDict()["task"],
            )

    flatten, encoded = encode_pareto(paretos)
    save_paretos(flatten, ConfDict()["output_folder"], "flatten")
    save_paretos(encoded, ConfDict()["output_folder"], "encoded")
    save_config(args.conf_file)

# %%
