from __future__ import annotations
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ConfigSpace import Configuration

from smac.facade.abstract_facade import AbstractFacade

from utils.input import ConfDict
from utils.output import adapt_to_mode, save_paretos

from pymoo.indicators.hv import Hypervolume
from performance.r2 import R2
from performance.spacing import Spacing
from performance.spread import Spread
from performance.my_hypervolume import MyHypervolume

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def get_pareto_indicators():
    ref_point, ideal_point = [1, 1], [0, 0]
    # for obj_idx in range(len(ConfDict()["objectives"])):
    #     ref_point[obj_idx] = adapt_to_mode(
    #         ConfDict()["objectives"][obj_idx]["upper_bound"]
    #         if ConfDict()["obj_modes"][obj_idx] == "min"
    #         else ConfDict()["objectives"][obj_idx]["lower_bound"],
    #         ConfDict()["obj_modes"][obj_idx],
    #     )

    return {
        "hv": {
            "indicator": MyHypervolume(ref_point=ref_point),
            "mode": getattr(pd.Series, "idxmax"),
        },
        "sp": {"indicator": Spacing(), "mode": getattr(pd.Series, "idxmin")},
        "ms": {
            "indicator": Spread(nadir=ref_point, ideal=ideal_point),
            "mode": getattr(pd.Series, "idxmax"),
        },
        "r2": {
            "indicator": R2(ideal=ideal_point),
            "mode": getattr(pd.Series, "idxmin"),
        },
    }


def get_pareto_from_history(history: list[dict]):
    def _get_pareto_indeces(costs):
        is_efficient = np.arange(costs.shape[0])
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[
                nondominated_point_mask
            ]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        return is_efficient

    costs = np.array(
        [
            [
                elem["evaluation"][ConfDict()["obj_metrics"][0]],
                elem["evaluation"][ConfDict()["obj_metrics"][1]],
            ]
            for elem in history
        ]
    )
    # average_costs = np.array([np.average(cost) for _, cost in costs])
    # configs = [config for config, _ in history]

    pareto_costs = [
        costs[i]
        for i in _get_pareto_indeces(
            np.array(
                [
                    [
                        adapt_to_mode(cost[0], ConfDict()["obj_modes"][0]),
                        adapt_to_mode(cost[1], ConfDict()["obj_modes"][1]),
                    ]
                    for cost in costs
                ]
            )
        )
    ]
    # average_pareto_costs = [average_costs[i] for i in get_pareto_indeces(costs)]
    # pareto_configs = [configs[i] for i in get_pareto_indeces(costs)]

    return {"costs": costs, "pareto_costs": pareto_costs}


def encode_pareto(paretos):
    encoded_paretos = []
    flatten_paretos = []
    for history in paretos:
        pareto_dict = get_pareto_from_history(history)
        pareto_list = [
            elem
            if np.any(np.all(elem == pareto_dict["pareto_costs"], axis=1))
            else np.nan * np.ones(elem.shape)
            for elem in pareto_dict["costs"]
        ]
        encoded_paretos.append(
            pd.DataFrame(pareto_list).ffill().bfill().values.tolist()
        )
        flatten_paretos.append(pareto_dict["costs"].tolist())
    return flatten_paretos, encoded_paretos


def get_pareto_from_smac(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    costs = []
    pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        for obj_idx in range(len(ConfDict()["objectives"])):
            average_cost[obj_idx] = adapt_to_mode(
                average_cost[obj_idx],
                ConfDict()["obj_modes"][obj_idx],
            )

            if average_cost[obj_idx] < ConfDict()["objectives"][obj_idx]["lower_bound"]:
                ConfDict()["objectives"][obj_idx]["lower_bound"] = average_cost[obj_idx]
            if average_cost[obj_idx] > ConfDict()["objectives"][obj_idx]["upper_bound"]:
                ConfDict()["objectives"][obj_idx]["upper_bound"] = average_cost[obj_idx]

        if config in incumbents:
            pareto_costs += [average_cost]
        else:
            costs += [average_cost]

    return {"costs": costs, "pareto_costs": pareto_costs}


def plot_pareto(summary, output_path, title):
    # Let's work with a numpy array
    costs = np.vstack(summary["costs"])
    pareto_costs = np.vstack(summary["pareto_costs"])
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(costs_x, costs_y, marker="x", label="Configuration")
    ax.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    ax.step(
        [pareto_costs_x[0]]
        + pareto_costs_x.tolist()
        + [np.max(costs_x)],  # We add bounds
        [
            np.max(costs_y)
            if ConfDict()["obj_modes"][0] == ConfDict()["obj_modes"][1]
            else np.min(costs_y)
        ]
        + pareto_costs_y.tolist()
        + [
            np.min(pareto_costs_y)
            if ConfDict()["obj_modes"][0] == ConfDict()["obj_modes"][1]
            else np.max(pareto_costs_y)
        ],  # We add bounds
        where="post",
        linestyle=":",
    )

    ax.set_xlim(
        [
            ConfDict()["objectives"][0]["lower_bound"],
            ConfDict()["objectives"][0]["upper_bound"],
        ]
    )
    ax.set_ylim(
        [
            ConfDict()["objectives"][1]["lower_bound"],
            ConfDict()["objectives"][1]["upper_bound"],
        ]
    )
    ax.set_title(ConfDict()["task"])
    ax.set_title(title)
    ax.set_xlabel(ConfDict()["obj_metrics"][0])
    ax.set_ylabel(ConfDict()["obj_metrics"][1])
    ax.legend()
    fig.savefig(output_path)

    plt.close()


def plot_pareto_from_history(history: list[dict], output_path, title):
    plot_pareto(get_pareto_from_history(history), output_path, title)


def plot_pareto_from_smac(
    smac: AbstractFacade, incumbents: list[Configuration], output_path, title
):
    plot_pareto(get_pareto_from_smac(smac, incumbents), output_path, title)
