import math
import os
import copy
from utils.input import ConfDict

import numpy as np
import pandas as pd

import json


def adapt_to_mode(value, mode):
    return value if mode == "min" else 1 - value


def adapt_paretos(paretos):
    for obj_idx in range(len(ConfDict()["objectives"])):
        if ConfDict()["obj_modes"][obj_idx] == "max":
            for pareto in paretos:
                for conf in pareto:
                    conf["evaluation"][
                        ConfDict()["obj_metrics"][obj_idx]
                    ] = adapt_to_mode(
                        conf["evaluation"][ConfDict()["obj_metrics"][obj_idx]],
                        ConfDict()["obj_modes"][obj_idx],
                    )


def adapt_encoded(paretos):
    for obj_idx in range(len(ConfDict()["objectives"])):
        if ConfDict()["obj_modes"][obj_idx] == "max":
            for _, pareto in paretos.items():
                for conf in pareto:
                    conf[obj_idx] = adapt_to_mode(
                        conf[obj_idx],
                        ConfDict()["obj_modes"][obj_idx],
                    )
    return paretos


def update_config(paretos):
    for obj_idx in range(len(ConfDict()["objectives"])):
        for bound in ["upper_bound", "lower_bound"]:
            func = np.max if bound == "upper_bound" else np.min
            if bound not in ConfDict()["objectives"][obj_idx]:
                ConfDict()["objectives"][obj_idx][bound] = func(
                    [
                        conf["evaluation"][ConfDict()["obj_metrics"][obj_idx]]
                        for pareto in paretos
                        for conf in pareto
                        if not math.isnan(
                            conf["evaluation"][ConfDict()["obj_metrics"][obj_idx]]
                        )
                    ]
                )


def save_paretos(paretos, path, file_name):
    with open(os.path.join(path, f"{file_name}.json"), "w") as f:
        json.dump({idx: pareto for idx, pareto in enumerate(paretos)}, f)


def save_preferences(preferences):
    preferences.to_csv(
        os.path.join(ConfDict()["output_folder"], "preferences.csv"), index=False
    )


def save_preference_scores(scores):
    with open(os.path.join(ConfDict()["output_folder"], "scores.json"), "w") as f:
        json.dump(scores, f)


def save_config(file_name):
    ConfDict()["output_folder"] = file_name.split(".")[0]
    with open(
        os.path.join("/", "home", "interactive-mo-ml", "input", file_name), "w"
    ) as f:
        json.dump(ConfDict(), f)


def check_dump(file_name="dump.json"):
    return os.path.isfile(os.path.join(ConfDict()["output_folder"], file_name))


def check_encoded():
    return os.path.isfile(os.path.join(ConfDict()["output_folder"], "encoded.json"))


def check_preferences(preference_path):
    return os.path.isfile(preference_path)


def check_pictures(output_path=None, file_name=None):
    if output_path == None:
        output_path = ConfDict()["output_folder"]
    if file_name == None:
        return all(
            [
                os.path.isfile(os.path.join(output_path, f"{idx}.png"))
                for idx in range(
                    ConfDict()[
                        "random_samples"
                        if ConfDict()["output_folder"].split("/")[-1] != "optimization"
                        else "optimization_samples"
                    ]
                )
            ]
        )
    else:
        return os.path.isfile(os.path.join(output_path, f"{file_name}.png"))


def load_json_file(file_path):
    with open(file_path) as file:
        json_file = json.load(file)
    return json_file


def load_dump(file_name="dump.json"):
    return load_json_file(os.path.join(ConfDict()["output_folder"], file_name)).values()


def load_encoded(path, file_name="encoded.json"):
    return load_json_file(os.path.join(path, file_name))


def load_preferences(path):
    return pd.read_csv(os.path.join(path, "preferences.csv"))
