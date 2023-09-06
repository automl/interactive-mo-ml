from functools import reduce
import os
import json
import copy
from utils.common import make_dir

from utils.dataset import load_dataset_from_openml


class ConfDict(dict):
    def __new__(cls, conf=None):
        if not hasattr(cls, "instance"):
            cls.instance = super(ConfDict, cls).__new__(cls)
        return cls.instance

    def get_dict(cls):
        return dict(cls.instance)


def get_configuration(file_name: str, origin: str = "preliminar_sampling"):
    with open(
        os.path.join("/", "home", "interactive-mo-ml", "input", file_name)
    ) as file:
        conf = json.load(file)

    conf["output_folder"] = os.path.join(
        "/",
        "home",
        "interactive-mo-ml",
        "output",
        conf["output_folder"],
        origin,
    )
    make_dir(conf["output_folder"])

    conf["objectives"] = [conf["performance_objective"], conf["use_case_objective"]]
    conf["obj_metrics"] = [c["metric"] for c in conf["objectives"]]
    conf["obj_modes"] = [c["mode"] for c in conf["objectives"]]
    if conf["model"] != "lcbench":
        (
            conf["X"],
            conf["y"],
            conf["categorical_indicator"],
            conf["feature_names"],
        ) = load_dataset_from_openml(conf["dataset"])
        if "sensitive_feature" in conf["use_case_objective"]:
            conf["use_case_objective"]["sensitive_feature_idx"] = conf[
                "feature_names"
            ].index(conf["use_case_objective"]["sensitive_feature"])

    return conf


def create_configuration(file_name, origin="preliminar_sampling"):
    if isinstance(file_name, str):
        conf = get_configuration(file_name, origin)
    else:
        conf = {f"{file}": get_configuration(file, origin) for file in file_name}

        test_list = list(conf.values())
        common_keys = reduce(
            lambda acc, key: acc
            + (
                [key] if all(test_list[0][key] == ele[key] for ele in test_list) else []
            ),
            list(test_list[0].keys()),
            [],
        )
        for key in common_keys:
            conf[key] = copy.deepcopy(test_list[0][key])

    ConfDict(conf)
