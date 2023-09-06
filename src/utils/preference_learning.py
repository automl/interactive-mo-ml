from itertools import combinations
import os
import random

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


from ConfigSpace import Configuration
from ConfigSpace.conditions import EqualsCondition, InCondition, NotEqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from ranker.my_rank_svc import MyRankSVM
from scipy.stats import kendalltau
from utils.common import make_dir
from utils.input import ConfDict
from utils.output import (
    load_encoded,
    load_preferences,
)


def create_preference_dataset(preference_path, indicator):
    flatten_encoded = {
        key: np.array(value).flatten()
        for key, value in load_encoded(preference_path).items()
    }

    # paretos = np.array(list(flatten_encoded.values()))
    # # for n_clusters in range(3, len(paretos)):
    # n_clusters = 6
    # kmedoids = KMedoids(n_clusters=n_clusters, random_state=ConfDict()["seed"]).fit(
    #     paretos
    # )
    # centers = kmedoids.cluster_centers_

    # # print(n_clusters)
    # # print(silhouette_score(paretos, kmedoids.labels_))
    # print(np.where(flatten_encoded == [elem for elem in centers]))
    # # print()

    preferences = load_preferences(path=preference_path)
    X = np.array(
        [
            np.array([flatten_encoded[str(pair[0])], flatten_encoded[str(pair[1])]])
            for pair in preferences[["pair_0", "pair_1"]].to_numpy()
        ]
    )
    y = np.array(
        [
            # np.array([0, 1] if preference == 0 else [1, 0])
            preference
            for preference in preferences[f"preference_{indicator}"].to_numpy()
        ]
    )
    return X, y, preferences


def configspace() -> ConfigurationSpace:
    # C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
    C = UniformFloatHyperparameter("C", 0.8, 1.5, log=False, default_value=1.0)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    dual = CategoricalHyperparameter("dual", ["True", "False"], default_value="False")
    loss = CategoricalHyperparameter(
        "loss", ["squared_hinge", "hinge"], default_value="squared_hinge"
    )
    penalty = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l1")

    # n_features = UniformIntegerHyperparameter(
    #     "n_features", 1, 19, default_value=19, log=False
    # )
    # svm_implementation = CategoricalHyperparameter(
    #     "svm_implementation",
    #     ["logistic", "linear"],
    #     default_value="linear",
    # )
    # features_implementation = CategoricalHyperparameter(
    #     "features_implementation",
    #     ["selection", "pca", "none"],
    #     default_value="none",
    # )
    normalize = CategoricalHyperparameter(
        "normalize", ["True", "False"], default_value="False"
    )

    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            C,
            tol,
            dual,
            loss,
            penalty,
            # n_features,
            # svm_implementation,
            # features_implementation,
            normalize,
        ]
    )

    # loss_condition = EqualsCondition(loss, svm_implementation, "linear")
    # dual_condition = EqualsCondition(dual, svm_implementation, "linear")
    # svm_implementation_condition = EqualsCondition(svm_implementation, penalty, "l2")
    # n_features_condition = InCondition(
    #     n_features, features_implementation, ["selection", "pca"]
    # )
    # cs.add_condition(dual_condition)
    # cs.add_condition(loss_condition)
    # cs.add_condition(svm_implementation_condition)
    # cs.add_condition(n_features_condition)
    return cs


def compute_raw_results(config_dict, result_dict, dataset, mode, seed):
    splits = KFold(n_splits=5).split(
        ConfDict()[dataset]["X"]
    )

    raw_results = []
    for train, test in splits:
        i = int(mode.split("_")[-1])
        fate = MyRankSVM(**config_dict, random_state=seed)
        random.seed(seed)
        new_train = random.sample(list(train), 28 * i)
        fate.fit(
            ConfDict()[dataset]["X"][new_train].copy(),
            ConfDict()[dataset]["Y"][new_train].copy(),
        )

        raw_results.append(
            pd.DataFrame(
                {
                    "pair_0": [
                        pref[0] for pref in ConfDict()[dataset]["preferences"][test]
                    ],
                    "pair_0_score": [
                        elem
                        for list in fate.predict_scores(
                            np.array(
                                [[elem[0]] for elem in ConfDict()[dataset]["X"][test]]
                            )
                        )
                        for elem in list
                    ],
                    "pair_1": [
                        pref[1] for pref in ConfDict()[dataset]["preferences"][test]
                    ],
                    "pair_1_score": [
                        elem
                        for list in fate.predict_scores(
                            np.array(
                                [[elem[1]] for elem in ConfDict()[dataset]["X"][test]]
                            )
                        )
                        for elem in list
                    ],
                    "y_true": ConfDict()[dataset]["Y"][test],
                    "y_pred": fate.predict(ConfDict()[dataset]["X"][test].copy()),
                }
            )
        )

    result_dict[mode].append(
        round(
            np.mean(
                [
                    accuracy_score(result["y_true"], result["y_pred"])
                    for result in raw_results
                ]
            ),
            2,
        )
    )
    pd.concat(raw_results, ignore_index=True).to_csv(
        os.path.join(
            make_dir(
                os.path.join(
                    ConfDict()[dataset]["output_folder"],
                    ConfDict()["current_indicator"],
                    f"{mode}",
                )
            ),
            f"""predictions_{ConfDict()["indicators"][ConfDict()["current_indicator"]]["iteration"]}.csv""",
        ),
        index=False,
    )


def get_index_of(my_list, reverse):
    return [sorted(my_list, reverse=reverse).index(elem) for elem in my_list]


def evaluate_model(config_dict, result_dict, dataset, mode, seed):
    splits = KFold(n_splits=5).split(
        ConfDict()[dataset]["X"]
    )

    raw_results, raw_results_pair = [], []
    for idx, (train, test) in enumerate(splits):
        i_shuffle = int(mode.split("_")[-1])
        fate = MyRankSVM(**config_dict, random_state=seed)
        random.seed(seed)
        new_train = random.sample(list(train), 28 * i_shuffle)
        fate.fit(
            ConfDict()[dataset]["X"][new_train].copy(),
            ConfDict()[dataset]["Y"][new_train].copy(),
        )
        # min_i = min(ConfDict()[dataset]["test_folds"][idx])
        # y_true = (
        #     get_index_of(
        #         [
        #             ConfDict()[dataset]["scores"][
        #                 f"""{ConfDict()["current_indicator"]}_{i}"""
        #             ]
        #             for i in ConfDict()[dataset]["test_folds"][idx]
        #         ]
        #     )
        #     + min_i
        # )
        y_true = get_index_of(
            ConfDict()["indicators"][ConfDict()["current_indicator"]][
                "binnerizer"
            ].predict(
                [
                    ConfDict()[dataset]["scores"][
                        f"""{ConfDict()["current_indicator"]}_{i}"""
                    ]
                    for i in ConfDict()[dataset]["test_folds"][idx]
                ]
            ),
            ConfDict()["current_indicator"] in ["hv", "ms"],
        )
        # if ConfDict()["current_indicator"] in ["hv", "ms"]:
        #     y_true = y_true[::-1]
        # else:
        #     y_true = y_true

        y_pred = get_index_of(
            fate.predict_scores(
                np.array(
                    [
                        [
                            ConfDict()[dataset]["flatten_encoded"][str(i)].copy()
                            for i in ConfDict()[dataset]["test_folds"][idx]
                        ]
                    ]
                )
            )[0],
            True,
        )
        corr, alpha = kendalltau(y_true, y_pred)

        raw_results.append(
            pd.DataFrame(
                {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "corr": corr,
                    "alpha": alpha,
                }
            )
        )

        raw_results_pair.append(
            pd.DataFrame(
                {
                    "pair_0": [
                        pref[0] for pref in ConfDict()[dataset]["preferences"][test]
                    ],
                    "pair_0_score": [
                        elem
                        for list in fate.predict_scores(
                            np.array(
                                [[elem[0]] for elem in ConfDict()[dataset]["X"][test]]
                            )
                        )
                        for elem in list
                    ],
                    "pair_1": [
                        pref[1] for pref in ConfDict()[dataset]["preferences"][test]
                    ],
                    "pair_1_score": [
                        elem
                        for list in fate.predict_scores(
                            np.array(
                                [[elem[1]] for elem in ConfDict()[dataset]["X"][test]]
                            )
                        )
                        for elem in list
                    ],
                    "y_true": ConfDict()[dataset]["Y"][test],
                    "y_pred": fate.predict(ConfDict()[dataset]["X"][test].copy()),
                }
            )
        )
    # for metric in ["corr", "alpha"]:
    #     for aggregation in ["mean", "std"]:
    #         values = [result[metric] for result in raw_results]
    #         result_dict[f"{mode}_{metric}_{aggregation}"] = round(
    #             np.mean(values) if aggregation == "mean" else np.std(values), 2
    #         )
    result_dict[mode].append(
        round(np.mean([result["corr"] for result in raw_results]), 2)
    )

    pd.concat(raw_results, ignore_index=True).to_csv(
        os.path.join(
            make_dir(
                os.path.join(
                    ConfDict()[dataset]["output_folder"],
                    ConfDict()["current_indicator"],
                    f"{mode}",
                )
            ),
            f"""predictions_{ConfDict()["indicators"][ConfDict()["current_indicator"]]["iteration"]}.csv""",
        ),
        index=False,
    )

    pd.concat(raw_results_pair, ignore_index=True).to_csv(
        os.path.join(
            make_dir(
                os.path.join(
                    ConfDict()[dataset]["output_folder"],
                    ConfDict()["current_indicator"],
                    f"{mode}",
                )
            ),
            f"""predictions_pair_{ConfDict()["indicators"][ConfDict()["current_indicator"]]["iteration"]}.csv""",
        ),
        index=False,
    )


def objective(config: Configuration, seed: int = 0) -> float:
    config_dict = config.get_dictionary()
    result_dict = {
        "iteration": ConfDict()["indicators"][ConfDict()["current_indicator"]][
            "iteration"
        ],
        "cross_validation_1": [],
        "cross_validation_2": [],
        "cross_validation_3": [],
        "cross_validation_4": [],
    }
    try:
        for dataset in ConfDict()["datasets"]:
            for mode in result_dict.keys():
                if mode != "iteration":
                    compute_raw_results(config_dict, result_dict, dataset, mode, seed)

        log = "success"
    except Exception as e:
        log = e

    ConfDict()["indicators"][ConfDict()["current_indicator"]]["summary"] = pd.concat(
        [
            ConfDict()["indicators"][ConfDict()["current_indicator"]]["summary"],
            pd.DataFrame(
                {
                    **{key: [value] for key, value in result_dict.items()},
                    **{key: [value] for key, value in config_dict.items()},
                    **{"log": [log]},
                }
            ),
        ],
        ignore_index=True,
    )

    ConfDict()["indicators"][ConfDict()["current_indicator"]]["iteration"] += 1

    return 1 - np.mean(result_dict["cross_validation_4"])


def objective_kendall(config: Configuration, seed: int = 0) -> float:
    config_dict = config.get_dictionary()
    result_dict = {
        "iteration": ConfDict()["indicators"][ConfDict()["current_indicator"]][
            "iteration"
        ],
        "cross_validation_1": [],
        "cross_validation_2": [],
        "cross_validation_3": [],
        "cross_validation_4": [],
    }

    try:
        for dataset in ConfDict()["datasets"]:
            for mode in result_dict.keys():
                if mode != "iteration":
                    evaluate_model(config_dict, result_dict, dataset, mode, seed)

        log = "success"
    except Exception as e:
        log = e

    ConfDict()["indicators"][ConfDict()["current_indicator"]]["summary"] = pd.concat(
        [
            ConfDict()["indicators"][ConfDict()["current_indicator"]]["summary"],
            pd.DataFrame(
                {
                    **{key: [value] for key, value in result_dict.items()},
                    **{
                        f"{key}_mean": [np.mean(value)]
                        for key, value in result_dict.items()
                    },
                    **{
                        f"{key}_std": [np.std(value)]
                        for key, value in result_dict.items()
                    },
                    **{key: [value] for key, value in config_dict.items()},
                    **{"log": [log]},
                }
            ),
        ],
        ignore_index=True,
    )
    ConfDict()["indicators"][ConfDict()["current_indicator"]]["iteration"] += 1

    return 1 - np.mean(result_dict["cross_validation_4"])


def get_preference_budgets():
    n_folds = 5
    combination_per_fold = len(
        list(combinations(range(int(ConfDict()["random_samples"] / n_folds)), 2))
    )
    preference_budgets = np.linspace(
        combination_per_fold,
        combination_per_fold * n_folds,
        n_folds,
        dtype=int,
        endpoint=True,
    )
    for preference_budget in preference_budgets:
        ConfDict(
            {
                f"indeces_{preference_budget}": random.sample(
                    range(combination_per_fold * n_folds), preference_budget
                )
            }
        )
    return preference_budgets
