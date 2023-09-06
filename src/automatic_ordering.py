# %%
from collections import ChainMap
import logging
import random
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from itertools import combinations

from sklearn.model_selection import KFold


from utils.argparse import parse_args
from utils.input import ConfDict, create_configuration
from utils.output import (
    adapt_paretos,
    load_encoded,
    check_preferences,
    load_preferences,
    save_preference_scores,
    save_preferences,
    adapt_to_mode,
    adapt_encoded,
)
from utils.pareto import get_pareto_indicators


logger = logging.getLogger()
logger.disabled = True


def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))


if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(args.conf_file)

    random.seed(ConfDict()["seed"])

    flatten = load_encoded(path=ConfDict()["output_folder"], file_name="flatten.json")
    combinations = np.array(
        [rSubset(fold, 2) for _, fold in KFold(n_splits=5).split(list(flatten.keys()))]
    )
    combinations = combinations.reshape((-1, 2))
    tot = len(combinations)
    # random.shuffle(combinations)

    paretos = adapt_encoded(flatten)
    indicators = get_pareto_indicators()

    single_scores = {
        f"{acronym}_{idx}": indicator["indicator"](np.array(pareto))
        for idx, pareto in paretos.items()
        for acronym, indicator in indicators.items()
    }
    save_preference_scores(single_scores)

    preferences = pd.DataFrame()

    for pair in combinations:
        scores = [
            {
                f"pair_{idx}": pair[idx],
                f"score_{idx}_{acronym}": single_scores[f"{acronym}_{pair[idx]}"],
            }
            for idx, pareto in enumerate(
                [np.array(paretos[str(elem)]) for elem in pair]
            )
            for acronym in indicators.keys()
        ]
        scores = dict(ChainMap(*scores))

        preferences = pd.concat(
            [
                preferences,
                pd.DataFrame(
                    dict(
                        ChainMap(
                            *[
                                {key: [value] for key, value in scores.items()},
                                {
                                    f"preference_{acronym}": [
                                        indicator["mode"](
                                            pd.Series(
                                                [
                                                    scores[f"score_{idx}_{acronym}"]
                                                    for idx, _ in enumerate(pair)
                                                ]
                                            )
                                        )
                                    ]
                                    for acronym, indicator in indicators.items()
                                },
                            ]
                        )
                    )
                ),
            ],
            ignore_index=True,
        )
        save_preferences(preferences)
    preferences[["pair_0"]]

# %%
