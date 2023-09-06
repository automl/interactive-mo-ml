# %%
import random
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from itertools import combinations
from IPython.display import clear_output


from utils.argparse import parse_args
from utils.input import ConfDict, create_configuration
from utils.output import (
    load_encoded,
    check_preferences,
    load_preferences,
    save_preferences,
)


def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))


if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(args.conf_file)

    random.seed(ConfDict()["seed"])

    encoded = load_encoded(ConfDict()["output_folder"])
    combinations = rSubset(encoded.keys(), 2)
    tot = len(combinations)
    random.shuffle(combinations)

    if check_preferences(ConfDict()["output_folder"]):
        preferences = load_preferences(ConfDict()["output_folder"])
        combinations = combinations[
            (
                combinations.index(
                    (
                        str(preferences.iloc[-1]["pair_0"]),
                        str(preferences.iloc[-1]["pair_1"]),
                    )
                )
                + 1
            ) :
        ]
    else:
        preferences = pd.DataFrame(columns=["pair_0", "pair_1", "preference"])

    for pair in combinations:
        clear_output(wait=True)
        print(f"{preferences.shape[0] + 1} out of {tot}")
        fig, axs = plt.subplots(1, 2)
        for i in range(len(pair)):
            axs[i].imshow(
                mpimg.imread(f"""{ConfDict()["output_folder"]}/{pair[i]}.png""")
            )
            axs[i].axis("off")
            axs[i].set_title(pair[i])
        fig.set_size_inches(15, 8)
        plt.show()
        pref = input("Preference?")
        os.system("clear")
        if pref == "s":
            preference = 0
        elif pref == "d":
            preference = 1
        else:
            raise Exception("Invalid option")
        preferences = pd.concat(
            [
                preferences,
                pd.DataFrame(
                    {
                        "pair_0": [pair[0]],
                        "pair_1": [pair[1]],
                        "preference": int(preference),
                    }
                ),
            ],
            ignore_index=True,
        )
        save_preferences(preferences)

# %%
