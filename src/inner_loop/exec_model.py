from __future__ import annotations

import warnings


from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

from sklearn.model_selection import train_test_split


from codecarbon import EmissionsTracker

from utils.input import ConfDict
from utils.output import adapt_to_mode


from utils.input import ConfDict


class ExecModel:
    @property
    def configspace(self) -> ConfigurationSpace:
        return ConfigurationSpace(
            {
                "n_layer": Integer("n_layer", (1, 10), default=1),
                "n_neurons": Integer("n_neurons", (8, 256), log=True, default=10),
                "activation": Categorical(
                    "activation", ["logistic", "tanh", "relu"], default="tanh"
                ),
                "solver": Categorical("solver", ["sgd", "adam"], default="adam"),
                "learning_rate_init": Float(
                    "learning_rate_init", (0.0001, 1.0), default=0.001, log=True
                ),
                "alpha": Float("alpha", (0.000001, 10.0), default=0.0001, log=True),
                "n_epochs": Integer("n_epochs", (10, 500), default=10, log=True),
            }
        )

    def train(
        self,
        config: Configuration,
        seed: int = 0,
        budget: int = 52,
    ) -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # try:
            self.implementation.build_model(config, budget)

            X_train, X_test, y_train, y_test = train_test_split(
                ConfDict()["X"],
                ConfDict()["y"],
                test_size=0.33,
                stratify=ConfDict()["y"],
                random_state=seed,
            )

            if ConfDict()["use_case"] == "green_automl":
                tracker = EmissionsTracker()
                tracker.start()

            self.implementation.train_model(X_train, y_train)

            if ConfDict()["use_case"] == "green_automl":
                use_case_dict = {
                    f"""{ConfDict()["use_case_objective"]["metric"]}""": adapt_to_mode(
                        tracker.stop(), ConfDict()["use_case_objective"]["mode"]
                    )
                }

            y_pred = self.implementation.evaluate_model(X_test)

            performance_dict = {
                f"""{ConfDict()["performance_objective"]["metric"]}""": adapt_to_mode(
                    globals()[f"""{ConfDict()["performance_objective"]["metric"]}"""](
                        y_test, y_pred
                    ),
                    ConfDict()["performance_objective"]["mode"],
                )
            }

            if ConfDict()["use_case"] == "fairness":
                use_case_dict = {
                    f"""{ConfDict()["use_case_objective"]["metric"]}""": globals()[
                        f"""{ConfDict()["use_case_objective"]["metric"]}"""
                    ](
                        y_test,
                        y_pred,
                        sensitive_features=X_test[
                            :, ConfDict()["use_case_objective"]["sensitive_feature_idx"]
                        ],
                    )
                }

            return {**performance_dict, **use_case_dict}

            # except:
            #     print("Something went wrong!")
