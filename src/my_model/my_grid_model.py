from __future__ import annotations
from typing import Any

import numpy as np

from typing import Any

from ConfigSpace import ConfigurationSpace

from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

from utils.sample import grid_search


__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class MyGridModel(AbstractModel):
    """AbstractModel which returns random values on a call to `fit`."""

    def __init__(
        self,
        configspace: ConfigurationSpace,
        num_steps=None,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ):
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )
        self._samples = grid_search(configspace, num_steps)
        self._current = 0

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    def _train(self, X: np.ndarray, Y: np.ndarray) -> MyGridModel:
        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray.")
        if not isinstance(Y, np.ndarray):
            raise NotImplementedError("Y has to be of type np.ndarray.")

        logger.debug("(Pseudo) fit model to data.")
        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        print(X)
        if covariance_type != "diagonal":
            raise ValueError(
                "`covariance_type` can only take `diagonal` for this model."
            )

        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray.")

        next_sample = self._samples[self._current]
        self._current += 1
        return next_sample
