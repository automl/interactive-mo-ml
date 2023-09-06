from __future__ import annotations

from typing import Any

from smac.model.random_model import RandomModel
from smac.multi_objective import MeanAggregationStrategy
from smac.utils.logging import get_logger

from smac.scenario import Scenario


__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class MyRandomModel(RandomModel, MeanAggregationStrategy):
    """RandomModel which can be used for MO."""

    def __init__(
        self, scenario: Scenario, objective_weights: list[float] | None = None
    ):
        RandomModel.__init__(self, scenario)
        MeanAggregationStrategy.__init__(self, scenario, objective_weights)

    def update_on_iteration_start(self) -> None:
        """Update the internal state on start of each SMBO iteration."""
        pass

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        return 0.0

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "types": self._types,
            "bounds": self._bounds,
            "pca_components": self._pca_components,
            "objective_weights": self._objective_weights,
        }
