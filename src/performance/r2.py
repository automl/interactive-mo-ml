import numpy as np
from scipy.spatial.distance import pdist, squareform
from pymoo.core.indicator import Indicator
from pymoo.indicators.distance_indicator import (
    at_least_2d_array,
    derive_ideal_and_nadir_from_pf,
)

from scipy.spatial import distance

from performance.my_indicator import MyIndicator

# =========================================================================================================
# Implementation
# =========================================================================================================


class R2(MyIndicator):
    def __init__(self, ideal=None):
        """Spacing indicator
        The smaller the value this indicator assumes, the most uniform is the distribution of elements on the pareto front.

        Parameters
        ----------

        ideal : 1d array, optional
            Ideal point, by default None
        """

        super().__init__(ideal=ideal)

    def do(self, F, *args, **kwargs):
        """Obtain the spacing indicator given a Pareto front

        Parameters
        ----------
        F : numpy.array (n_samples, n_obj)
            Pareto front

        Returns
        -------
        float
            Spacing indicator
        """
        return super().do(F, *args, **kwargs)

    def _do(self, F, *args, **kwargs):
        """
        Calculates the average weighted Chebyshev distance (R2 indicator) given a Pareto front and a reference point.

        Args:
            pareto_front (ndarray): The Pareto front represented as a 2D array, where each row represents a solution.
            reference_point (ndarray): The reference point represented as a 1D array.
            weights (ndarray, optional): Weights for each objective. If None, assumes equal weights. Default is None.

        Returns:
            float: The average weighted Chebyshev distance (R2 indicator).
        """
        # num_solutions, num_objectives = F.shape

        # if weights is None:
        #     weights = np.ones(num_objectives) / num_objectives

        # distances = np.abs(F - self.ideal)
        # max_distances = np.max(distances, axis=0)
        # weighted_distances = weights * max_distances
        # r2_indicator = np.mean(weighted_distances)

        return np.min([distance.chebyshev(f, self.ideal) for f in F])
