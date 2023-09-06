import logging

import numpy as np

from sklearn.preprocessing import StandardScaler

from csrank.core.pairwise_svm import PairwiseSVM
from csrank.objectranking.object_ranker import ObjectRanker
from csrank.objectranking.util import generate_complete_pairwise_dataset

from ranker.my_pairwise_svm import MyPairwiseSVM

__all__ = ["RankSVM"]
logger = logging.getLogger(__name__)


class MyRankSVM(ObjectRanker, MyPairwiseSVM):
    def __init__(
        self,
        C=1.0,
        tol=1e-4,
        dual=False,
        loss="squared_hinge",
        penalty="l1",
        normalize=False,
        fit_intercept=True,
        svm_implementation="linear",
        features_implementation="none",
        n_features=None,
        random_state=None,
        **kwargs,
    ):
        """
        Create an instance of the :class:`PairwiseSVM` model for learning a object ranking function.
        It learns a linear deterministic utility function of the form :math:`U(x) = w \\cdot x`, where :math:`w` is
        the weight vector. It is estimated using *pairwise preferences* generated from the rankings.
        The ranking for the given query set :math:`Q` is defined as:

        .. math::

            ρ(Q)  = \\operatorname{argsort}_{x \\in Q}  \\; U(x)

        Parameters
        ----------
        C : float, optional
            Penalty parameter of the error term
        tol : float, optional
            Optimization tolerance
        normalize : bool, optional
            If True, the data will be normalized before fitting.
        fit_intercept : bool, optional
            If True, the linear model will also fit an intercept.
        random_state : int, RandomState instance or None, optional
            Seed of the pseudorandom generator or a RandomState instance
        **kwargs
            Keyword arguments for the algorithms

        References
        ----------
            [1] Joachims, T. (2002, July). "Optimizing search engines using clickthrough data.", Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 133-142). ACM.
        """
        super().__init__(
            C=C,
            tol=tol,
            dual=dual,
            loss=loss,
            penalty=penalty,
            normalize=normalize,
            fit_intercept=fit_intercept,
            svm_implementation=svm_implementation,
            features_implementation=features_implementation,
            n_features=n_features,
            random_state=random_state,
            **kwargs,
        )
        logger.info("Initializing network")

    def fit(self, X, Y, **kwargs):
        self._pre_fit()
        _n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape
        return super().fit(X, Y, **kwargs)

    def _convert_instances_(self, X, Y):
        logger.debug("Creating the Dataset")
        if self.normalize:
            self.scaler_ = StandardScaler().fit(
                np.unique(X.reshape(X.shape[0] * X.shape[1], X.shape[2]), axis=0)
            )
            X = np.array(
                [
                    [
                        self.scaler_.transform([elem[0]]),
                        self.scaler_.transform([elem[1]]),
                    ]
                    for elem in X
                ]
            ).reshape(X.shape)
        exp_dataset = [
            [
                [X[idx][y] - X[idx][int(not (y))], 1],
                [X[idx][int(not (y))] - X[idx][y], -1],
            ]
            for idx, y in enumerate(Y)
        ]
        x_train, y_single = np.array(
            [elem[0] for pair in exp_dataset for elem in pair]
        ), np.array([elem[1] for pair in exp_dataset for elem in pair])
        logger.debug("Finished the Dataset with instances {}".format(x_train.shape[0]))
        return x_train, y_single

    def predict(self, X, **kwargs):
        return [
            0 if (preference == np.array([0, 1])).all() else 1
            for preference in super().predict(X, **kwargs)
        ]

    def super_predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
