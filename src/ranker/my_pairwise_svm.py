import logging

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest


from csrank.learner import Learner

from utils.common import check_for_bool
from utils.input import ConfDict

logger = logging.getLogger(__name__)


class MyPairwiseSVM(Learner):
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
        """Create an instance of the PairwiseSVM model for any preference learner.

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
        use_logistic_regression : bool, optional
            Whether to fit a Linear Support Vector machine or a Logistic
            Regression model. You may want to prefer the simpler Logistic
            Regression model on a large sample size.
        random_state : int, RandomState instance or None, optional
            Seed of the pseudorandom generator or a RandomState instance
        **kwargs
            Keyword arguments for the algorithms

        References
        ----------
            [1] Joachims, T. (2002, July). "Optimizing search engines using clickthrough data.", Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 133-142). ACM.
        """
        self.C = float(C)
        self.tol = float(tol)
        self.dual = check_for_bool(dual)
        self.loss = loss
        self.penalty = penalty
        self.normalize = check_for_bool(normalize)
        self.fit_intercept = fit_intercept
        self.svm_implementation = svm_implementation
        self.features_implementation = features_implementation
        self.n_features = n_features
        self.random_state = random_state

    def _pre_fit(self):
        super()._pre_fit()
        self.random_state_ = check_random_state(self.random_state)

    def _build_pipeline(self):
        if self.svm_implementation == "logistic":
            self.model_ = LogisticRegression(
                C=self.C,
                tol=self.tol,
                dual=self.dual,
                penalty=self.penalty,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state_,
            )
            logger.info("Logistic Regression model ")
        elif self.svm_implementation == "linear":
            self.model_ = LinearSVC(
                C=self.C,
                tol=self.tol,
                dual=self.dual,
                loss=self.loss,
                penalty=self.penalty,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state_,
            )
            logger.info("Linear SVC model ")
        # elif self.implementation == "kernel":
        #     self.model_ = SVC(
        #         C=self.C,
        #         kernel=self.kernel,
        #         degree=self.degree,
        #         gamma=self.gamma,
        #         coef0=self.coef0,
        #         shrinking=self.shrinking,
        #         tol=self.tol,
        #         class_weight=self.class_weight,
        #         max_iter=self.max_iter,
        #         random_state=self.random_state,
        #         decision_function_shape="ovr",
        #         random_state=self.random_state_,
        #     )
        else:
            raise Exception("Invalid SVM implementation")

        # self.scaler_ = StandardScaler()

        if self.features_implementation == "selection":
            self.features_ = SelectKBest(k=self.n_features)
        elif self.features_implementation == "pca":
            self.features_ = PCA(n_components=self.n_features)
        else:
            self.features_ = FunctionTransformer()

        return Pipeline(
            [
                # ("scaler", self.scaler_),
                ("features", self.features_),
                ("model", self.model_),
            ]
        )

    def fit(self, X, Y, **kwargs):
        """
        Fit a generic preference learning model on a provided set of queries.
        The provided queries can be of a fixed size (numpy arrays).

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_objects, n_features)
            Feature vectors of the objects
        Y : numpy array, shape (n_samples, n_objects, n_features)
            Preferences in form of Orderings or Choices for given n_objects
        **kwargs
            Keyword arguments for the fit function

        """

        self._pre_fit()
        _n_instances, self.n_objects_fit_, self.n_object_features_fit_ = X.shape

        if self.n_objects_fit_ < 2:
            # Nothing to learn, cannot create pairwise instances.
            return self

        x_train, y_single = self._convert_instances_(X, Y)

        logger.debug("Finished Creating the model, now fitting started")

        self.pipeline_ = self._build_pipeline()

        self.pipeline_.fit(x_train, y_single)
        if self.features_implementation == "selection":
            self.features_expl_ = str(self.features_.scores_)
        # elif self.features_implementation == "pca":
        #     self.features_expl_ = str(self.features_.components_)
        else:
            self.features_expl_ = ""
        self.weights_ = self.model_.coef_.flatten()
        if self.fit_intercept:
            self.weights_ = np.append(self.weights_, self.model_.intercept_)
        logger.debug("Fitting Complete")

        return self

    def _predict_scores_fixed(self, X, **kwargs):
        assert X.shape[-1] == self.n_object_features_fit_
        logger.info("For Test instances {} objects {} features {}".format(*X.shape))

        x_test = X.copy()

        if self.normalize:
            x_test = np.array([self.scaler_.transform(elem) for elem in x_test])

        if self.features_implementation != "none":
            x_test = np.array([self.features_.transform(elem) for elem in x_test])

        if self.fit_intercept:
            scores = np.dot(x_test, self.weights_[:-1])
        else:
            scores = np.dot(x_test, self.weights_)

        logger.info("Done predicting scores")

        return np.array(scores)

    def _convert_instances_(self, X, Y):
        raise NotImplementedError
