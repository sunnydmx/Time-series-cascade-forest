"""
Implementation of the forest model for classification in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
"""

__all__ = ["RandomForestClassifiers",
           "ExtraTreesClassifier",
           "ShapeletForestClassifier",
           "PairShapeletForestClassifier",
           "ProximityForestClassifier",
           "DrCIF",
           "SupervisedTimeSeriesForest"]

from scipy import signal, stats

from sklearn.preprocessing import StandardScaler

from sktime.utils.slope_and_trend import _slope
import math
import time

from sktime.base._base import _clone_estimator

from sktime.classification.base import BaseClassifier
from sktime.classification.sklearn._continuous_interval_tree import (
    ContinuousIntervalTree,
)
from sktime.transformations.panel.catch22 import Catch22
from sktime.utils.validation.panel import check_X_y
import numbers
from warnings import warn
import threading
from typing import List

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from joblib import Parallel, delayed
from joblib import effective_n_jobs

from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import ClassifierMixin, MultiOutputMixin
from sklearn.utils import (check_random_state,
                           compute_sample_weight)
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args

from . import _cutils as _LIB
from . import _forest as _C_FOREST
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from .tree import ShapeletTreeClassifier, PairShapeletTreeClassifier
from .tree._tree import DOUBLE
from .tree.ProximityTree import ProximityTreeClassifier

MAX_INT = np.iinfo(np.int32).max


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_mask(random_state, n_samples, n_samples_bootstrap):
    """Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    sample_indices = sample_indices.astype(np.int32)
    sample_mask = _LIB._c_sample_mask(sample_indices, n_samples)

    return sample_mask


def _parallel_build_trees(
        tree,
        X,
        y,
        n_samples_bootstrap,
        sample_weight,
        out,
        mask,
        lock
):
    """
    Private function used to fit a single tree in parallel."""
    n_samples = X.shape[0]

    sample_mask = _generate_sample_mask(tree.random_state, n_samples,
                                        n_samples_bootstrap)

    # Fit the tree on the bootstrapped samples
    if sample_weight is not None:
        sample_weight = sample_weight[sample_mask]
    print(type(X[sample_mask]))
    feature, threshold, children, value = tree.fit(
        X[sample_mask],
        y[sample_mask],
        sample_weight=sample_weight,
        check_input=False,
    )

    if not children.flags["C_CONTIGUOUS"]:
        children = np.ascontiguousarray(children)

    if not value.flags["C_CONTIGUOUS"]:
        value = np.ascontiguousarray(value)

    value = np.squeeze(value, axis=1)
    value /= value.sum(axis=1)[:, np.newaxis]

    # Set the OOB predictions
    oob_prediction = _C_FOREST.predict(
        X[~sample_mask, :], feature, threshold, children, value
    )

    with lock:
        mask += ~sample_mask
        out[~sample_mask, :] += oob_prediction

    return feature, threshold, children, value


# [Source] Sklearn.ensemble._base.py
def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int or RandomState, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


# [Source] Sklearn.ensemble._base.py
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(
        n_jobs, n_estimators // n_jobs, dtype=np.int
    )
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _accumulate_prediction(feature, threshold, children, value, X, out, lock):
    """This is a utility function for joblib's Parallel."""
    prediction = _C_FOREST.predict(X, feature, threshold, children, value)

    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


# [Source] Sklearn.ensemble._base.py
class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(self, base_estimator, *, n_estimators=10,
                 estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute.

        Sets the base_estimator_` attributes.
        """
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})

        # Pass the inferred class information to avoid redudant finding.
        estimator.classes_ = self.classes_
        estimator.n_classes_ = np.array(self.n_classes_, dtype=np.int32)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)


class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 class_weight=None,
                 max_samples=None):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.max_samples = max_samples

        # Internal containers
        self.features = []
        self.thresholds = []
        self.childrens = []
        self.values = []

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """

        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        trees = [self._make_estimator(append=False,
                                      random_state=random_state)
                 for i in range(self.n_estimators)]

        # Pre-allocate OOB estimations
        oob_decision_function = np.zeros((n_samples,
                                          self.classes_[0].shape[0]))
        mask = np.zeros(n_samples)

        lock = threading.Lock()
        rets = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                        **_joblib_parallel_args(prefer='threads',
                                                require="sharedmem"))(
            delayed(_parallel_build_trees)(
                t,
                X,
                y,
                n_samples_bootstrap,
                sample_weight,
                oob_decision_function,
                mask,
                lock, )
            for i, t in enumerate(trees))

        # Collect newly grown trees
        for feature, threshold, children, value in rets:
            # No check on feature and threshold since 1-D array is always
            # C-aligned and F-aligned.
            self.features.append(feature)
            self.thresholds.append(threshold)
            self.childrens.append(children)
            self.values.append(value)

        # Check the OOB predictions
        if (oob_decision_function.sum(axis=1) == 0).any():
            warn("Some inputs do not have OOB predictions. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob predictions.")

        prediction = (oob_decision_function /
                      oob_decision_function.sum(axis=1)[:, np.newaxis])

        self.oob_decision_function_ = prediction

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None


class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 class_weight=None,
                 max_samples=None):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples)

    def _validate_y_class_weight(self, y):

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = \
                np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ('balanced', 'balanced_subsample')
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include '
                                     '"balanced" and "balanced_subsample".'
                                     'Given "%s".'
                                     % self.class_weight)

            if (self.class_weight != 'balanced_subsample' or
                    not self.bootstrap):
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight,
                                                              y_original)

        return y, expanded_class_weight

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        check_is_fitted(self)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(
                self.features[i],
                self.thresholds[i],
                self.childrens[i],
                self.values[i],
                X,
                all_proba,
                lock)
            for i in range(self.n_estimators))

        for proba in all_proba:
            proba /= len(self.features)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba


class RandomForestClassifiers(ForestClassifier):

    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 class_weight=None,
                 max_samples=None):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "min_impurity_decrease",
                              "min_impurity_split", "random_state"),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def predict(self, X):

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):

        return self.bagging_classifier_.predict_proba(X)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """

        random_state = check_random_state(self.random_state)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        # self.n_shapelets = int(self.n_timestep_ * (self.max_shapelet_size - self.min_shapelet_size) - (
        #         self.max_shapelet_size + self.min_shapelet_size) * (
        #                                self.max_shapelet_size - self.min_shapelet_size) / 2 * n_samples * 0.05)
        # if self.n_shapelets == 0:
        #     self.n_shapelets = 1
        # print("forest.py " + self.metric)
        # self.n_shapelets = 15
        # estimator = RandomForestClassifier(n_estimators=1, oob_score=True, criterion="gini", max_features='sqrt')
        estimator = DecisionTreeClassifier(criterion='gini',
                                           max_depth=None,
                                           min_samples_leaf=1)
        # if n_dims > 1:
        #     shapelet_tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=estimator,
            bootstrap=True,
            n_jobs=-1,
            n_estimators=50,
            random_state=1,
            oob_score=True
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)
        self.bagging_classifier_.fit(X, y, sample_weight=sample_weight)
        self.oob_decision_function_ = self.bagging_classifier_.oob_decision_function_
        return self


class ExtraTreesClassifier(ForestClassifier):

    @_deprecate_positional_args
    def __init__(self,
                 n_estimators=100, *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 class_weight=None,
                 max_samples=None):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "min_impurity_decrease",
                              "min_impurity_split", "random_state"),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            class_weight=class_weight,
            max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def predict(self, X):

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):

        return self.bagging_classifier_.predict_proba(X)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """

        random_state = check_random_state(self.random_state)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        # estimator = RandomForestClassifier(n_estimators=1, oob_score=True, criterion="gini", max_features=1)
        estimator = ExtraTreeClassifier(criterion='gini',
                                        splitter="random",
                                        max_depth=None,
                                        min_samples_leaf=1)
        # if n_dims > 1:
        #     shapelet_tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=estimator,
            bootstrap=True,
            n_jobs=-1,
            n_estimators=50,
            random_state=1,
            oob_score=True
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)
        self.bagging_classifier_.fit(X, y, sample_weight=sample_weight)
        self.oob_decision_function_ = self.bagging_classifier_.oob_decision_function_
        return self


class ShapeletForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=20,  # 1
                 max_depth=None,  # 1
                 min_samples_split=2,
                 n_shapelets=20,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 bootstrap=True,
                 n_jobs=None,  # 1
                 random_state=None):  # 1
        """A shapelet forest classifier
        """
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.random_state = random_state

    def predict(self, X):
        # self.predict_proba(X)
        # 不进
        # print("??")
        # print(self.classes_[np.argmax(self.predict_proba(X), axis=1)])
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        # 进
        return self.bagging_classifier_.predict_proba(X)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """

        random_state = check_random_state(self.random_state)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        # self.n_shapelets = int(self.n_timestep_ * (self.max_shapelet_size - self.min_shapelet_size) - (
        #         self.max_shapelet_size + self.min_shapelet_size) * (
        #                                self.max_shapelet_size - self.min_shapelet_size) / 2 * n_samples * 0.05)
        # if self.n_shapelets == 0:
        #     self.n_shapelets = 1
        # print("forest.py " + self.metric)
        self.n_shapelets = 15
        shapelet_tree_classifier = ShapeletTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_shapelets=self.n_shapelets,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )

        if n_dims > 1:
            shapelet_tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=shapelet_tree_classifier,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            oob_score=True
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)
        self.bagging_classifier_.fit(X, y, sample_weight=sample_weight)
        self.oob_decision_function_ = self.bagging_classifier_.oob_decision_function_
        return self


class PairShapeletForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=20,  # 1
                 max_depth=20,  # 1
                 min_samples_split=2,
                 n_shapelets=10,
                 min_shapelet_size=0.6,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 bootstrap=True,
                 n_jobs=None,  # 1
                 random_state=None):  # 1
        """A shapelet forest classifier
        """
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.random_state = random_state

    def predict(self, X):

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):

        return self.bagging_classifier_.predict_proba(X)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """

        random_state = check_random_state(self.random_state)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        # self.n_shapelets = int(self.n_timestep_ * (self.max_shapelet_size - self.min_shapelet_size) - (
        #         self.max_shapelet_size + self.min_shapelet_size) * (
        #                                self.max_shapelet_size - self.min_shapelet_size) / 2 * n_samples * 0.01)
        # if self.n_shapelets == 0:
        #     self.n_shapelets = 1
        self.n_shapelets = 15
        shapelet_tree_classifier = PairShapeletTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_shapelets=self.n_shapelets,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )

        if n_dims > 1:
            shapelet_tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=shapelet_tree_classifier,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            oob_score=True
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)

        self.bagging_classifier_.fit(X, y, sample_weight=None)

        self.oob_decision_function_ = self.bagging_classifier_.oob_decision_function_
        return self


class ProximityForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth=100,
                 n_estimators=100,
                 n_candidates=5,
                 random_state=None,
                 bootstrap=True,
                 n_jobs=None):  # 1
        """A shapelet forest classifier
        """
        self.max_depth = max_depth
        self.n_candidates = n_candidates
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, check_input=True):
        random_state = check_random_state(self.random_state)

        # self.result.start_time_train = timeit.default_timer()
        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)


        # print("Training dataStructures ...")
        # y = np.array(y)
        # print(type(y))
        # print(y)
        # print(self.classes_)
        self.n_classes = len(self.classes_)
        n_classes = self.n_classes
        # print(self.n_classes)
        proximity_tree_classifier = ProximityTreeClassifier(
            max_depth=self.max_depth,
            n_classes=self.n_classes,
            random_state=self.random_state,
        )

        # if n_dims > 1:
        #     proximity_tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=proximity_tree_classifier,
            bootstrap=True,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            oob_score=True
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)
        # print(y)
        self.bagging_classifier_.fit(X, y, sample_weight=sample_weight)
        self.oob_decision_function_ = self.bagging_classifier_.oob_decision_function_
        return self
        # for i in range(0, app.AppContext.num_trees):
        #     self.trees[i].train(dataset)
        # self.result.end_time_train = timeit.default_timer()
        # self.result.elapsed_time_train = self.result.end_time_train - self.result.start_time_train

    """
    Testing function.
    Having a ListDataset:
    We get the serie list of each class, we try to predict the class
    and compare it with its actual class
    """
    # def predict(self, X):
    #     return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    #
    # def predict_proba(self, X):
    #     return self.bagging_classifier_.predict_proba(X)

    def predict_proba(self, X):
        return self.bagging_classifier_.predict_proba(X)

    def predict(self, X):
        # print("predict")
        # self.classes_ = [0, 1, 2, 3, 4, 5, 6]
        # print(self.classes_)  # [1. 2. 3. 4. 5. 6. 7.]
        return self.predict_proba(X), self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_trees(self):
        return self.trees

    def set_trees(self, trees):
        self.trees = trees

    def get_tree(self, i):
        return self.trees[i]

    def get_result_set(self):
        return self.result

    def get_forest_stat_collection(self):
        self.result.collate_results()
        return self.result

    def get_forest_ID(self):
        return self.forest_id

    def set_forest_ID(self, forest_id):
        self.forest_id = forest_id

    def print_results(self, dataset_name: str, experiment_id: int, prefix: str):
        self.result.print_results(dataset_name, experiment_id, prefix)

    def calculate_simple_query(self, query_serie):
        print(np.array(query_serie).tolist())
        self.predict(np.array(query_serie).tolist())


class DrCIF(BaseClassifier):
    """Diverse Representation Canonical Interval Forest Classifier (DrCIF).

    Extension of the CIF algorithm using multple representations. Implementation of the
    interval based forest making use of the catch22 feature set on randomly selected
    intervals on the base series, periodogram representation and differences
    representation described in the HIVE-COTE 2.0 paper Middlehurst et al (2021). [1]_

    Overview: Input "n" series with "d" dimensions of length "m".
    For each tree
        - Sample n_intervals intervals per representation of random position and length
        - Subsample att_subsample_size catch22 or summary statistic attributes randomly
        - Randomly select dimension for each interval
        - Calculate attributes for each interval from its representation, concatenate
          to form new data set
        - Build decision tree on new data set
    Ensemble the trees with averaged probability estimates

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_intervals : int, length 3 list of int or None, default=None
        Number of intervals to extract per representation per tree as an int for all
        representations or list for individual settings, if None extracts
        (4 + (sqrt(representation_length) * sqrt(n_dims)) / 3) intervals.
    att_subsample_size : int, default=10
        Number of catch22 or summary statistic attributes to subsample per tree.
    min_interval : int or length 3 list of int, default=4
        Minimum length of an interval per representation as an int for all
        representations or list for individual settings.
    max_interval : int, length 3 list of int or None, default=None
        Maximum length of an interval per representation as an int for all
        representations or list for individual settings, if None set to
        (representation_length / 2).
    base_estimator : BaseEstimator or str, default="DTC"
        Base estimator for the ensemble, can be supplied a sklearn BaseEstimator or a
        string for suggested options.
        "DTC" uses the sklearn DecisionTreeClassifier using entropy as a splitting
        measure.
        "CIT" uses the sktime ContinuousIntervalTree, an implementation of the original
        tree used with embedded attribute processing for faster predictions.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of ndarray with shape (total_intervals,2)
        Stores indexes of each intervals start and end points for all classifiers.
    atts_ : list of shape (n_estimators) of array with shape (att_subsample_size)
        Attribute indexes of the subsampled catch22 or summary statistic for all
        classifiers.
    dims_ : list of shape (n_estimators) of array with shape (total_intervals)
        The dimension to extract attributes from each interval for all classifiers.
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances,total_intervals * att_subsample_size)
        The transformed dataset for all classifiers. Only saved when
        save_transformed_data is true.

    See Also
    --------
    CanonicalIntervalForest

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/interval_based/DrCIF.java>`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." arXiv preprint arXiv:2104.07551 (2021).

    Examples
    --------
    >>> from sktime.classification.interval_based import DrCIF
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True) # doctest: +SKIP
    >>> clf = DrCIF(
    ...     n_estimators=3, n_intervals=2, att_subsample_size=2
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    DrCIF(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "interval",
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        n_estimators=50,
        n_intervals=None,
        att_subsample_size=10,
        min_interval=4,
        max_interval=None,
        base_estimator="CIT",
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        save_transformed_data=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.att_subsample_size = att_subsample_size
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.base_estimator = base_estimator

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.total_intervals_ = 0
        self.estimators_ = []
        self.intervals_ = []
        self.atts_ = []
        self.dims_ = []
        self.transformed_data_ = []

        self._n_estimators = n_estimators
        self._n_intervals = n_intervals
        self._att_subsample_size = att_subsample_size
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._base_estimator = base_estimator

        self.oob_decision_function_ = None

        super(DrCIF, self).__init__()

    def _fit(self, X, y):
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        if self.base_estimator.lower() == "dtc":
            self._base_estimator = DecisionTreeClassifier(criterion="entropy")
        elif self.base_estimator.lower() == "cit":
            self._base_estimator = ContinuousIntervalTree()
        elif isinstance(self.base_estimator, BaseEstimator):
            self._base_estimator = self.base_estimator
        else:
            raise ValueError("DrCIF invalid base estimator given.")

        X_p = np.zeros(
            (
                self.n_instances_,
                self.n_dims_,
                int(
                    math.pow(2, math.ceil(math.log(self.series_length_, 2)))
                    - self.series_length_
                ),
            )
        )
        X_p = np.concatenate((X, X_p), axis=2)
        X_p = np.abs(np.fft.fft(X_p)[:, :, : int(X_p.shape[2] / 2)])

        X_d = np.diff(X, 1)

        if self.n_intervals is None:
            self._n_intervals = [None, None, None]
            self._n_intervals[0] = 4 + int(
                (math.sqrt(self.series_length_) * math.sqrt(self.n_dims_)) / 3
            )
            self._n_intervals[1] = 4 + int(
                (math.sqrt(X_p.shape[2]) * math.sqrt(self.n_dims_)) / 3
            )
            self._n_intervals[2] = 4 + int(
                (math.sqrt(X_d.shape[2]) * math.sqrt(self.n_dims_)) / 3
            )
        elif isinstance(self.n_intervals, int):
            self._n_intervals = [self.n_intervals, self.n_intervals, self.n_intervals]
        elif isinstance(self.n_intervals, list) and len(self.n_intervals) == 3:
            self._n_intervals = self.n_intervals
        else:
            raise ValueError("DrCIF n_intervals must be an int or list of length 3.")
        for i, n in enumerate(self._n_intervals):
            if n <= 0:
                self._n_intervals[i] = 1

        if self.att_subsample_size > 29:
            self._att_subsample_size = 29

        if isinstance(self.min_interval, int):
            self._min_interval = [
                self.min_interval,
                self.min_interval,
                self.min_interval,
            ]
        elif isinstance(self.min_interval, list) and len(self.min_interval) == 3:
            self._min_interval = self.min_interval
        else:
            raise ValueError("DrCIF min_interval must be an int or list of length 3.")
        if self.series_length_ <= self._min_interval[0]:
            self._min_interval[0] = self.series_length_ - 1
        if X_p.shape[2] <= self._min_interval[1]:
            self._min_interval[1] = X_p.shape[2] - 1
        if X_d.shape[2] <= self._min_interval[2]:
            self._min_interval[2] = X_d.shape[2] - 1

        if self.max_interval is None:
            self._max_interval = [
                int(self.series_length_ / 2),
                int(X_p.shape[2] / 2),
                int(X_d.shape[2] / 2),
            ]
        elif isinstance(self.max_interval, int):
            self._max_interval = [
                self.max_interval,
                self.max_interval,
                self.max_interval,
            ]
        elif isinstance(self.max_interval, list) and len(self.max_interval) == 3:
            self._max_interval = self.max_interval
        else:
            raise ValueError("DrCIF max_interval must be an int or list of length 3.")
        for i, n in enumerate(self._max_interval):
            if n < self._min_interval[i]:
                self._max_interval[i] = self._min_interval[i]

        self.total_intervals_ = sum(self._n_intervals)

        if time_limit > 0:
            self._n_estimators = 0
            self.estimators_ = []
            self.intervals_ = []
            self.atts_ = []
            self.dims_ = []
            self.transformed_data_ = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._threads_to_use)(
                    delayed(self._fit_estimator)(
                        X,
                        X_p,
                        X_d,
                        y,
                        i,
                    )
                    for i in range(self._threads_to_use)
                )

                (
                    estimators,
                    intervals,
                    dims,
                    atts,
                    transformed_data,
                ) = zip(*fit)

                self.estimators_ += estimators
                self.intervals_ += intervals
                self.atts_ += atts
                self.dims_ += dims
                self.transformed_data_ += transformed_data

                self._n_estimators += self._threads_to_use
                train_time = time.time() - start_time
        else:
            fit = Parallel(n_jobs=self._threads_to_use)(
                delayed(self._fit_estimator)(
                    X,
                    X_p,
                    X_d,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            (
                self.estimators_,
                self.intervals_,
                self.dims_,
                self.atts_,
                self.transformed_data_,
            ) = zip(*fit)
            # self.oob_decision_function_ = self._get_train_probs(X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        # self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        rng = check_random_state(self.random_state)
        # return np.array(
        #     [
        #         self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
        #         for prob in self._predict_proba(X)
        #     ]
        # )
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        n_test_instances, _, series_length = X.shape
        if series_length != self.series_length_:
            raise ValueError(
                "ERROR number of attributes in the train does not match "
                "that in the test data"
            )

        X_p = np.zeros(
            (
                n_test_instances,
                self.n_dims_,
                int(
                    math.pow(2, math.ceil(math.log(self.series_length_, 2)))
                    - self.series_length_
                ),
            )
        )
        X_p = np.concatenate((X, X_p), axis=2)
        X_p = np.abs(np.fft.fft(X_p)[:, :, : int(X_p.shape[2] / 2)])

        X_d = np.diff(X, 1)

        y_probas = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._predict_proba_for_estimator)(
                X,
                X_p,
                X_d,
                self.estimators_[i],
                self.intervals_[i],
                self.dims_[i],
                self.atts_[i],
            )
            for i in range(self._n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes_) * self._n_estimators
        )
        return output

    def _get_train_probs(self, X, y) -> np.ndarray:
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return self._single_class_y_pred(X, method="predict_proba")

        n_instances, n_dims, series_length = X.shape

        if (
            n_instances != self.n_instances_
            or n_dims != self.n_dims_
            or series_length != self.series_length_
        ):
            raise ValueError(
                "n_instances, n_dims, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        p = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._train_probas_for_estimator)(
                y,
                i,
            )
            for i in range(self._n_estimators)
        )
        y_probas, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        divisors = np.zeros(n_instances)
        for oob in oobs:
            for inst in oob:
                divisors[inst] += 1

        for i in range(n_instances):
            results[i] = (
                np.ones(self.n_classes_) * (1 / self.n_classes_)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes_) * divisors[i])
            )

        return results

    def _fit_estimator(self, X, X_p, X_d, y, idx):
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            _drcif_feature,
        )

        c22 = Catch22(outlier_norm=True)
        T = [X, X_p, X_d]
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        transformed_x = np.empty(
            shape=(self._att_subsample_size * self.total_intervals_, self.n_instances_),
            dtype=np.float32,
        )

        atts = rng.choice(29, self._att_subsample_size, replace=False)
        dims = rng.choice(self.n_dims_, self.total_intervals_, replace=True)
        intervals = np.zeros((self.total_intervals_, 2), dtype=int)

        p = 0
        j = 0
        for r in range(0, len(T)):
            transform_length = T[r].shape[2]

            # Find the random intervals for classifier i, transformation r
            # and concatenate features
            for _ in range(0, self._n_intervals[r]):
                if rng.random() < 0.5:
                    intervals[j][0] = rng.randint(
                        0, transform_length - self._min_interval[r]
                    )
                    len_range = min(
                        transform_length - intervals[j][0],
                        self._max_interval[r],
                    )
                    length = (
                        rng.randint(0, len_range - self._min_interval[r])
                        + self._min_interval[r]
                        if len_range - self._min_interval[r] > 0
                        else self._min_interval[r]
                    )
                    intervals[j][1] = intervals[j][0] + length
                else:
                    intervals[j][1] = (
                        rng.randint(0, transform_length - self._min_interval[r])
                        + self._min_interval[r]
                    )
                    len_range = min(intervals[j][1], self._max_interval[r])
                    length = (
                        rng.randint(0, len_range - self._min_interval[r])
                        + self._min_interval[r]
                        if len_range - self._min_interval[r] > 0
                        else self._min_interval[r]
                    )
                    intervals[j][0] = intervals[j][1] - length

                for a in range(0, self._att_subsample_size):
                    transformed_x[p] = _drcif_feature(
                        T[r], intervals[j], dims[j], atts[a], c22, case_id=j
                    )
                    p += 1

                j += 1

        tree = _clone_estimator(self._base_estimator, random_state=rs)
        transformed_x = transformed_x.T
        transformed_x = transformed_x.round(8)
        if isinstance(self._base_estimator, ContinuousIntervalTree):
            transformed_x = np.nan_to_num(
                transformed_x, False, posinf=np.nan, neginf=np.nan
            )
        else:
            transformed_x = np.nan_to_num(transformed_x, False, 0, 0, 0)
        tree.fit(transformed_x, y)

        return [
            tree,
            intervals,
            dims,
            atts,
            transformed_x if self.save_transformed_data else None,
        ]

    def _predict_proba_for_estimator(
        self, X, X_p, X_d, classifier, intervals, dims, atts
    ):
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            _drcif_feature,
        )

        c22 = Catch22(outlier_norm=True)
        if isinstance(self._base_estimator, ContinuousIntervalTree):
            return classifier._predict_proba_drcif(
                X, X_p, X_d, c22, self._n_intervals, intervals, dims, atts
            )
        else:
            T = [X, X_p, X_d]

            transformed_x = np.empty(
                shape=(self._att_subsample_size * self.total_intervals_, X.shape[0]),
                dtype=np.float32,
            )

            p = 0
            j = 0
            for r in range(0, len(T)):
                for _ in range(0, self._n_intervals[r]):
                    for a in range(0, self._att_subsample_size):
                        transformed_x[p] = _drcif_feature(
                            T[r], intervals[j], dims[j], atts[a], c22, case_id=j
                        )
                        p += 1
                    j += 1

            transformed_x = transformed_x.T
            transformed_x.round(8)
            np.nan_to_num(transformed_x, False, 0, 0, 0)

            return classifier.predict_proba(transformed_x)

    def _train_probas_for_estimator(self, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        indices = range(self.n_instances_)
        subsample = rng.choice(self.n_instances_, size=self.n_instances_)
        oob = [n for n in indices if n not in subsample]

        results = np.zeros((self.n_instances_, self.n_classes_))
        if len(oob) == 0:
            return [results, oob]

        clf = _clone_estimator(self._base_estimator, rs)
        clf.fit(self.transformed_data_[idx][subsample], y[subsample])
        probas = clf.predict_proba(self.transformed_data_[idx][oob])

        if probas.shape[1] != self.n_classes_:
            new_probas = np.zeros((probas.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas

        for n, proba in enumerate(probas):
            results[oob[n]] += proba

        return [results, oob]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10, "n_intervals": 2, "att_subsample_size": 4}
        else:
            return {
                "n_estimators": 2,
                "n_intervals": 2,
                "att_subsample_size": 2,
                "save_transformed_data": True,
            }


class SupervisedTimeSeriesForest(BaseClassifier):
    """Supervised Time Series Forest (STSF).

    An ensemble of decision trees built on intervals selected through a supervised
    process as described in _[1].
    Overview: Input n series length m
    For each tree
        - sample X using class-balanced bagging
        - sample intervals for all 3 representations and 7 features using supervised
        - method
        - find mean, median, std, slope, iqr, min and max using their corresponding
        - interval for each rperesentation, concatenate to form new data set
        - build decision tree on new data set
    Ensemble the trees with averaged probability estimates.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_instances_ : int
        The number of train cases.
    series_length_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    intervals : array-like of shape [n_estimators][3][7][n_intervals][2]
        Stores indexes of all start and end points for all estimators. Each estimator
        contains indexes for each representaion and feature combination.
    estimators_ : list of shape (n_estimators) of DecisionTreeClassifier
        The collections of estimators trained in fit.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/STSF.java>`_.

    References
    ----------
    .. [1] Cabello, Nestor, et al. "Fast and Accurate Time Series Classification
       Through Supervised Interval Search." IEEE ICDM 2020

    Examples
    --------
    >>> from sktime.classification.interval_based import SupervisedTimeSeriesForest
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = SupervisedTimeSeriesForest(n_estimators=5)
    >>> clf.fit(X_train, y_train)
    SupervisedTimeSeriesForest(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "interval",
    }

    def __init__(
        self,
        n_estimators=200,
        n_jobs=1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_instances_ = 0
        self.series_length_ = 0
        self.estimators_ = []
        self.intervals_ = []

        self._base_estimator = DecisionTreeClassifier(criterion="entropy")
        self._stats = [np.mean, np.median, np.std, _slope, stats.iqr, np.min, np.max]

        super(SupervisedTimeSeriesForest, self).__init__()

    def _fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

        Uses supervised intervals and summary features

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification. STSF has no bespoke method for multivariate
            classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        X = X.squeeze(1)

        self.n_instances_, self.series_length_ = X.shape
        rng = check_random_state(self.random_state)
        cls, class_counts = np.unique(y, return_counts=True)

        self.intervals_ = [[[] for _ in range(3)] for _ in range(self.n_estimators)]

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)

        balance_cases = np.zeros(0, dtype=np.int32)
        average = math.floor(self.n_instances_ / self.n_classes_)
        for i, c in enumerate(cls):
            if class_counts[i] < average:
                cls_idx = np.where(y == c)[0]
                balance_cases = np.concatenate(
                    (rng.choice(cls_idx, size=average - class_counts[i]), balance_cases)
                )

        fit = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._fit_estimator)(
                X,
                X_p,
                X_d,
                y,
                balance_cases,
                i,
            )
            for i in range(self.n_estimators)
        )

        self.estimators_, self.intervals_ = zip(*fit)

        return self

    def _predict(self, X) -> np.ndarray:
        """Find predictions for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : nd.array of shape = (n_instances, n_classes)
            Predicted probabilities
        """
        X = X.squeeze(1)

        _, X_p = signal.periodogram(X)
        X_d = np.diff(X, 1)

        y_probas = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._predict_proba_for_estimator)(
                X,
                X_p,
                X_d,
                self.intervals_[i],
                self.estimators_[i],
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes_) * self.n_estimators
        )
        return output

    def _transform(self, X, intervals):
        """Compute summary stats.

        Find the mean, median, standard deviation, slope, iqr, min and max using
        intervals of input data X generated for each.
        """
        n_instances, _ = X.shape
        total_intervals = 0

        for i in range(len(self._stats)):
            total_intervals += len(intervals[i])
        transformed_x = np.zeros((total_intervals, n_instances), dtype=np.float32)

        p = 0
        for i, f in enumerate(self._stats):
            n_intervals = len(intervals[i])

            for j in range(n_intervals):
                X_slice = X[:, intervals[i][j][0] : intervals[i][j][1]]
                transformed_x[p] = f(X_slice, axis=1)
                p += 1

        return transformed_x.T

    def _get_intervals(self, X, y, rng):
        """Generate intervals using a recursive function and random split point."""
        n_instances, series_length = X.shape
        split_point = (
            int(series_length / 2)
            if series_length <= 8
            else rng.randint(4, series_length - 4)
        )

        cls, class_counts = np.unique(y, return_counts=True)

        s = StandardScaler()
        X_norm = s.fit_transform(X)

        intervals = []
        for function in self._stats:
            function_intervals = []
            self._supervised_interval_search(
                X_norm,
                y,
                function,
                function_intervals,
                cls,
                class_counts,
                0,
                split_point + 1,
            )
            self._supervised_interval_search(
                X_norm,
                y,
                function,
                function_intervals,
                cls,
                class_counts,
                split_point + 1,
                series_length,
            )
            intervals.append(function_intervals)

        return intervals

    def _supervised_interval_search(
        self, X, y, function, function_intervals, classes, class_counts, start, end
    ):
        """Recursive function for finding intervals for a feature using fisher score.

        Given a start and end point the series is split in half and both intervals
        are evaluated. The half with the higher score is retained and used as the new
        start and end for a recursive call.
        """
        series_length = end - start
        if series_length < 4:
            return

        e = start + int(series_length / 2)

        X_l = function(X[:, start:e], axis=1)
        X_r = function(X[:, e:end], axis=1)

        s1 = _fisher_score(X_l, y, classes, class_counts)
        s2 = _fisher_score(X_r, y, classes, class_counts)

        if s2 < s1:
            function_intervals.append([start, e])
            self._supervised_interval_search(
                X,
                y,
                function,
                function_intervals,
                classes,
                class_counts,
                start,
                e,
            )
        else:
            function_intervals.append([e, end])
            self._supervised_interval_search(
                X,
                y,
                function,
                function_intervals,
                classes,
                class_counts,
                e,
                end,
            )

    def _fit_estimator(self, X, X_p, X_d, y, balance_cases, idx):
        """Fit an estimator - a clone of base_estimator - on input data (X, y).

        Transformed using the supervised intervals for each feature and representation.
        """
        estimator = clone(self._base_estimator)
        rs = 5465 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        estimator.set_params(random_state=rs)
        rng = check_random_state(rs)

        class_counts = np.zeros(0)
        while class_counts.shape[0] != self.n_classes_:
            bag = np.concatenate(
                (rng.choice(self.n_instances_, size=self.n_instances_), balance_cases)
            )
            _, class_counts = np.unique(y[bag], return_counts=True)
        n_instances = bag.shape[0]
        bag = bag.astype(int)

        transformed_x = np.zeros((n_instances, 0), dtype=np.float32)

        intervals = self._get_intervals(X[bag], y[bag], rng)
        transformed_x = np.concatenate(
            (transformed_x, self._transform(X[bag], intervals)),
            axis=1,
        )

        intervals_p = self._get_intervals(X_p[bag], y[bag], rng)
        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_p[bag], intervals_p)),
            axis=1,
        )

        intervals_d = self._get_intervals(X_d[bag], y[bag], rng)
        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_d[bag], intervals_d)),
            axis=1,
        )

        return [
            estimator.fit(transformed_x, y[bag]),
            [intervals, intervals_p, intervals_d],
        ]

    def _predict_proba_for_estimator(self, X, X_p, X_d, intervals, estimator):
        """Find probability estimates for each class for all cases in X."""
        n_instances, _ = X.shape
        transformed_x = np.zeros((n_instances, 0), dtype=np.float32)

        transformed_x = np.concatenate(
            (transformed_x, self._transform(X, intervals[0])),
            axis=1,
        )

        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_p, intervals[1])),
            axis=1,
        )

        transformed_x = np.concatenate(
            (transformed_x, self._transform(X_d, intervals[2])),
            axis=1,
        )

        return estimator.predict_proba(transformed_x)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"n_estimators": 10}
        else:
            return {"n_estimators": 2}
