"""
Implementation of the decision tree in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py
"""

__all__ = ["DecisionTreeClassifier",
           "ExtraTreeClassifier",
           "ShapeletTreeClassifier",
           "PairShapeletTreeClassifier"]

import numbers
import warnings
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from ._tree_builder import ClassificationShapeletTreeBuilder
from ._tree_builder import ClassificationShapeletTreePredictor
from ._tree_builder import ClassificationPairShapeletTreeBuilder
from ._tree_builder import ClassificationPairShapeletTreePredictor
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import DepthFirstTreeBuilder
from ._tree import Tree
from . import _tree, _splitter, _criterion, _distance, _dtw_distance, _euclidean_distance, _impurity, _tree_builder, distance
from .distance import DISTANCE_MEASURE

# =============================================================================
# Types and constants
# =============================================================================

# DTYPE = _tree.DTYPE
# DOUBLE = _tree.DOUBLE
from numpy import float64 as DOUBLE
from numpy import float64 as DTYPE

CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
CRITERIA_REG = {"mse": _criterion.MSE, "mae": _criterion.MAE}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter,
                   "random": _splitter.RandomSplitter}


# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    @_deprecate_positional_args
    def __init__(self, *,
                 criterion,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 max_features,
                 random_state,
                 min_impurity_decrease,
                 min_impurity_split,
                 class_weight=None,
                 presort='deprecated'):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort

    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    @property
    def n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    @property
    def n_internals(self):
        """Return the number of internal nodes of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of internal nodes.
        """
        check_is_fitted(self)
        return self.tree_.n_internals

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):

        random_state = check_random_state(self.random_state)

        # if X.dtype != np.uint8:
        #     msg = 'The dtype of `X` should be `np.uint8`, but got {} instead.'
        #     raise RuntimeError(msg.format(X.dtype))

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(X, y,
                                       validate_separately=(check_X_params,
                                                            check_y_params))

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        # `classes_` and `n_classes_` were set by the forest.
        if not hasattr(self, "classes_"):
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=np.int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                       return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

            self.n_classes_ = np.array(self.n_classes_, dtype=np.int32)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features in ["auto", "sqrt"]:
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn("The min_impurity_split parameter is deprecated. "
                          "Its default value has changed from 1e-7 to 0 in "
                          "version 0.23, and it will be removed in 0.25. "
                          "Use the min_impurity_decrease parameter instead.",
                          FutureWarning)

            if min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        if self.presort != 'deprecated':
            warnings.warn("The parameter 'presort' is deprecated and has no "
                          "effect. It will be removed in v0.24. You can "
                          "suppress this warning by not passing any value "
                          "to the 'presort' parameter.",
                          FutureWarning)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                     self.n_classes_)

        SPLITTERS = DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                random_state)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        max_depth,
                                        self.min_impurity_decrease,
                                        min_impurity_split)

        builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        # Only return the essential data for using a tree for prediction
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        children = np.vstack((self.tree_.children_left,
                              self.tree_.children_right)).T
        value = self.tree_.value

        return feature, threshold, children, value

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):

    @_deprecate_positional_args
    def __init__(self, *,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort='deprecated'):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        return super().fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)

    def predict_proba(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)

        proba = self.tree_.predict(X)
        print(proba)
        proba = proba[:, :self.n_classes_]
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba


class ExtraTreeClassifier(DecisionTreeClassifier):

    @_deprecate_positional_args
    def __init__(self, *,
                 criterion="gini",
                 splitter="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 random_state=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state)


class ShapeletTreeClassifier(BaseEstimator, ClassifierMixin):
    """A shapelet tree classifier."""

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 n_shapelets=10,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 force_dim=None,
                 random_state=None):
        """A shapelet decision tree

        :param max_depth: The maximum depth of the tree. If `None` the
           tree is expanded until all leafs are pure or until all
           leafs contain less than `min_samples_split` samples
           (default: None).

        :param min_samples_split: The minimum number of samples to
           split an internal node (default: 2).拆分内部节点的最小样本数

        :param n_shapelets: The number of shapelets to sample at each
           node (default: 10).每个节点要采样的shapelets集数

        :param min_shapelet_size: The minimum length of a sampled
           shapelet expressed as a fraction, computed as
           `min(ceil(X.shape[-1] * min_shapelet_size), 2)` (default:
           0).

        :param max_shapelet_size: The maximum length of a sampled
           shapelet, expressed as a fraction and computed as
           `ceil(X.shape[-1] * max_shapelet_size)`.

        :param metric: Distance metric used to identify the best
           match. (default: `'euclidean'`)

        :param metric_params: Paramters to the distace measure

        :param force_dim: Force the number of dimensions (default:
           None). If `int`, `force_dim` reshapes the input to the
           shape `[n_samples, force_dim, -1]` to support the
           `BaggingClassifier` interface.

        :param random_state: If `int`, `random_state` is the seed used
           by the random number generator; If `RandomState` instance,
           `random_state` is the random number generator; If `None`,
           the random number generator is the `RandomState` instance
           used by `np.random`.

        """
        if min_shapelet_size < 0 or min_shapelet_size > max_shapelet_size:
            raise ValueError(
                "`min_shapelet_size` {0} <= 0 or {0} > {1}".format(
                    min_shapelet_size, max_shapelet_size))
        if max_shapelet_size > 1:
            raise ValueError(
                "`max_shapelet_size` {0} > 1".format(max_shapelet_size))

        self.max_depth = max_depth
        self.max_depth = max_depth or 2 ** 31
        self.min_samples_split = min_samples_split
        self.random_state = check_random_state(random_state)
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.force_dim = force_dim

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a shapelet tree classifier from the training set (X, y)

        :param X: array-like, shape `[n_samples, n_timesteps]` or
           `[n_samples, n_dimensions, n_timesteps]`. The training time
           series.

        :param y: array-like, shape `[n_samples, n_classes]` or
           `[n_classes]`. Target values (class labels) as integers or
           strings.

        :param sample_weight: If `None`, then samples are equally
            weighted. Splits that would create child nodes with net
            zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would
            result in any single class carrying a negative weight in
            either child node.

        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        :returns: `self`

        """
        random_state = check_random_state(self.random_state)

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions")

        n_samples = X.shape[0]
        if isinstance(self.force_dim, int):
            X = np.reshape(X, [n_samples, self.force_dim, -1])

        n_timesteps = X.shape[-1]

        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        y = np.round(y).astype(int)

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:

            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)
        # print(self.classes_)  # [0 1 2]


        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        metric_params = self.metric_params
        if self.metric_params is None:
            metric_params = {}

        distance_measure = DISTANCE_MEASURE[self.metric](n_timesteps,
                                                         **metric_params)
        # print("tree611 ")
        # print(distance_measure)
        max_shapelet_size = int(n_timesteps * self.max_shapelet_size)
        min_shapelet_size = int(n_timesteps * self.min_shapelet_size)

        if min_shapelet_size < 2:
            min_shapelet_size = 2
        min_sample_split = self.min_samples_split
        self.n_classes_ = len(self.classes_)

        self.n_timestep_ = n_timesteps
        self.n_dims_ = n_dims

        tree_builder = ClassificationShapeletTreeBuilder(
            self.n_shapelets,
            min_shapelet_size,
            max_shapelet_size,
            self.max_depth,
            min_sample_split,
            distance_measure,
            X,
            y,
            sample_weight,
            random_state,
            self.n_classes_,
        )

        self.root_node_ = tree_builder.build_tree()

        return self

    def predict(self, X, check_input=True):
        """Predict the class for X

        :param X: array-like, shape `[n_samples, n_timesteps]` or
            `[n_samples, n_dimensions, n_timesteps]`. The input time
            series.

        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        :returns: array of `shape = [n_samples]`. The predicted
            classes

        """
        # 不进
        return self.classes_[np.argmax(
            self.predict_proba(X, check_input=check_input), axis=1)]

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.  The predicted
        class probability is the fraction of samples of the same class
        in a leaf.

        :param X: array-like, shape `[n_samples, n_timesteps]` or
           `[n_samples, n_dimensions, n_timesteps]`. The input time
           series.

        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        :returns: array of `shape = [n_samples, n_classes]`. The
            class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

        if isinstance(self.force_dim, int):
            X = np.reshape(X, [X.shape[0], self.force_dim, -1])

        if X.shape[-1] != self.n_timestep_:
            raise ValueError("illegal input shape ({} != {})".format(
                X.shape[-1], self.n_timestep_))

        if X.ndim > 2 and X.shape[1] != self.n_dims_:
            raise ValueError("illegal input shape ({} != {}".format(
                X.shape[1], self.n_dims))

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        metric_params = self.metric_params
        if self.metric_params is None:
            metric_params = {}

        distance_measure = DISTANCE_MEASURE[self.metric](self.n_timestep_,
                                                         **metric_params)

        predictor = ClassificationShapeletTreePredictor(
            X, distance_measure, len(self.classes_))
        # print(predictor.predict(self.root_node_))  # [1. 0. 0.]
        return predictor.predict(self.root_node_)


class PairShapeletTreeClassifier(BaseEstimator, ClassifierMixin):
    """A pairshapelet tree classifier."""

    def __init__(self,
                 max_depth=20,
                 min_samples_split=2,
                 n_shapelets=10,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 force_dim=None,
                 random_state=None):
        """A pairshapelet decision tree

        :param max_depth: The maximum depth of the tree. If `None` the
           tree is expanded until all leafs are pure or until all
           leafs contain less than `min_samples_split` samples
           (default: None).

        :param min_samples_split: The minimum number of samples to
           split an internal node (default: 2).拆分内部节点的最小样本数

        :param n_shapelets: The number of shapelets to sample at each
           node (default: 10).每个节点要采样的shapelets集数

        :param min_shapelet_size: The minimum length of a sampled
           shapelet expressed as a fraction, computed as
           `min(ceil(X.shape[-1] * min_shapelet_size), 2)` (default:
           0).

        :param max_shapelet_size: The maximum length of a sampled
           shapelet, expressed as a fraction and computed as
           `ceil(X.shape[-1] * max_shapelet_size)`.

        :param metric: Distance metric used to identify the best
           match. (default: `'euclidean'`)

        :param metric_params: Paramters to the distace measure

        :param force_dim: Force the number of dimensions (default:
           None). If `int`, `force_dim` reshapes the input to the
           shape `[n_samples, force_dim, -1]` to support the
           `BaggingClassifier` interface.

        :param random_state: If `int`, `random_state` is the seed used
           by the random number generator; If `RandomState` instance,
           `random_state` is the random number generator; If `None`,
           the random number generator is the `RandomState` instance
           used by `np.random`.

        """
        if min_shapelet_size < 0 or min_shapelet_size > max_shapelet_size:
            raise ValueError(
                "`min_shapelet_size` {0} <= 0 or {0} > {1}".format(
                    min_shapelet_size, max_shapelet_size))
        if max_shapelet_size > 1:
            raise ValueError(
                "`max_shapelet_size` {0} > 1".format(max_shapelet_size))

        self.max_depth = max_depth
        self.max_depth = max_depth or 20
        self.min_samples_split = min_samples_split
        self.random_state = check_random_state(random_state)
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.force_dim = force_dim

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a pairshapelet tree classifier from the training set (X, y)

        :param X: array-like, shape `[n_samples, n_timesteps]` or
           `[n_samples, n_dimensions, n_timesteps]`. The training time
           series.

        :param y: array-like, shape `[n_samples, n_classes]` or
           `[n_classes]`. Target values (class labels) as integers or
           strings.

        :param sample_weight: If `None`, then samples are equally
            weighted. Splits that would create child nodes with net
            zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would
            result in any single class carrying a negative weight in
            either child node.

        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        :returns: `self`

        """
        random_state = check_random_state(self.random_state)

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions")

        n_samples = X.shape[0]
        if isinstance(self.force_dim, int):
            X = np.reshape(X, [n_samples, self.force_dim, -1])

        n_timesteps = X.shape[-1]

        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        y = np.round(y).astype(int)

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:

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

        metric_params = self.metric_params
        if self.metric_params is None:
            metric_params = {}

        distance_measure = DISTANCE_MEASURE[self.metric](n_timesteps,
                                                         **metric_params)

        max_shapelet_size = int(n_timesteps * self.max_shapelet_size)
        min_shapelet_size = int(n_timesteps * self.min_shapelet_size)

        if min_shapelet_size < 2:
            min_shapelet_size = 2
        min_sample_split = self.min_samples_split
        self.n_classes_ = len(self.classes_)

        self.n_timestep_ = n_timesteps
        self.n_dims_ = n_dims

        tree_builder = ClassificationPairShapeletTreeBuilder(
            self.n_shapelets,
            min_shapelet_size,
            max_shapelet_size,
            self.max_depth,
            min_sample_split,
            distance_measure,
            X,
            y,
            sample_weight,
            random_state,
            self.n_classes_,
        )

        self.root_node_ = tree_builder.build_tree_p()

        return self

    def predict(self, X, check_input=True):
        """Predict the class for X

        :param X: array-like, shape `[n_samples, n_timesteps]` or
            `[n_samples, n_dimensions, n_timesteps]`. The input time
            series.

        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        :returns: array of `shape = [n_samples]`. The predicted
            classes

        """
        return self.classes_[np.argmax(
            self.predict_proba(X, check_input=check_input), axis=1)]

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.  The predicted
        class probability is the fraction of samples of the same class
        in a leaf.

        :param X: array-like, shape `[n_samples, n_timesteps]` or
           `[n_samples, n_dimensions, n_timesteps]`. The input time
           series.

        :param check_input: Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        :returns: array of `shape = [n_samples, n_classes]`. The
            class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

        if isinstance(self.force_dim, int):
            X = np.reshape(X, [X.shape[0], self.force_dim, -1])

        if X.shape[-1] != self.n_timestep_:
            raise ValueError("illegal input shape ({} != {})".format(
                X.shape[-1], self.n_timestep_))

        if X.ndim > 2 and X.shape[1] != self.n_dims_:
            raise ValueError("illegal input shape ({} != {}".format(
                X.shape[1], self.n_dims))

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        metric_params = self.metric_params
        if self.metric_params is None:
            metric_params = {}

        distance_measure = DISTANCE_MEASURE[self.metric](self.n_timestep_,
                                                         **metric_params)

        predictor = ClassificationPairShapeletTreePredictor(
            X, distance_measure, len(self.classes_))

        return predictor.predict(self.root_node_)
