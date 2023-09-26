"""A wrapper on the base estimator for the naming consistency."""


__all__ = ["Estimator"]

from .forest import RandomForestClassifiers, ExtraTreesClassifier, ShapeletForestClassifier, PairShapeletForestClassifier, ProximityForestClassifier, DrCIF, SupervisedTimeSeriesForest
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def make_estimator(
    name,
    criterion,
    n_trees=20,
    max_depth=None,
    min_samples_leaf=1,
    backend="custom",
    n_jobs=None,
    random_state=None
):
    # RandomForestClassifier
    if name == "rf":
        estimator = RandomForestClassifiers(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    # ExtraTreesClassifier
    elif name == "erf":
        estimator = ExtraTreesClassifier(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state
        )
    elif name == "srf":
        estimator = ShapeletForestClassifier(
            # n_estimators=n_trees, n_shapelets=15, metric="scaled_dtw", metric_params={"r": 0.1}, max_depth=2 ** 31
            n_estimators=n_trees, n_shapelets=1, metric="euclidean", metric_params=None, max_depth=2**31
            # n_estimators=n_trees, n_shapelets=None, metric="scaled_dtw", metric_params={"r": 0.1}, max_depth=2 ** 31
        )
    elif name == "pairsrf":
        estimator = PairShapeletForestClassifier(
            # n_estimators=n_trees, n_shapelets=1, metric="scaled_dtw", metric_params={"r": 0.1}, max_depth=2**31
            n_estimators=n_trees, n_shapelets=20, metric="euclidean", metric_params=None, max_depth=20
        )
    elif name == "proximity forest":
        # estimator = PairShapeletForestClassifier(
        #     # n_estimators=n_trees, n_shapelets=1, metric="scaled_dtw", metric_params={"r": 0.1}, max_depth=2**31
        #     n_estimators=n_trees, n_shapelets=20, metric="euclidean", metric_params=None, max_depth=20
        # )
        print(n_trees)
        estimator = ProximityForestClassifier(n_estimators=n_trees, max_depth=2**31)
    elif name == "DrCIF":
        print(n_trees)
        estimator = DrCIF(n_estimators=n_trees)
    elif name == "STSF":
        print(n_trees)
        estimator = SupervisedTimeSeriesForest(n_estimators=n_trees)
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


class Estimator(object):

    def __init__(
        self,
        name,
        criterion,
        n_trees=20,
        max_depth=None,
        min_samples_leaf=1,
        backend="custom",
        n_jobs=None,
        random_state=None
    ):
        self.estimator_ = make_estimator(name,
                                         criterion,
                                         n_trees,
                                         max_depth,
                                         min_samples_leaf,
                                         backend,
                                         n_jobs,
                                         random_state)
        self.name = name

    @property
    def feature_importances_(self):
        """Return the impurity-based feature importances from the estimator."""

        return self.estimator_.feature_importances_

    def oob_decision_function_(self):
        return self.estimator_.oob_decision_function_

    def fit_transform(self, X, y):
        self.estimator_.fit(X, y)
        if self.name == 'DrCIF':
            X_aug = self.estimator_._get_train_probs(X, y)
        else:
            X_aug = self.estimator_.oob_decision_function_

        return X_aug

    def transform(self, X):

        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict_proba(X)
