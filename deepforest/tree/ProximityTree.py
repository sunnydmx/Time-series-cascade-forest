from random import random
from . import Node
# from Pforests.core.TreeStatCollector import TreeStatCollector
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from . import FileReader
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_array

"""
A tree has:
- id
- root
- 
"""


class ProximityTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 max_depth=2 ** 31,
                 n_classes=0,
                 random_state=1,
                 ):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.random_state = check_random_state(random_state)
        self.random = random()
        self.node_counter = 0
        self.tree_depth = 0
        self.time_best_splits = 0
        self.root = None
        # self.id = id
        # if forest is not None:
        #     self.proximity_forest_id = forest.get_forest_ID()
        #     self.stats = TreeStatCollector(id, self.proximity_forest_id)

    def get_root_node(self):
        return self.root

    def fit(self, X, y, sample_weight=None, check_input=True):
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions")
        # print(X)
        n_samples = X.shape[0]
        n_timesteps = X.shape[-1]

        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        y = np.round(y).astype(int)

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



        self.n_timestep_ = n_timesteps
        self.n_dims_ = n_dims

        # print(y)
        # print(self.classes_)
        data = FileReader.FileReader.load_data(X, y)
        # print(data)
        self.node_counter = self.node_counter + 1
        # print("node")
        self.root = Node.Node(parent=None, label=None, node_id=self.node_counter, depth=self.tree_depth, tree=self)
        self.root.train(data)

    def predict_proba(self, X, check_input=True):
        # print("jinzheli")
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

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

        n_samples = X.shape[0]
        # self.n_classes_ = len(self.classes_)
        # print(self.classes_)
        output = np.zeros([n_samples, self.n_classes], dtype=np.float64)
        # print(output)
        for i in range(n_samples):
            node = self.root
            if node is None:
                return -1
            while not node.is_leaf:
                posicion = node.splitter.find_closest_branch_(X[i])
                if posicion == -1:
                    node.is_leaf = True
                    continue
                node = node.children[posicion]
            label = node.label
            output[i][label] = 1
        # print(output[1,:])
        return output

    def get_treestat_collection(self):
        self.stats.collate_results(self)
        return self.stats

    def get_num_nodes(self):
        return self._get_num_nodes(self.root)

    def _get_num_nodes(self, node):
        count = 0
        if node.children is None:
            return 1
        for i in range(0, len(node.children)):
            count = count + self._get_num_nodes(node.children[i])
        return count + 1

    def get_num_leaves(self):
        return self._get_num_leaves(self.root)

    def _get_num_leaves(self, n):
        count = 0
        if n.children is None or n.children == 0:
            return 1
        for i in range(0, len(n.children)):
            count = count + self._get_num_leaves(n.children[i])
        return count

    def get_num_internal_node(self):
        return self._get_num_internal_node(self.root)

    def _get_num_internal_node(self, n):
        count = 0
        if n.children is None:
            return 0
        for i in range(0, len(n.children)):
            count = count + self._get_num_internal_node(n.childen[i])

        return count + 1

    def get_height(self):
        return self._get_height(self.root)

    def _get_height(self, n):
        max_depth = 0
        if n.children is None or n.children == 0:
            return 0
        for i in range(0, len(n.children)):
            max_depth = max(max_depth, self._get_height(n.children[i]))
        return max_depth + 1

    def get_min_depth(self, node):
        max_depth = 0
        if node.children is not None:
            return 0
        for i in range(0, len(node.children)):
            max_depth = min(max_depth, self._get_height(node.children[i]))
        return max_depth + 1

