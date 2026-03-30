"""Tree and Model classes for prediction."""

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np


@dataclass
class Leaf:
    """A leaf node with its prediction value."""

    value: float
    n_samples: int


@dataclass
class SplitNode:
    """A split node in the decision tree."""

    feature_idx: int
    threshold: float
    gain: float
    left: "TreeNode"
    right: "TreeNode"


TreeNode = Union[SplitNode, Leaf]


@dataclass
class Tree:
    """A single decision tree."""

    root: TreeNode

    def predict_one(self, features: np.ndarray) -> float:
        """Predict for a single sample."""
        node = self.root
        while True:
            match node:
                case SplitNode(feature_idx=f, threshold=t, left=l, right=r):
                    node = l if features[f] <= t else r
                case Leaf(value=v):
                    return v


@dataclass
class Model:
    """Ensemble of trees for prediction."""

    initial_prediction: float
    learning_rate: float
    trees: List[Tree] = field(default_factory=list)

    def add_tree(self, tree: Tree) -> None:
        """Add a tree to the ensemble."""
        self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples."""
        preds = np.full(len(X), self.initial_prediction)
        for i, features in enumerate(X):
            for tree in self.trees:
                preds[i] += self.learning_rate * tree.predict_one(features)
        return preds

    def predict_one(self, features: np.ndarray) -> float:
        """Predict for a single sample."""
        pred = self.initial_prediction
        for tree in self.trees:
            pred += self.learning_rate * tree.predict_one(features)
        return pred
