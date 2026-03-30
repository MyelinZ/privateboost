pub enum TreeNode {
    Split {
        feature_idx: usize,
        threshold: f64,
        gain: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    Leaf {
        value: f64,
    },
}

pub struct Tree {
    pub root: TreeNode,
}

impl Tree {
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        let mut node = &self.root;
        loop {
            match node {
                TreeNode::Split {
                    feature_idx,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    node = if features[*feature_idx] <= *threshold {
                        left
                    } else {
                        right
                    };
                }
                TreeNode::Leaf { value, .. } => return *value,
            }
        }
    }
}

pub struct Model {
    pub initial_prediction: f64,
    pub learning_rate: f64,
    pub trees: Vec<Tree>,
}

impl Model {
    pub fn new(initial_prediction: f64, learning_rate: f64) -> Self {
        Self {
            initial_prediction,
            learning_rate,
            trees: Vec::new(),
        }
    }

    pub fn add_tree(&mut self, tree: Tree) {
        self.trees.push(tree);
    }

    pub fn predict_one(&self, features: &[f64]) -> f64 {
        let mut pred = self.initial_prediction;
        for tree in &self.trees {
            pred += self.learning_rate * tree.predict_one(features);
        }
        pred
    }

    pub fn predict(&self, features_matrix: &[Vec<f64>]) -> Vec<f64> {
        features_matrix
            .iter()
            .map(|f| self.predict_one(f))
            .collect()
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Model(initial={:.4}, trees={}, lr={})",
            self.initial_prediction,
            self.trees.len(),
            self.learning_rate
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> Tree {
        Tree {
            root: TreeNode::Split {
                feature_idx: 0,
                threshold: 5.0,
                gain: 1.0,
                left: Box::new(TreeNode::Leaf {
                    value: 1.0,
                }),
                right: Box::new(TreeNode::Leaf {
                    value: -1.0,
                }),
            },
        }
    }

    #[test]
    fn test_tree_predict_left() {
        let tree = simple_tree();
        assert_eq!(tree.predict_one(&[3.0, 0.0]), 1.0);
    }

    #[test]
    fn test_tree_predict_right() {
        let tree = simple_tree();
        assert_eq!(tree.predict_one(&[7.0, 0.0]), -1.0);
    }

    #[test]
    fn test_tree_predict_boundary() {
        let tree = simple_tree();
        assert_eq!(tree.predict_one(&[5.0, 0.0]), 1.0); // <= goes left
    }

    #[test]
    fn test_model_predict() {
        let mut model = Model::new(0.5, 0.1);
        model.add_tree(simple_tree());
        assert!((model.predict_one(&[3.0]) - 0.6).abs() < 1e-10);
        assert!((model.predict_one(&[7.0]) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_model_predict_batch() {
        let mut model = Model::new(0.5, 0.1);
        model.add_tree(simple_tree());
        let preds = model.predict(&[vec![3.0], vec![7.0]]);
        assert!((preds[0] - 0.6).abs() < 1e-10);
        assert!((preds[1] - 0.4).abs() < 1e-10);
    }
}
