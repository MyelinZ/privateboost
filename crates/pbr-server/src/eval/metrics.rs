//! Classification metrics for the held-out evaluation.
//!
//! `pbr-core` exposes `auc_roc`, `accuracy` and `f1_score` but not precision,
//! recall or log loss. These fill the gap under the same convention
//! `pbr_core::{accuracy, f1_score}` use: `Model::predict` returns a raw
//! log-odds score, and a row counts as positive when
//! `sigmoid(score) >= threshold`. They call `pbr_core::metrics::sigmoid` rather
//! than redefining it, so the two cannot drift.

use pbr_core::metrics::sigmoid;
use pbr_core::{Model, accuracy, auc_roc, f1_score};
use serde::Serialize;

/// Decision threshold applied to `sigmoid(score)`, matching what `pbr-core`'s
/// heart_disease tests pass to `accuracy`/`f1_score`. Every per-tree document
/// records it as `thresholdUsed`, so a reader knows the operating point.
pub const THRESHOLD: f64 = 0.5;

/// Clamp on a predicted probability before it enters `ln`, so a confidently
/// wrong prediction yields a large but finite loss instead of infinity.
const LOGLOSS_EPS: f64 = 1e-15;

/// The six quality numbers scored for one partial model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TreeMetrics {
    pub auc: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub logloss: f64,
}

/// `(true positives, false positives, false negatives)` at `threshold` on the
/// sigmoid of each raw score. Shared by `precision` and `recall`, so both count
/// a row exactly as `f1_score` does.
fn confusion(predictions: &[f64], targets: &[f64], threshold: f64) -> (u64, u64, u64) {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    for (&p, &t) in predictions.iter().zip(targets) {
        let positive = sigmoid(p) >= threshold;
        if positive && t == 1.0 {
            tp += 1;
        } else if positive && t == 0.0 {
            fp += 1;
        } else if !positive && t == 1.0 {
            fn_ += 1;
        }
    }
    (tp, fp, fn_)
}

/// tp / (tp + fp); 0 when nothing is predicted positive.
pub fn precision(predictions: &[f64], targets: &[f64], threshold: f64) -> f64 {
    let (tp, fp, _) = confusion(predictions, targets, threshold);
    if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    }
}

/// tp / (tp + fn); 0 when there are no positive labels.
pub fn recall(predictions: &[f64], targets: &[f64], threshold: f64) -> f64 {
    let (tp, _, fn_) = confusion(predictions, targets, threshold);
    if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    }
}

/// Mean binary cross-entropy of `sigmoid(score)` against 0/1 targets. Empty
/// input scores 0.0.
pub fn logloss(predictions: &[f64], targets: &[f64]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let sum: f64 = predictions
        .iter()
        .zip(targets)
        .map(|(&p, &t)| {
            let q = sigmoid(p).clamp(LOGLOSS_EPS, 1.0 - LOGLOSS_EPS);
            -(t * q.ln() + (1.0 - t) * (1.0 - q).ln())
        })
        .sum();
    sum / targets.len() as f64
}

/// Score `model` on the held-out split: raw log-odds via `Model::predict`,
/// then the six metrics under [`THRESHOLD`].
pub fn evaluate(model: &Model, features: &[Vec<f64>], targets: &[f64]) -> TreeMetrics {
    let preds = model.predict(features);
    TreeMetrics {
        auc: auc_roc(&preds, targets),
        accuracy: accuracy(&preds, targets, THRESHOLD),
        precision: precision(&preds, targets, THRESHOLD),
        recall: recall(&preds, targets, THRESHOLD),
        f1: f1_score(&preds, targets, THRESHOLD),
        logloss: logloss(&preds, targets),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Raw log-odds toy scores: sigmoid(2.0) = 0.88 >= 0.5 is positive,
    // sigmoid(-2.0) = 0.12 < 0.5 is negative.

    #[test]
    fn precision_recall_perfect() {
        let preds = [2.0, -2.0, 2.0, -2.0];
        let targets = [1.0, 0.0, 1.0, 0.0];
        assert!((precision(&preds, &targets, THRESHOLD) - 1.0).abs() < 1e-12);
        assert!((recall(&preds, &targets, THRESHOLD) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn precision_recall_with_one_error_each() {
        // idx0 pos/1 = tp, idx1 pos/0 = fp, idx2 neg/1 = fn, idx3 neg/0 = tn.
        let preds = [2.0, 2.0, -2.0, -2.0];
        let targets = [1.0, 0.0, 1.0, 0.0];
        assert!((precision(&preds, &targets, THRESHOLD) - 0.5).abs() < 1e-12);
        assert!((recall(&preds, &targets, THRESHOLD) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn precision_zero_when_no_positive_predicted() {
        let preds = [-2.0, -2.0];
        let targets = [1.0, 0.0];
        assert_eq!(precision(&preds, &targets, THRESHOLD), 0.0);
    }

    #[test]
    fn recall_zero_when_no_positive_labels() {
        let preds = [2.0, -2.0];
        let targets = [0.0, 0.0];
        assert_eq!(recall(&preds, &targets, THRESHOLD), 0.0);
    }

    #[test]
    fn logloss_of_half_probabilities_is_ln2() {
        // sigmoid(0) = 0.5, so every term is -ln(0.5) = ln 2 regardless of label.
        let preds = [0.0, 0.0, 0.0];
        let targets = [1.0, 0.0, 1.0];
        assert!((logloss(&preds, &targets) - std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn logloss_of_confident_correct_is_near_zero() {
        let preds = [10.0, -10.0];
        let targets = [1.0, 0.0];
        assert!(logloss(&preds, &targets) < 1e-3);
    }

    #[test]
    fn logloss_clamps_confident_wrong_to_finite() {
        // Without the clamp sigmoid(-40) underflows and ln(0) is -inf.
        let preds = [-40.0];
        let targets = [1.0];
        assert!(logloss(&preds, &targets).is_finite());
    }
}
