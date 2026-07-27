pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn accuracy(predictions: &[f64], targets: &[f64], threshold: f64) -> f64 {
    let correct = predictions
        .iter()
        .zip(targets)
        .filter(|&(&p, &t)| {
            let class = if sigmoid(p) >= threshold { 1.0 } else { 0.0 };
            (class - t).abs() < 1e-10
        })
        .count();
    correct as f64 / targets.len() as f64
}

pub fn f1_score(predictions: &[f64], targets: &[f64], threshold: f64) -> f64 {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_count = 0u64;

    for (&p, &t) in predictions.iter().zip(targets) {
        let predicted = if sigmoid(p) >= threshold { 1.0 } else { 0.0 };
        if predicted == 1.0 && t == 1.0 {
            tp += 1;
        } else if predicted == 1.0 && t == 0.0 {
            fp += 1;
        } else if predicted == 0.0 && t == 1.0 {
            fn_count += 1;
        }
    }

    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };

    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

pub fn auc_roc(predictions: &[f64], targets: &[f64]) -> f64 {
    let mut scored: Vec<(f64, f64)> = predictions
        .iter()
        .map(|&p| sigmoid(p))
        .zip(targets.iter().copied())
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = targets.iter().filter(|&&t| t == 1.0).count() as f64;
    let n_neg = targets.iter().filter(|&&t| t == 0.0).count() as f64;

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_fpr = 0.0;
    let mut prev_tpr = 0.0;
    let mut i = 0;

    while i < scored.len() {
        // Collect all points with the same score (tied group)
        let score = scored[i].0;
        let mut j = i;
        while j < scored.len() && scored[j].0 == score {
            if scored[j].1 == 1.0 { tp += 1.0; } else { fp += 1.0; }
            j += 1;
        }
        let tpr = tp / n_pos;
        let fpr = fp / n_neg;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_fpr = fpr;
        prev_tpr = tpr;
        i = j;
    }

    auc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let preds = vec![2.0, -2.0, 2.0, -2.0];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        assert!((accuracy(&preds, &targets, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_half() {
        let preds = vec![2.0, 2.0, -2.0, -2.0];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        assert!((accuracy(&preds, &targets, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_f1_perfect() {
        let preds = vec![2.0, -2.0, 2.0, -2.0];
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        assert!((f1_score(&preds, &targets, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_no_positives_predicted() {
        let preds = vec![-2.0, -2.0];
        let targets = vec![1.0, 0.0];
        assert!((f1_score(&preds, &targets, 0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_auc_perfect() {
        let preds = vec![2.0, 1.0, -1.0, -2.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        assert!((auc_roc(&preds, &targets) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_random() {
        let preds = vec![-2.0, -1.0, 1.0, 2.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        assert!((auc_roc(&preds, &targets) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_coin_flip() {
        let preds = vec![0.0, 0.0, 0.0, 0.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        assert!((auc_roc(&preds, &targets) - 0.5).abs() < 1e-10);
    }
}
