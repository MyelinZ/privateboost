use privateboost::*;
use std::path::Path;

#[test]
fn test_xgboost_heart_disease_shamir() {
    // CSV is pre-shuffled with numpy random_state=42 to match Python test split
    let dataset = read_csv(Path::new("tests/data/heart_disease.csv"), "target").unwrap();

    let features = dataset.features;
    let targets = dataset.targets;

    let split_idx = (features.len() as f64 * 0.8) as usize;
    let train_features = &features[..split_idx];
    let train_targets = &targets[..split_idx];
    let test_features = &features[split_idx..];
    let test_targets = &targets[split_idx..];

    let threshold = 2;
    let min_clients = 10;
    let learning_rate = 0.15;
    let lambda_reg = 2.0;
    let n_bins = 10;
    let n_shareholders = 3;
    let n_trees = 15;
    let max_depth = 3;

    let shareholders: Vec<ShareHolder> = (0..n_shareholders)
        .map(|i| ShareHolder::new(i, (i + 1) as u64, min_clients))
        .collect();

    let mut aggregator = Aggregator::new(
        shareholders,
        n_bins,
        threshold,
        min_clients,
        learning_rate,
        lambda_reg,
    )
    .unwrap();

    let mut clients: Vec<Client> = train_features
        .iter()
        .zip(train_targets.iter())
        .enumerate()
        .map(|(idx, (feats, &target))| {
            Client::new(
                format!("client_{}", idx),
                feats.clone(),
                target,
                n_shareholders,
                threshold,
                None,
            )
        })
        .collect();

    // Phase 1: Statistics
    for client in &mut clients {
        let shares = client.submit_stats().unwrap();
        for (i, share) in shares.into_iter().enumerate() {
            aggregator.shareholders_mut()[i].receive_stats(share);
        }
    }
    let bins = aggregator.define_bins().unwrap();

    println!("Training clients: {}", aggregator.n_clients());
    println!(
        "Initial prediction (target mean): {:.4}",
        aggregator.means().unwrap().last().unwrap()
    );

    // Phase 2: Tree training
    for round_id in 0..n_trees {
        for depth in 0..max_depth {
            for client in &mut clients {
                let shares = client
                    .submit_gradients(
                        &bins,
                        aggregator.model(),
                        aggregator.splits(),
                        round_id as u64,
                        depth,
                        &Loss::Logistic,
                    )
                    .unwrap();
                for (i, share) in shares.into_iter().enumerate() {
                    aggregator.shareholders_mut()[i].receive_gradients(share);
                }
            }
            if !aggregator.compute_splits(depth, 0.0, 1.0).unwrap() {
                break;
            }
        }
        aggregator.finish_round();
    }

    // Phase 3: Prediction
    let test_features_vec: Vec<Vec<f64>> = test_features.to_vec();
    let predictions = aggregator.model().predict(&test_features_vec);

    // Threshold at 0.5 on raw log-odds to match Python reference behavior
    let correct: usize = predictions
        .iter()
        .zip(test_targets.iter())
        .filter(|&(&pred, &target)| {
            let class = if pred >= 0.5 { 1.0 } else { 0.0 };
            (class - target).abs() < 1e-10
        })
        .count();

    let accuracy = correct as f64 / test_targets.len() as f64;
    println!("Test accuracy: {:.2}%", accuracy * 100.0);
    assert!(
        accuracy >= 0.75,
        "Expected >=75% accuracy, got {:.2}%",
        accuracy * 100.0
    );
}

#[test]
fn test_min_clients_enforcement() {
    let min_clients = 10;
    let n_shareholders = 3;
    let threshold = 2;

    let shareholders: Vec<ShareHolder> = (0..n_shareholders)
        .map(|i| ShareHolder::new(i, (i + 1) as u64, min_clients))
        .collect();

    let mut aggregator =
        Aggregator::new(shareholders, 10, threshold, min_clients, 0.1, 1.0).unwrap();

    let mut clients: Vec<Client> = (0..5)
        .map(|i| {
            Client::new(
                format!("c{}", i),
                vec![1.0, 2.0],
                0.0,
                n_shareholders,
                threshold,
                None,
            )
        })
        .collect();

    for client in &mut clients {
        let shares = client.submit_stats().unwrap();
        for (i, share) in shares.into_iter().enumerate() {
            aggregator.shareholders_mut()[i].receive_stats(share);
        }
    }

    let result = aggregator.define_bins();
    assert!(result.is_err());
}
