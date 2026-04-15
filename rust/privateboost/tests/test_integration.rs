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
        BinMethod::Gaussian,
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
        let shares = client.compute_stat_shares().unwrap();
        for (i, share) in shares.into_iter().enumerate() {
            aggregator.shareholders_mut()[i].receive_stats(share);
        }
    }
    aggregator.define_bins().unwrap();

    println!("Training clients: {}", aggregator.n_clients());
    println!(
        "Initial prediction (target mean): {:.4}",
        aggregator.feature_stats().unwrap().last().unwrap().mean
    );

    // Phase 2: Tree training
    for _ in 0..n_trees {
        for _ in 0..max_depth {
            let ctx = aggregator.round_context();
            for client in &mut clients {
                let shares = client
                    .compute_gradient_shares(&ctx, &Loss::Logistic, false)
                    .unwrap();
                for (i, share) in shares.into_iter().enumerate() {
                    aggregator.shareholders_mut()[i].receive_gradients(share);
                }
            }
            if !aggregator.compute_splits(0.0, 1.0).unwrap() {
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
fn test_xgboost_heart_disease_path_hiding() {
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
    let n_shareholders = 3;
    let n_trees = 15;
    let max_depth = 3;

    let shareholders: Vec<ShareHolder> = (0..n_shareholders)
        .map(|i| ShareHolder::new(i, (i + 1) as u64, min_clients))
        .collect();

    let mut aggregator = Aggregator::new(shareholders, 10, threshold, min_clients, 0.15, 2.0, BinMethod::Gaussian).unwrap();

    let mut clients: Vec<Client> = train_features
        .iter()
        .zip(train_targets.iter())
        .enumerate()
        .map(|(idx, (feats, &target))| {
            Client::new(format!("client_{}", idx), feats.clone(), target, n_shareholders, threshold, None)
        })
        .collect();

    for client in &mut clients {
        let shares = client.compute_stat_shares().unwrap();
        for (i, share) in shares.into_iter().enumerate() {
            aggregator.shareholders_mut()[i].receive_stats(share);
        }
    }
    aggregator.define_bins().unwrap();

    for _ in 0..n_trees {
        for _ in 0..max_depth {
            let ctx = aggregator.round_context();
            for client in &mut clients {
                let shares = client
                    .compute_gradient_shares(&ctx, &Loss::Logistic, true)
                    .unwrap();
                for share in shares {
                    let sh_idx = share.share.x as usize - 1;
                    aggregator.shareholders_mut()[sh_idx].receive_gradients(share);
                }
            }
            if !aggregator.compute_splits(0.0, 1.0).unwrap() {
                break;
            }
        }
        aggregator.finish_round();
    }

    let predictions = aggregator.model().predict(&test_features.to_vec());
    let correct: usize = predictions
        .iter()
        .zip(test_targets.iter())
        .filter(|&(&pred, &target)| {
            let class = if pred >= 0.5 { 1.0 } else { 0.0 };
            (class - target).abs() < 1e-10
        })
        .count();
    let accuracy = correct as f64 / test_targets.len() as f64;
    println!("Path hiding accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy >= 0.75, "Expected >=75% with path hiding, got {:.2}%", accuracy * 100.0);
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
        Aggregator::new(shareholders, 10, threshold, min_clients, 0.1, 1.0, BinMethod::Gaussian).unwrap();

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
        let shares = client.compute_stat_shares().unwrap();
        for (i, share) in shares.into_iter().enumerate() {
            aggregator.shareholders_mut()[i].receive_stats(share);
        }
    }

    let result = aggregator.define_bins();
    assert!(result.is_err());
}

#[test]
fn test_gaussian_binning_edges_denser_at_center() {
    let n = 200;
    // Linearly spaced data from -3 to 3 (uniform, but mean≈0, std≈1.73)
    let features: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![-3.0 + 6.0 * (i as f64) / (n as f64 - 1.0)])
        .collect();
    let targets: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();

    let n_shareholders = 3;
    let threshold = 2;
    let min_clients = 5;
    let n_bins = 10;

    let shareholders: Vec<ShareHolder> = (0..n_shareholders)
        .map(|i| ShareHolder::new(i, (i + 1) as u64, min_clients))
        .collect();

    let mut aggregator = Aggregator::new(
        shareholders, n_bins, threshold, min_clients, 0.1, 1.0, BinMethod::Gaussian,
    ).unwrap();

    let mut clients: Vec<Client> = features.iter().zip(targets.iter()).enumerate()
        .map(|(i, (f, &t))| Client::new(format!("c{i}"), f.clone(), t, n_shareholders, threshold, None))
        .collect();

    for client in &mut clients {
        let shares = client.compute_stat_shares().unwrap();
        for (i, share) in shares.into_iter().enumerate() {
            aggregator.shareholders_mut()[i].receive_stats(share);
        }
    }

    let configs = aggregator.define_bins().unwrap();
    let config = &configs[0];

    // Gaussian quantile edges should be more concentrated near the center
    let inner = &config.edges[1..config.edges.len() - 1];
    let mid = inner.len() / 2;
    let center_gap = (inner[mid] - inner[mid - 1]).abs();
    let outer_gap = (inner[1] - inner[0]).abs();
    assert!(
        center_gap < outer_gap,
        "Gaussian edges should be denser near center: center_gap={center_gap}, outer_gap={outer_gap}"
    );
}
