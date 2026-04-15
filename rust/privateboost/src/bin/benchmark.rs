use clap::Parser;
use privateboost::{
    Aggregator, BinMethod, Client, CommittedGradientShare, CommittedStatsShare, Direction, Loss,
    ShareHolder, accuracy, auc_roc, f1_score, read_csv, stratified_split,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "benchmark", about = "PrivateBoost benchmark runner")]
struct Args {
    #[arg(long)]
    dataset: PathBuf,

    #[arg(long, default_value = "target")]
    target_column: String,

    #[arg(long, default_value_t = 15)]
    n_trees: usize,

    #[arg(long, default_value_t = 3)]
    max_depth: usize,

    #[arg(long, default_value_t = 0.15)]
    learning_rate: f64,

    #[arg(long, default_value_t = 2.0)]
    lambda: f64,

    #[arg(long, default_value_t = 10)]
    n_bins: usize,

    #[arg(long, default_value_t = 3)]
    n_shareholders: usize,

    #[arg(long, default_value_t = 2)]
    threshold: usize,

    #[arg(long, default_value_t = 10)]
    min_clients: usize,

    #[arg(long, default_value_t = 0.2)]
    test_split: f64,

    #[arg(long, default_value_t = 5)]
    n_splits: usize,

    #[arg(long, default_value_t = false)]
    hide_path: bool,

    #[arg(long, default_value_t = 0.0)]
    dropout_rate: f64,

    /// Run dropout sweep: test rates 0.0, 0.1, ..., 0.9 and write dropout.csv
    #[arg(long, default_value_t = false)]
    dropout_sweep: bool,

    /// Binning method: "gaussian" (default) or "uniform"
    #[arg(long, default_value = "gaussian")]
    bin_method: String,

    #[arg(long, default_value = "results")]
    output_dir: PathBuf,
}

fn main() -> privateboost::Result<()> {
    let args = Args::parse();

    let bin_method = match args.bin_method.as_str() {
        "gaussian" => BinMethod::Gaussian,
        "uniform" => BinMethod::Uniform,
        other => {
            eprintln!("Unknown bin method: {other}, using gaussian");
            BinMethod::Gaussian
        }
    };

    std::fs::create_dir_all(&args.output_dir)?;

    let dataset = read_csv(&args.dataset, &args.target_column)?;
    let n_features = dataset.feature_names.len();

    eprintln!(
        "Loaded dataset: {} samples, {} features",
        dataset.targets.len(),
        n_features
    );

    let mut metrics_wtr =
        csv::Writer::from_path(args.output_dir.join("metrics.csv"))?;
    metrics_wtr.write_record([
        "split_id", "accuracy", "auc_roc", "f1", "n_train", "n_test", "n_features",
    ])?;

    let mut totals_wtr =
        csv::Writer::from_path(args.output_dir.join("traffic_totals.csv"))?;
    totals_wtr.write_record([
        "split_id",
        "c2s_bytes",
        "s2a_bytes",
        "broadcast_bytes",
        "total_bytes",
        "hide_path",
    ])?;

    let mut per_round_wtr =
        csv::Writer::from_path(args.output_dir.join("traffic_per_round.csv"))?;
    per_round_wtr.write_record([
        "split_id",
        "tree",
        "depth",
        "direction",
        "message_type",
        "bytes",
        "count",
    ])?;

    let mut curve_wtr =
        csv::Writer::from_path(args.output_dir.join("learning_curve.csv"))?;
    curve_wtr.write_record([
        "split_id", "n_trees", "accuracy", "auc_roc", "f1",
    ])?;

    for split_id in 0..args.n_splits {
        eprintln!("Running split {}/{}", split_id + 1, args.n_splits);

        let split_data = stratified_split(&dataset.targets, args.test_split, split_id as u64);

        // Export split indices for reproducibility with external baselines
        let splits_dir = args.output_dir.join("splits");
        std::fs::create_dir_all(&splits_dir)?;
        let mut split_wtr = csv::Writer::from_path(
            splits_dir.join(format!("split_{}.csv", split_id)),
        )?;
        split_wtr.write_record(["index", "set"])?;
        for &i in &split_data.train_indices {
            split_wtr.write_record(&[i.to_string(), "train".to_string()])?;
        }
        for &i in &split_data.test_indices {
            split_wtr.write_record(&[i.to_string(), "test".to_string()])?;
        }
        split_wtr.flush()?;

        let train_features: Vec<Vec<f64>> = split_data
            .train_indices
            .iter()
            .map(|&i| dataset.features[i].clone())
            .collect();
        let train_targets: Vec<f64> = split_data
            .train_indices
            .iter()
            .map(|&i| dataset.targets[i])
            .collect();
        let test_features: Vec<Vec<f64>> = split_data
            .test_indices
            .iter()
            .map(|&i| dataset.features[i].clone())
            .collect();
        let test_targets: Vec<f64> = split_data
            .test_indices
            .iter()
            .map(|&i| dataset.targets[i])
            .collect();

        let n_train = train_features.len();
        let n_test = test_features.len();

        let shareholders: Vec<ShareHolder> = (0..args.n_shareholders)
            .map(|i| ShareHolder::new(i, (i + 1) as u64, args.min_clients))
            .collect();

        let mut aggregator = Aggregator::new(
            shareholders,
            args.n_bins,
            args.threshold,
            args.min_clients,
            args.learning_rate,
            args.lambda,
            bin_method,
        )?;
        aggregator.enable_traffic_log();

        let mut clients: Vec<Client> = train_features
            .iter()
            .zip(train_targets.iter())
            .enumerate()
            .map(|(idx, (feats, &target))| {
                Client::new(
                    format!("c{}_{}", split_id, idx),
                    feats.clone(),
                    target,
                    args.n_shareholders,
                    args.threshold,
                    Some((split_id * 100_000 + idx) as u64),
                )
            })
            .collect();

        // Stats phase (parallel)
        let n_clients = clients.len() as u64;
        let stat_shares: Vec<Vec<CommittedStatsShare>> = clients
            .par_iter_mut()
            .map(|client| client.compute_stat_shares().unwrap())
            .collect();

        let mut stats_recorded = false;
        for shares in stat_shares {
            if !stats_recorded {
                stats_recorded = true;
                if let Some(log) = aggregator.traffic_log_mut()
                    && let Some(share) = shares.first() {
                        log.record(share, Direction::ClientToShareholder, 0, 0,
                                   "CommittedStatsShare",
                                   shares.len() as u64 * n_clients);
                    }
            }
            for (i, share) in shares.into_iter().enumerate() {
                aggregator.shareholders_mut()[i].receive_stats(share);
            }
        }
        aggregator.define_bins()?;

        // Tree training
        for tree in 0..args.n_trees {
            for depth in 0..args.max_depth {
                let ctx = aggregator.round_context();

                // Client participation (dropout)
                let mut round_rng = StdRng::seed_from_u64(
                    (split_id * 1000 + tree * 100 + depth) as u64
                );
                let mut participating: Vec<usize> = (0..clients.len()).collect();
                if args.dropout_rate > 0.0 {
                    let n_keep = ((1.0 - args.dropout_rate) * clients.len() as f64).round() as usize;
                    participating.shuffle(&mut round_rng);
                    participating.truncate(n_keep.max(1));
                }

                // Parallel gradient computation
                let participating_set: HashSet<usize> = participating.iter().copied().collect();
                let hide_path = args.hide_path;
                let all_shares: Vec<Vec<CommittedGradientShare>> = clients
                    .par_iter_mut()
                    .enumerate()
                    .map(|(ci, client)| {
                        if participating_set.contains(&ci) {
                            client.compute_gradient_shares(&ctx, &Loss::Logistic, hide_path).unwrap()
                        } else {
                            Vec::new()
                        }
                    })
                    .collect();

                let mut grad_recorded = false;
                let n_participating = participating.len() as u64;
                for shares in all_shares {
                    if shares.is_empty() { continue; }
                    if !grad_recorded {
                        grad_recorded = true;
                        if let Some(log) = aggregator.traffic_log_mut()
                            && let Some(share) = shares.first() {
                                log.record(share, Direction::ClientToShareholder, tree, depth,
                                           "CommittedGradientShare",
                                           shares.len() as u64 * n_participating);
                            }
                    }
                    if hide_path {
                        for share in shares {
                            let sh_idx = share.share.x as usize - 1;
                            aggregator.shareholders_mut()[sh_idx].receive_gradients(share);
                        }
                    } else {
                        for (i, share) in shares.into_iter().enumerate() {
                            aggregator.shareholders_mut()[i].receive_gradients(share);
                        }
                    }
                }

                if !aggregator.compute_splits(0.0, 1.0)? {
                    break;
                }
            }
            aggregator.finish_round();

            // Per-tree evaluation for learning curve
            let curve_preds = aggregator.model().predict(&test_features);
            let curve_acc = accuracy(&curve_preds, &test_targets, 0.5);
            let curve_auc = auc_roc(&curve_preds, &test_targets);
            let curve_f1 = f1_score(&curve_preds, &test_targets, 0.5);
            curve_wtr.write_record(&[
                split_id.to_string(),
                (tree + 1).to_string(),
                format!("{:.6}", curve_acc),
                format!("{:.6}", curve_auc),
                format!("{:.6}", curve_f1),
            ])?;
        }

        // Final evaluation (same as last tree)
        let predictions = aggregator.model().predict(&test_features);
        let acc = accuracy(&predictions, &test_targets, 0.5);
        let auc = auc_roc(&predictions, &test_targets);
        let f1 = f1_score(&predictions, &test_targets, 0.5);

        eprintln!(
            "  Split {}: accuracy={:.4}, auc={:.4}, f1={:.4}, train={}, test={}",
            split_id, acc, auc, f1, n_train, n_test
        );

        metrics_wtr.write_record(&[
            split_id.to_string(),
            format!("{:.6}", acc),
            format!("{:.6}", auc),
            format!("{:.6}", f1),
            n_train.to_string(),
            n_test.to_string(),
            n_features.to_string(),
        ])?;

        let traffic = aggregator.traffic_log().expect("traffic log enabled");
        totals_wtr.write_record(&[
            split_id.to_string(),
            traffic.total_c2s().to_string(),
            traffic.total_s2a().to_string(),
            traffic.total_broadcast().to_string(),
            traffic.total_bytes().to_string(),
            args.hide_path.to_string(),
        ])?;

        for entry in traffic.entries() {
            per_round_wtr.write_record(&[
                split_id.to_string(),
                entry.tree.to_string(),
                entry.depth.to_string(),
                entry.direction.to_string(),
                entry.message_type.to_string(),
                entry.bytes.to_string(),
                entry.count.to_string(),
            ])?;
        }
    }

    metrics_wtr.flush()?;
    totals_wtr.flush()?;
    per_round_wtr.flush()?;
    curve_wtr.flush()?;

    eprintln!("Results written to {}", args.output_dir.display());

    // Dropout sweep
    if args.dropout_sweep {
        eprintln!("\nRunning dropout sweep...");
        let mut dropout_wtr =
            csv::Writer::from_path(args.output_dir.join("dropout.csv"))?;
        dropout_wtr.write_record([
            "dropout_rate", "split_id", "accuracy", "auc_roc", "f1",
        ])?;

        for rate_pct in (0..=90).step_by(10) {
            let rate = rate_pct as f64 / 100.0;
            eprintln!("  Dropout rate: {:.0}%", rate * 100.0);

            for split_id in 0..args.n_splits {
                let split_data = stratified_split(&dataset.targets, args.test_split, split_id as u64);

                let train_features: Vec<Vec<f64>> = split_data.train_indices.iter().map(|&i| dataset.features[i].clone()).collect();
                let train_targets: Vec<f64> = split_data.train_indices.iter().map(|&i| dataset.targets[i]).collect();
                let test_features: Vec<Vec<f64>> = split_data.test_indices.iter().map(|&i| dataset.features[i].clone()).collect();
                let test_targets: Vec<f64> = split_data.test_indices.iter().map(|&i| dataset.targets[i]).collect();

                let shareholders: Vec<ShareHolder> = (0..args.n_shareholders)
                    .map(|i| ShareHolder::new(i, (i + 1) as u64, args.min_clients))
                    .collect();
                let mut aggregator = Aggregator::new(
                    shareholders, args.n_bins, args.threshold, args.min_clients,
                    args.learning_rate, args.lambda, bin_method,
                )?;

                let mut clients: Vec<Client> = train_features.iter()
                    .zip(train_targets.iter())
                    .enumerate()
                    .map(|(idx, (feats, &target))| {
                        Client::new(format!("d{}_{}", split_id, idx), feats.clone(), target,
                                    args.n_shareholders, args.threshold,
                                    Some((split_id * 100_000 + idx) as u64))
                    })
                    .collect();

                let stat_shares: Vec<Vec<CommittedStatsShare>> = clients
                    .par_iter_mut()
                    .map(|client| client.compute_stat_shares().unwrap())
                    .collect();
                for shares in stat_shares {
                    for (i, share) in shares.into_iter().enumerate() {
                        aggregator.shareholders_mut()[i].receive_stats(share);
                    }
                }
                aggregator.define_bins()?;

                for tree in 0..args.n_trees {
                    for depth in 0..args.max_depth {
                        let ctx = aggregator.round_context();
                        let mut round_rng = StdRng::seed_from_u64(
                            (split_id * 10000 + rate_pct * 100 + tree * 10 + depth) as u64
                        );
                        let mut participating: Vec<usize> = (0..clients.len()).collect();
                        if rate > 0.0 {
                            let n_keep = ((1.0 - rate) * clients.len() as f64).round() as usize;
                            participating.shuffle(&mut round_rng);
                            participating.truncate(n_keep.max(1));
                        }

                        let participating_set: HashSet<usize> = participating.iter().copied().collect();
                        let all_shares: Vec<Vec<CommittedGradientShare>> = clients
                            .par_iter_mut()
                            .enumerate()
                            .map(|(ci, client)| {
                                if participating_set.contains(&ci) {
                                    client.compute_gradient_shares(&ctx, &Loss::Logistic, false).unwrap()
                                } else {
                                    Vec::new()
                                }
                            })
                            .collect();
                        for shares in all_shares {
                            if shares.is_empty() { continue; }
                            for (i, share) in shares.into_iter().enumerate() {
                                aggregator.shareholders_mut()[i].receive_gradients(share);
                            }
                        }

                        if !aggregator.compute_splits(0.0, 1.0)? {
                            break;
                        }
                    }
                    aggregator.finish_round();
                }

                let predictions = aggregator.model().predict(&test_features);
                let acc = accuracy(&predictions, &test_targets, 0.5);
                let auc = auc_roc(&predictions, &test_targets);
                let f1 = f1_score(&predictions, &test_targets, 0.5);

                dropout_wtr.write_record(&[
                    format!("{:.2}", rate),
                    split_id.to_string(),
                    format!("{:.6}", acc),
                    format!("{:.6}", auc),
                    format!("{:.6}", f1),
                ])?;
            }
        }
        dropout_wtr.flush()?;
        eprintln!("Dropout results written to {}", args.output_dir.join("dropout.csv").display());
    }

    Ok(())
}
