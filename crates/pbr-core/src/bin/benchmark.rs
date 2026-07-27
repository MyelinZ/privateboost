use clap::Parser;
use pbr_core::{
    Aggregator, BinMethod, Client, Dataset, Loss, ShareHolder, accuracy, auc_roc,
    f1_score, read_csv, stratified_split,
};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::path::PathBuf;
use std::time::{Duration, Instant};

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

    /// Fraction of clients that skip the statistics phase (bin edges derived from the rest)
    #[arg(long, default_value_t = 0.0)]
    stats_dropout_rate: f64,

    /// Run statistics-phase dropout sweep: rates 0.0, 0.1, ..., 0.9 and write stats_dropout.csv
    #[arg(long, default_value_t = false)]
    stats_dropout_sweep: bool,

    /// Run per-(client, shareholder) message loss sweep: rates 0.0, 0.1, ..., 0.5
    /// and write share_loss.csv
    #[arg(long, default_value_t = false)]
    share_loss_sweep: bool,

    /// Binning method: "gaussian" (default) or "uniform"
    #[arg(long, default_value = "gaussian")]
    bin_method: String,

    #[arg(long, default_value = "results")]
    output_dir: PathBuf,
}

fn main() -> pbr_core::Result<()> {
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

    let mut curve_wtr =
        csv::Writer::from_path(args.output_dir.join("learning_curve.csv"))?;
    curve_wtr.write_record([
        "split_id", "n_trees", "accuracy", "auc_roc", "f1",
    ])?;

    let mut timing_wtr =
        csv::Writer::from_path(args.output_dir.join("timing.csv"))?;
    timing_wtr.write_record([
        "split_id",
        "n_train",
        "wall_s",
        "client_stats_s",
        "client_grad_s",
        "server_s",
        "grad_share_calls",
        "hide_path",
        "n_shareholders",
        "threshold",
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
                    None,
                )
            })
            .collect();

        let mut t_client_stats = Duration::ZERO;
        let mut t_client_grad = Duration::ZERO;
        let mut t_server = Duration::ZERO;
        let mut t_eval = Duration::ZERO;
        let mut grad_share_calls: u64 = 0;
        let t_wall = Instant::now();

        // Stats phase (optionally with statistics-phase dropout)
        let mut stats_participating: Vec<usize> = (0..clients.len()).collect();
        if args.stats_dropout_rate > 0.0 {
            let mut stats_rng = StdRng::seed_from_u64(split_id as u64 * 31 + 7);
            let n_keep =
                ((1.0 - args.stats_dropout_rate) * clients.len() as f64).round() as usize;
            stats_participating.shuffle(&mut stats_rng);
            stats_participating.truncate(n_keep.max(args.min_clients));
        }
        for &ci in &stats_participating {
            let t = Instant::now();
            let shares = clients[ci].compute_stat_shares()?;
            t_client_stats += t.elapsed();
            let t = Instant::now();
            for (i, share) in shares.into_iter().enumerate() {
                aggregator.shareholders_mut()[i].receive_stats(share);
            }
            t_server += t.elapsed();
        }
        let t = Instant::now();
        aggregator.define_bins()?;
        t_server += t.elapsed();

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

                for &ci in &participating {
                    let t = Instant::now();
                    let shares = clients[ci].compute_gradient_shares(
                        &ctx,
                        &Loss::Logistic,
                        args.hide_path,
                    )?;
                    t_client_grad += t.elapsed();
                    grad_share_calls += 1;
                    let t = Instant::now();
                    if args.hide_path {
                        for share in shares {
                            let sh_idx = share.share.x as usize - 1;
                            aggregator.shareholders_mut()[sh_idx].receive_gradients(share);
                        }
                    } else {
                        for (i, share) in shares.into_iter().enumerate() {
                            aggregator.shareholders_mut()[i].receive_gradients(share);
                        }
                    }
                    t_server += t.elapsed();
                }

                let t = Instant::now();
                let more_splits = aggregator.compute_splits(0.0, 1.0)?;
                t_server += t.elapsed();
                if !more_splits {
                    break;
                }
            }
            let t = Instant::now();
            aggregator.finish_round();
            t_server += t.elapsed();

            // Per-tree evaluation for learning curve
            let te = Instant::now();
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
            t_eval += te.elapsed();
        }

        let wall_s = t_wall.elapsed().as_secs_f64() - t_eval.as_secs_f64();
        timing_wtr.write_record(&[
            split_id.to_string(),
            n_train.to_string(),
            format!("{:.6}", wall_s),
            format!("{:.6}", t_client_stats.as_secs_f64()),
            format!("{:.6}", t_client_grad.as_secs_f64()),
            format!("{:.6}", t_server.as_secs_f64()),
            grad_share_calls.to_string(),
            args.hide_path.to_string(),
            args.n_shareholders.to_string(),
            args.threshold.to_string(),
        ])?;

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
    }

    metrics_wtr.flush()?;
    curve_wtr.flush()?;
    timing_wtr.flush()?;

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
                                    args.n_shareholders, args.threshold, None)
                    })
                    .collect();

                for client in &mut clients {
                    let shares = client.compute_stat_shares()?;
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

                        for &ci in &participating {
                            let shares = clients[ci].compute_gradient_shares(
                                &ctx, &Loss::Logistic, false,
                            )?;
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

    // Statistics-phase dropout sweep
    if args.stats_dropout_sweep {
        eprintln!("\nRunning statistics-phase dropout sweep...");
        let mut wtr = csv::Writer::from_path(args.output_dir.join("stats_dropout.csv"))?;
        wtr.write_record(["stats_dropout_rate", "split_id", "accuracy", "auc_roc", "f1"])?;

        for rate_pct in (0..=90).step_by(10) {
            let rate = rate_pct as f64 / 100.0;
            eprintln!("  Stats dropout rate: {:.0}%", rate * 100.0);

            for split_id in 0..args.n_splits {
                let (acc, auc, f1) = run_sweep_split(
                    &dataset, split_id, &args, bin_method, rate, 0.0, 0.0,
                    20_000 + rate_pct as u64,
                )?;
                wtr.write_record(&[
                    format!("{:.2}", rate),
                    split_id.to_string(),
                    format!("{:.6}", acc),
                    format!("{:.6}", auc),
                    format!("{:.6}", f1),
                ])?;
            }
        }
        wtr.flush()?;
        eprintln!(
            "Stats dropout results written to {}",
            args.output_dir.join("stats_dropout.csv").display()
        );
    }

    // Per-(client, shareholder) message loss sweep
    if args.share_loss_sweep {
        eprintln!("\nRunning share loss sweep...");
        let mut wtr = csv::Writer::from_path(args.output_dir.join("share_loss.csv"))?;
        wtr.write_record(["share_loss_rate", "split_id", "accuracy", "auc_roc", "f1"])?;

        for rate_pct in (0..=50).step_by(10) {
            let rate = rate_pct as f64 / 100.0;
            eprintln!("  Share loss rate: {:.0}%", rate * 100.0);

            for split_id in 0..args.n_splits {
                let (acc, auc, f1) = run_sweep_split(
                    &dataset, split_id, &args, bin_method, 0.0, 0.0, rate,
                    40_000 + rate_pct as u64,
                )?;
                wtr.write_record(&[
                    format!("{:.2}", rate),
                    split_id.to_string(),
                    format!("{:.6}", acc),
                    format!("{:.6}", auc),
                    format!("{:.6}", f1),
                ])?;
            }
        }
        wtr.flush()?;
        eprintln!(
            "Share loss results written to {}",
            args.output_dir.join("share_loss.csv").display()
        );
    }

    Ok(())
}

/// Train one split with the given statistics-phase dropout, gradient-round dropout,
/// and per-(client, shareholder) message loss rates. Message loss is applied at the
/// granularity of a client's per-round submission to one shareholder, matching how
/// shares are transmitted; the commitment mechanism excludes clients whose shares
/// did not reach the selected shareholder subset. Returns (accuracy, auc_roc, f1).
#[allow(clippy::too_many_arguments)]
fn run_sweep_split(
    dataset: &Dataset,
    split_id: usize,
    args: &Args,
    bin_method: BinMethod,
    stats_dropout: f64,
    grad_dropout: f64,
    share_loss: f64,
    seed_base: u64,
) -> pbr_core::Result<(f64, f64, f64)> {
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
            Client::new(format!("s{}_{}_{}", seed_base, split_id, idx), feats.clone(), target,
                        args.n_shareholders, args.threshold, None)
        })
        .collect();

    let mut loss_rng = StdRng::seed_from_u64(seed_base.wrapping_mul(1_000_003) + split_id as u64);

    // Stats phase
    let mut stats_participating: Vec<usize> = (0..clients.len()).collect();
    if stats_dropout > 0.0 {
        let mut rng = StdRng::seed_from_u64(seed_base + split_id as u64 * 13);
        let n_keep = ((1.0 - stats_dropout) * clients.len() as f64).round() as usize;
        stats_participating.shuffle(&mut rng);
        stats_participating.truncate(n_keep.max(args.min_clients));
    }
    for &ci in &stats_participating {
        let shares = clients[ci].compute_stat_shares()?;
        let deliver: Vec<bool> = (0..args.n_shareholders)
            .map(|_| !(share_loss > 0.0 && loss_rng.random::<f64>() < share_loss))
            .collect();
        for (i, share) in shares.into_iter().enumerate() {
            if deliver[i] {
                aggregator.shareholders_mut()[i].receive_stats(share);
            }
        }
    }
    aggregator.define_bins()?;

    // Tree training
    for tree in 0..args.n_trees {
        for depth in 0..args.max_depth {
            let ctx = aggregator.round_context();
            let mut round_rng = StdRng::seed_from_u64(
                seed_base + (split_id * 100_000 + tree * 100 + depth) as u64,
            );
            let mut participating: Vec<usize> = (0..clients.len()).collect();
            if grad_dropout > 0.0 {
                let n_keep = ((1.0 - grad_dropout) * clients.len() as f64).round() as usize;
                participating.shuffle(&mut round_rng);
                participating.truncate(n_keep.max(1));
            }
            for &ci in &participating {
                let shares = clients[ci].compute_gradient_shares(&ctx, &Loss::Logistic, false)?;
                let deliver: Vec<bool> = (0..args.n_shareholders)
                    .map(|_| !(share_loss > 0.0 && loss_rng.random::<f64>() < share_loss))
                    .collect();
                for share in shares {
                    let sh_idx = share.share.x as usize - 1;
                    if deliver[sh_idx] {
                        aggregator.shareholders_mut()[sh_idx].receive_gradients(share);
                    }
                }
            }
            if !aggregator.compute_splits(0.0, 1.0)? {
                break;
            }
        }
        aggregator.finish_round();
    }

    let predictions = aggregator.model().predict(&test_features);
    Ok((
        accuracy(&predictions, &test_targets, 0.5),
        auc_roc(&predictions, &test_targets),
        f1_score(&predictions, &test_targets, 0.5),
    ))
}
