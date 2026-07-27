use clap::Parser;
use pbr_core::{
    Aggregator, BinMethod, Client, Loss, ShareHolder, read_csv, stratified_split,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "gain_retention", about = "Measure gain retention: binned vs exact splits")]
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

    #[arg(long, default_value = "gaussian")]
    bin_method: String,

    #[arg(long, default_value = "results")]
    output_dir: PathBuf,
}

/// Compute the best exact (unbinned) gain for a given feature over samples in a node.
/// Returns None if no valid split exists.
fn exact_best_gain(
    feature_values: &[f64],
    gradients: &[f64],
    hessians: &[f64],
    lambda_reg: f64,
    min_child_weight: f64,
) -> Option<f64> {
    // Collect (value, gradient, hessian) and sort by value
    let mut samples: Vec<(f64, f64, f64)> = feature_values
        .iter()
        .zip(gradients.iter())
        .zip(hessians.iter())
        .map(|((&v, &g), &h)| (v, g, h))
        .collect();
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_g: f64 = samples.iter().map(|s| s.1).sum();
    let total_h: f64 = samples.iter().map(|s| s.2).sum();
    let base_score = (total_g * total_g) / (total_h + lambda_reg);

    let mut best_gain: Option<f64> = None;
    let mut g_left = 0.0;
    let mut h_left = 0.0;

    for i in 0..samples.len() - 1 {
        g_left += samples[i].1;
        h_left += samples[i].2;

        // Only split between distinct values
        if samples[i].0 == samples[i + 1].0 {
            continue;
        }

        let g_right = total_g - g_left;
        let h_right = total_h - h_left;

        if h_left < min_child_weight || h_right < min_child_weight {
            continue;
        }

        let left_score = (g_left * g_left) / (h_left + lambda_reg);
        let right_score = (g_right * g_right) / (h_right + lambda_reg);
        let gain = left_score + right_score - base_score;

        if gain > best_gain.unwrap_or(0.0) {
            best_gain = Some(gain);
        }
    }

    best_gain
}

/// Track which node each sample belongs to, given the splits made so far.
fn assign_nodes(
    features: &[Vec<f64>],
    splits: &std::collections::BTreeMap<usize, pbr_core::SplitDecision>,
) -> Vec<usize> {
    features
        .iter()
        .map(|f| {
            let mut node_id = 0usize;
            while let Some(split) = splits.get(&node_id) {
                if f[split.feature_idx] <= split.threshold {
                    node_id = split.left_child_id;
                } else {
                    node_id = split.right_child_id;
                }
            }
            node_id
        })
        .collect()
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

    let mut wtr = csv::Writer::from_path(args.output_dir.join("gain_retention.csv"))?;
    wtr.write_record([
        "split_id",
        "tree",
        "depth",
        "node_id",
        "feature_idx",
        "binned_gain",
        "exact_gain",
        "retention",
        "n_samples_in_node",
    ])?;

    for split_id in 0..args.n_splits {
        eprintln!("Running split {}/{}", split_id + 1, args.n_splits);

        let split_data = stratified_split(&dataset.targets, args.test_split, split_id as u64);

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

        // Stats phase
        for client in &mut clients {
            let shares = client.compute_stat_shares()?;
            for (i, share) in shares.into_iter().enumerate() {
                aggregator.shareholders_mut()[i].receive_stats(share);
            }
        }
        aggregator.define_bins()?;

        // Tree training with gain retention tracking
        for tree in 0..args.n_trees {
            // Compute current predictions and gradients for all train samples
            let predictions = aggregator.model().predict(&train_features);
            let gradients: Vec<f64> = predictions
                .iter()
                .zip(train_targets.iter())
                .map(|(&pred, &target)| {
                    let p = 1.0 / (1.0 + (-pred).exp());
                    p - target
                })
                .collect();
            let hessians: Vec<f64> = predictions
                .iter()
                .map(|&pred| {
                    let p = 1.0 / (1.0 + (-pred).exp());
                    p * (1.0 - p)
                })
                .collect();

            for depth in 0..args.max_depth {
                let ctx = aggregator.round_context();

                // Assign samples to nodes before this depth's splits
                let node_assignments = assign_nodes(&train_features, &ctx.splits);

                // Run the protocol split
                for client in &mut clients {
                    let shares = client.compute_gradient_shares(
                        &ctx,
                        &Loss::Logistic,
                        false,
                    )?;
                    for (i, share) in shares.into_iter().enumerate() {
                        aggregator.shareholders_mut()[i].receive_gradients(share);
                    }
                }

                let made_progress = aggregator.compute_splits(0.0, 1.0)?;
                if !made_progress {
                    break;
                }

                // Now compare: for each new split the aggregator made,
                // compute exact gain on the same node and feature
                let new_ctx = aggregator.round_context();
                for (node_id, split) in &new_ctx.splits {
                    // Only process splits at this depth
                    if ctx.splits.contains_key(node_id) {
                        continue;
                    }

                    // Collect samples in this node
                    let sample_indices: Vec<usize> = node_assignments
                        .iter()
                        .enumerate()
                        .filter(|&(_, n)| *n == *node_id)
                        .map(|(i, _)| i)
                        .collect();

                    let node_feature_values: Vec<f64> = sample_indices
                        .iter()
                        .map(|&i| train_features[i][split.feature_idx])
                        .collect();
                    let node_gradients: Vec<f64> =
                        sample_indices.iter().map(|&i| gradients[i]).collect();
                    let node_hessians: Vec<f64> =
                        sample_indices.iter().map(|&i| hessians[i]).collect();

                    let exact = exact_best_gain(
                        &node_feature_values,
                        &node_gradients,
                        &node_hessians,
                        args.lambda,
                        1.0,
                    );

                    if let Some(exact_gain) = exact {
                        let retention = if exact_gain > 0.0 {
                            (split.gain / exact_gain).min(1.0)
                        } else {
                            1.0
                        };

                        wtr.write_record(&[
                            split_id.to_string(),
                            tree.to_string(),
                            depth.to_string(),
                            node_id.to_string(),
                            split.feature_idx.to_string(),
                            format!("{:.8}", split.gain),
                            format!("{:.8}", exact_gain),
                            format!("{:.8}", retention),
                            sample_indices.len().to_string(),
                        ])?;
                    }
                }
            }
            aggregator.finish_round();
        }
    }

    wtr.flush()?;
    eprintln!(
        "Gain retention results written to {}",
        args.output_dir.join("gain_retention.csv").display()
    );

    Ok(())
}
