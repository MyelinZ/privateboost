use std::collections::HashMap;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use tonic::transport::Channel;

use privateboost::crypto::{commitment, fixed_point, shamir};
use privateboost::domain::model::{
    BinConfiguration, Model, SplitDecision, SplitNode, Tree, TreeNode,
};
use privateboost::grpc::scalars_to_bytes;
use privateboost::proto;
use privateboost::proto::coordinator_service_client::CoordinatorServiceClient;
use privateboost::proto::shareholder_service_client::ShareholderServiceClient;

// -- CLI definition --

#[derive(Parser)]
#[command(name = "pbctl", about = "Admin CLI for privateboost training runs")]
struct Cli {
    /// Coordinator address (host:port)
    #[arg(long, global = true, env = "COORDINATOR")]
    coordinator: Option<String>,

    /// Comma-separated shareholder addresses
    #[arg(long, global = true, env = "SHAREHOLDERS")]
    shareholders: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all runs from coordinator
    List,
    /// Show run status from shareholders
    Status {
        /// Run ID (defaults to "default")
        #[arg(long, env = "RUN_ID")]
        run_id: Option<String>,
    },
    /// Create a new run from YAML config
    Create {
        /// Path to YAML job config
        #[arg(long)]
        config: String,
    },
    /// Cancel a run
    Cancel {
        /// Run ID to cancel
        run_id: String,
    },
    /// Show training config for a run
    Config {
        /// Run ID
        run_id: String,
    },
    /// Submit data and train (full client workflow)
    Submit {
        /// Path to YAML job config
        #[arg(long)]
        config: String,
        /// Use existing run ID instead of creating new one
        #[arg(long, env = "RUN_ID")]
        run_id: Option<String>,
    },
}

// -- YAML config structs --

#[derive(Deserialize)]
struct JobConfig {
    training: TrainingParams,
    features: Vec<FeatureSpec>,
    target: String,
    data: DataConfig,
}

#[derive(Deserialize)]
struct TrainingParams {
    n_trees: i32,
    max_depth: i32,
    learning_rate: f64,
    lambda_reg: f64,
    n_bins: i32,
    loss: String,
    threshold: usize,
    scale: f64,
}

#[derive(Deserialize)]
struct FeatureSpec {
    name: String,
}

#[derive(Deserialize)]
struct DataConfig {
    dataset: String,
    test_split: f64,
}

// -- Helpers --

fn coordinator_addr(cli: &Cli) -> String {
    cli.coordinator
        .clone()
        .unwrap_or_else(|| "localhost:50053".into())
}

fn shareholder_addrs(cli: &Cli) -> Vec<String> {
    cli.shareholders
        .as_deref()
        .unwrap_or("localhost:50061,localhost:50062,localhost:50063")
        .split(',')
        .map(|s| s.trim().to_string())
        .collect()
}

fn format_url(addr: &str) -> String {
    if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    }
}

fn read_job_config(path: &str) -> Result<JobConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {path}"))?;
    serde_yaml::from_str(&contents).context("Failed to parse YAML config")
}

fn job_to_training_config(job: &JobConfig, train_count: i32) -> proto::TrainingConfig {
    proto::TrainingConfig {
        features: job
            .features
            .iter()
            .enumerate()
            .map(|(i, f)| proto::FeatureSpec {
                index: i as i32,
                name: f.name.clone(),
                scale: job.training.scale,
            })
            .collect(),
        target_column: job.target.clone(),
        loss: job.training.loss.clone(),
        n_bins: job.training.n_bins,
        n_trees: job.training.n_trees,
        max_depth: job.training.max_depth,
        learning_rate: job.training.learning_rate,
        lambda_reg: job.training.lambda_reg,
        min_clients: train_count,
        target: Some(proto::training_config::Target::TargetCount(train_count)),
        gradient_scale: job.training.scale,
    }
}

// -- Main --

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match &cli.command {
        Commands::List => cmd_list(&cli).await,
        Commands::Status { run_id } => {
            let rid = run_id.as_deref().unwrap_or("default");
            cmd_status(&cli, rid).await
        }
        Commands::Create { config } => cmd_create(&cli, config).await,
        Commands::Cancel { run_id } => cmd_cancel(&cli, run_id).await,
        Commands::Config { run_id } => cmd_config(&cli, run_id).await,
        Commands::Submit { config, run_id } => cmd_submit(&cli, config, run_id.as_deref()).await,
    }
}

// -- Command implementations --

async fn cmd_list(cli: &Cli) -> Result<()> {
    let mut coord =
        CoordinatorServiceClient::connect(format_url(&coordinator_addr(cli))).await?;
    let resp = coord
        .list_runs(proto::ListRunsRequest {})
        .await?
        .into_inner();

    if resp.runs.is_empty() {
        println!("No runs found.");
        return Ok(());
    }

    println!("{:<40} STATUS", "RUN ID");
    for run in &resp.runs {
        let status = match run.status {
            s if s == proto::RunStatus::RunActive as i32 => "Active",
            s if s == proto::RunStatus::RunCancelled as i32 => "Cancelled",
            s if s == proto::RunStatus::RunComplete as i32 => "Complete",
            _ => "Unknown",
        };
        println!("{:<40} {status}", run.run_id);
    }
    Ok(())
}

async fn cmd_status(cli: &Cli, run_id: &str) -> Result<()> {
    let addrs = shareholder_addrs(cli);

    println!("Run: {run_id}");
    println!();
    println!("{:<30} {:<25} {:<10} {:<10}", "SHAREHOLDER", "PHASE", "ROUND", "DEPTH");

    for addr in &addrs {
        match ShareholderServiceClient::connect(format_url(addr)).await {
            Ok(mut client) => {
                match client
                    .get_run_state(proto::GetRunStateRequest {
                        run_id: run_id.to_string(),
                    })
                    .await
                {
                    Ok(resp) => {
                        let state = resp.into_inner();
                        let phase = phase_name(state.phase);
                        println!(
                            "{:<30} {:<25} {:<10} {:<10}",
                            addr, phase, state.round_id, state.depth
                        );
                    }
                    Err(e) => println!("{:<30} Error: {e}", addr),
                }
            }
            Err(e) => println!("{:<30} Unreachable: {e}", addr),
        }
    }
    Ok(())
}

fn phase_name(phase: i32) -> &'static str {
    match phase {
        p if p == proto::Phase::WaitingForClients as i32 => "WaitingForClients",
        p if p == proto::Phase::CollectingStats as i32 => "CollectingStats",
        p if p == proto::Phase::FrozenStats as i32 => "FrozenStats",
        p if p == proto::Phase::CollectingGradients as i32 => "CollectingGradients",
        p if p == proto::Phase::FrozenGradients as i32 => "FrozenGradients",
        p if p == proto::Phase::TrainingComplete as i32 => "TrainingComplete",
        _ => "Unknown",
    }
}

async fn cmd_create(cli: &Cli, config_path: &str) -> Result<()> {
    let job = read_job_config(config_path)?;

    // We don't know train count yet for create-only; use 0 and let submit set it
    let config = job_to_training_config(&job, 0);

    let mut coord =
        CoordinatorServiceClient::connect(format_url(&coordinator_addr(cli))).await?;
    let resp = coord
        .create_run(proto::CreateRunRequest {
            config: Some(config),
        })
        .await?
        .into_inner();

    println!("{}", resp.run_id);
    Ok(())
}

async fn cmd_cancel(cli: &Cli, run_id: &str) -> Result<()> {
    let mut coord =
        CoordinatorServiceClient::connect(format_url(&coordinator_addr(cli))).await?;
    coord
        .cancel_run(proto::CancelRunRequest {
            run_id: run_id.to_string(),
        })
        .await?;
    println!("Cancelled run: {run_id}");
    Ok(())
}

async fn cmd_config(cli: &Cli, run_id: &str) -> Result<()> {
    let mut coord =
        CoordinatorServiceClient::connect(format_url(&coordinator_addr(cli))).await?;
    let resp = coord
        .get_run_config(proto::GetRunConfigRequest {
            run_id: run_id.to_string(),
        })
        .await?
        .into_inner();

    match resp.config {
        Some(c) => {
            println!("Run: {run_id}");
            println!("Loss:          {}", c.loss);
            println!("Trees:         {}", c.n_trees);
            println!("Max depth:     {}", c.max_depth);
            println!("Learning rate: {}", c.learning_rate);
            println!("Lambda:        {}", c.lambda_reg);
            println!("Bins:          {}", c.n_bins);
            println!("Min clients:   {}", c.min_clients);
            println!("Gradient scale:{}", c.gradient_scale);
            println!("Features:");
            for f in &c.features {
                println!("  [{}] {} (scale={})", f.index, f.name, f.scale);
            }
        }
        None => println!("No config found for run: {run_id}"),
    }
    Ok(())
}

// -- Submit command (full client workflow) --

const DATASET_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data";

struct Sample {
    features: Vec<f64>,
    target: f64,
}

async fn cmd_submit(cli: &Cli, config_path: &str, existing_run_id: Option<&str>) -> Result<()> {
    let job = read_job_config(config_path)?;
    let scale = job.training.scale;
    let loss = job.training.loss.clone();
    let threshold = job.training.threshold;
    let learning_rate = job.training.learning_rate;
    let n_trees = job.training.n_trees;
    let max_depth = job.training.max_depth;
    let feature_names: Vec<String> = job.features.iter().map(|f| f.name.clone()).collect();

    let sh_addrs = shareholder_addrs(cli);
    let n_parties = sh_addrs.len();

    println!(
        "Config: {} trees, depth {}, lr={}, loss={}, threshold={}",
        n_trees, max_depth, learning_rate, loss, threshold
    );

    // Download dataset
    let data = download_dataset(&job.data.dataset).await?;
    let (train, test) = train_test_split(&data, 1.0 - job.data.test_split);
    println!("Train: {} samples, Test: {} samples", train.len(), test.len());

    // Create or use existing run
    let run_id = match existing_run_id {
        Some(id) => {
            println!("Using existing run: {id}");
            id.to_string()
        }
        None => {
            let config = job_to_training_config(&job, train.len() as i32);
            let mut coord =
                CoordinatorServiceClient::connect(format_url(&coordinator_addr(cli))).await?;
            let id = coord
                .create_run(proto::CreateRunRequest {
                    config: Some(config),
                })
                .await?
                .into_inner()
                .run_id;
            println!("Created run: {id}");
            id
        }
    };

    // Connect to shareholders and register run
    let mut sh_clients: Vec<ShareholderServiceClient<Channel>> = Vec::new();
    for addr in &sh_addrs {
        let client = ShareholderServiceClient::connect(format_url(addr)).await?;
        sh_clients.push(client);
    }

    for client in &mut sh_clients {
        client
            .register_run(proto::RegisterRunRequest {
                run_id: run_id.clone(),
                target_count: train.len() as i32,
                n_trees,
                max_depth,
            })
            .await?;
    }
    println!("Run registered on {} shareholders", sh_clients.len());

    // Submit stats
    println!("Submitting statistics...");
    for (i, sample) in train.iter().enumerate() {
        let client_id = format!("client_{i}");
        let nonce: [u8; 32] = rand::random();

        let mut values = Vec::with_capacity(feature_names.len() * 2 + 2);
        for &f in &sample.features {
            values.push(f);
            values.push(f * f);
        }
        values.push(sample.target);
        values.push(sample.target * sample.target);

        let encoded = fixed_point::encode_vec(&values, scale);
        let shares = shamir::share(&encoded, n_parties, threshold)?;
        let commitment_hash = commitment::compute_commitment(0, &client_id, &nonce, &[]);

        for (j, share) in shares.iter().enumerate() {
            let proto_share = proto::Share {
                x: share.x,
                scalars: scalars_to_bytes(&share.y),
                count: share.y.len() as i32,
            };
            sh_clients[j]
                .submit_stats(proto::SubmitStatsRequest {
                    run_id: run_id.clone(),
                    commitment: commitment_hash.to_vec(),
                    share: Some(proto_share),
                })
                .await?;
        }

        if (i + 1) % 50 == 0 {
            println!("  {}/{} clients submitted stats", i + 1, train.len());
        }
    }
    println!("All {} clients submitted stats", train.len());

    // Wait for bins consensus
    println!("Waiting for bins consensus...");
    let (bin_configs_proto, initial_prediction) = loop {
        let state = sh_clients[0]
            .get_run_state(proto::GetRunStateRequest {
                run_id: run_id.clone(),
            })
            .await?
            .into_inner();
        if state.phase == proto::Phase::CollectingGradients as i32 {
            let resp = sh_clients[0]
                .get_consensus_bins(proto::GetConsensusBinsRequest {
                    run_id: run_id.clone(),
                })
                .await?
                .into_inner();
            break (resp.bins, resp.initial_prediction);
        }
        if state.phase == proto::Phase::TrainingComplete as i32 {
            println!("Training completed during stats phase");
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    };

    let bins: Vec<BinConfiguration> = bin_configs_proto.iter().map(parse_bin_config).collect();
    println!(
        "Bins computed ({} features), initial_prediction={:.4}",
        bins.len(),
        initial_prediction
    );

    // Training loop
    let mut model = Model::new(initial_prediction, learning_rate);
    let mut current_round: i32 = -1;
    let mut current_depth: i32 = -1;
    let mut splits: HashMap<i32, SplitDecision> = HashMap::new();

    loop {
        let state = sh_clients[0]
            .get_run_state(proto::GetRunStateRequest {
                run_id: run_id.clone(),
            })
            .await?
            .into_inner();

        if state.phase == proto::Phase::TrainingComplete as i32 {
            break;
        }

        if state.phase == proto::Phase::CollectingGradients as i32
            || state.phase == proto::Phase::FrozenStats as i32
        {
            if state.round_id == current_round && state.depth == current_depth {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }

            current_round = state.round_id;
            current_depth = state.depth;

            if current_depth > 0 {
                let resp = sh_clients[0]
                    .get_consensus_splits(proto::GetConsensusSplitsRequest {
                        run_id: run_id.clone(),
                        round_id: current_round,
                        depth: current_depth - 1,
                    })
                    .await?
                    .into_inner();
                if resp.ready {
                    for (&nid, sd) in &resp.splits {
                        splits.insert(
                            nid,
                            SplitDecision {
                                node_id: sd.node_id,
                                feature_idx: sd.feature_idx as usize,
                                threshold: sd.threshold,
                                gain: sd.gain,
                                left_child_id: sd.left_child_id,
                                right_child_id: sd.right_child_id,
                                g_left: 0.0,
                                h_left: 0.0,
                                g_right: 0.0,
                                h_right: 0.0,
                            },
                        );
                    }
                }
            } else if current_round > 0 {
                let model_resp = sh_clients[0]
                    .get_consensus_model(proto::GetConsensusModelRequest {
                        run_id: run_id.clone(),
                    })
                    .await?
                    .into_inner();
                if let Some(proto_model) = model_resp.model {
                    if !proto_model.trees.is_empty() {
                        model = proto_to_model(&proto_model, learning_rate);
                    }
                }
                splits.clear();
            }

            println!("  Round {current_round}, depth {current_depth}");

            for (i, sample) in train.iter().enumerate() {
                let client_id = format!("client_{i}");
                submit_gradients(
                    &mut sh_clients,
                    &run_id,
                    &client_id,
                    sample,
                    &bins,
                    &model,
                    &splits,
                    current_round,
                    current_depth,
                    &loss,
                    n_parties,
                    threshold,
                    scale,
                )
                .await?;
            }
        } else {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    }

    // Fetch final model and evaluate
    println!("Training complete, fetching model...");
    let model_resp = sh_clients[0]
        .get_consensus_model(proto::GetConsensusModelRequest {
            run_id: run_id.clone(),
        })
        .await?
        .into_inner();

    let final_model = match model_resp.model {
        Some(proto_model) if !proto_model.trees.is_empty() => {
            proto_to_model(&proto_model, learning_rate)
        }
        _ => {
            println!("Warning: Could not fetch final model, using last known model");
            model
        }
    };
    println!("Model has {} trees", final_model.trees.len());

    let mut correct = 0;
    for sample in &test {
        let pred = final_model.predict_one(&sample.features);
        let sigmoid = 1.0 / (1.0 + (-pred).exp());
        let predicted_class = if sigmoid >= 0.5 { 1.0 } else { 0.0 };
        if (predicted_class - sample.target).abs() < 0.5 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / test.len() as f64;
    println!("\nTest accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

// -- Dataset loading --

async fn download_dataset(name: &str) -> Result<Vec<Sample>> {
    let url = match name {
        "heart-disease" => DATASET_URL,
        _ => anyhow::bail!("Unknown dataset: {name}. Available: heart-disease"),
    };

    println!("Downloading {name} dataset...");
    let body = reqwest::get(url)
        .await
        .context("Failed to download dataset")?
        .text()
        .await
        .context("Failed to read dataset body")?;

    let mut samples = Vec::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(body.as_bytes());

    for result in rdr.records() {
        let record = result?;
        if record.len() < 14 {
            continue;
        }
        if record.iter().any(|field| field.trim() == "?") {
            continue;
        }
        let values: Result<Vec<f64>, _> = record.iter().map(|f| f.trim().parse::<f64>()).collect();
        let values = match values {
            Ok(v) => v,
            Err(_) => continue,
        };
        let features = values[..13].to_vec();
        let target = if values[13] > 0.0 { 1.0 } else { 0.0 };
        samples.push(Sample { features, target });
    }

    Ok(samples)
}

fn train_test_split(data: &[Sample], train_frac: f64) -> (Vec<&Sample>, Vec<&Sample>) {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    let mut seed: u64 = 42;
    for i in (1..indices.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (seed >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    let split_idx = (data.len() as f64 * train_frac) as usize;
    let train: Vec<&Sample> = indices[..split_idx].iter().map(|&i| &data[i]).collect();
    let test: Vec<&Sample> = indices[split_idx..].iter().map(|&i| &data[i]).collect();
    (train, test)
}

// -- Proto conversion helpers --

fn parse_bin_config(pb: &proto::BinConfiguration) -> BinConfiguration {
    BinConfiguration {
        feature_idx: pb.feature_idx as usize,
        edges: ndarray_to_f64(pb.edges.as_ref()),
        inner_edges: ndarray_to_f64(pb.inner_edges.as_ref()),
        n_bins: pb.n_bins as usize,
    }
}

fn ndarray_to_f64(arr: Option<&proto::NdArray>) -> Vec<f64> {
    match arr {
        Some(a) => a
            .data
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
        None => Vec::new(),
    }
}

fn proto_to_model(pb: &proto::Model, learning_rate: f64) -> Model {
    let lr = if pb.learning_rate > 0.0 {
        pb.learning_rate
    } else {
        learning_rate
    };
    let mut model = Model::new(pb.initial_prediction, lr);
    for tree_pb in &pb.trees {
        if let Some(root) = &tree_pb.root {
            model.add_tree(Tree {
                root: proto_to_tree_node(root),
            });
        }
    }
    model
}

fn proto_to_tree_node(pb: &proto::TreeNode) -> TreeNode {
    match &pb.node {
        Some(proto::tree_node::Node::Split(s)) => TreeNode::Split(SplitNode {
            feature_idx: s.feature_idx as usize,
            threshold: s.threshold,
            gain: 0.0,
            left: Box::new(
                s.left
                    .as_ref()
                    .map(|n| proto_to_tree_node(n))
                    .unwrap_or(TreeNode::Leaf(
                        privateboost::domain::model::LeafNode {
                            value: 0.0,
                            n_samples: 0,
                        },
                    )),
            ),
            right: Box::new(
                s.right
                    .as_ref()
                    .map(|n| proto_to_tree_node(n))
                    .unwrap_or(TreeNode::Leaf(
                        privateboost::domain::model::LeafNode {
                            value: 0.0,
                            n_samples: 0,
                        },
                    )),
            ),
        }),
        Some(proto::tree_node::Node::Leaf(l)) => {
            TreeNode::Leaf(privateboost::domain::model::LeafNode {
                value: l.value,
                n_samples: 0,
            })
        }
        None => TreeNode::Leaf(privateboost::domain::model::LeafNode {
            value: 0.0,
            n_samples: 0,
        }),
    }
}

// -- Gradient submission --

async fn submit_gradients(
    sh_clients: &mut [ShareholderServiceClient<Channel>],
    run_id: &str,
    client_id: &str,
    sample: &Sample,
    bins: &[BinConfiguration],
    model: &Model,
    splits: &HashMap<i32, SplitDecision>,
    round_id: i32,
    depth: i32,
    loss: &str,
    n_parties: usize,
    threshold: usize,
    scale: f64,
) -> Result<()> {
    let node_id = get_node_id(&sample.features, splits);
    let nonce: [u8; 32] = rand::random();
    let commitment_hash = commitment::compute_commitment(round_id, client_id, &nonce, &[]);

    let prediction = model.predict_one(&sample.features);
    let (gradient, hessian) = if loss == "logistic" {
        let p = 1.0 / (1.0 + (-prediction).exp());
        (p - sample.target, p * (1.0 - p))
    } else {
        (prediction - sample.target, 1.0)
    };

    let mut all_values = Vec::new();
    for config in bins {
        let n_total = config.n_bins + 2;
        let bin_idx = find_bin_index(sample.features[config.feature_idx], &config.edges, n_total);
        let mut g_vec = vec![0.0; n_total];
        g_vec[bin_idx] = gradient;
        all_values.extend_from_slice(&g_vec);
    }
    for config in bins {
        let n_total = config.n_bins + 2;
        let bin_idx = find_bin_index(sample.features[config.feature_idx], &config.edges, n_total);
        let mut h_vec = vec![0.0; n_total];
        h_vec[bin_idx] = hessian;
        all_values.extend_from_slice(&h_vec);
    }

    let encoded = fixed_point::encode_vec(&all_values, scale);
    let shares = shamir::share(&encoded, n_parties, threshold)?;

    for (j, share) in shares.iter().enumerate() {
        let proto_share = proto::Share {
            x: share.x,
            scalars: scalars_to_bytes(&share.y),
            count: share.y.len() as i32,
        };
        sh_clients[j]
            .submit_gradients(proto::SubmitGradientsRequest {
                run_id: run_id.to_string(),
                round_id,
                depth,
                commitment: commitment_hash.to_vec(),
                share: Some(proto_share),
                node_id,
            })
            .await?;
    }

    Ok(())
}

fn get_node_id(features: &[f64], splits: &HashMap<i32, SplitDecision>) -> i32 {
    let mut node_id = 0;
    while let Some(split) = splits.get(&node_id) {
        if features[split.feature_idx] <= split.threshold {
            node_id = split.left_child_id;
        } else {
            node_id = split.right_child_id;
        }
    }
    node_id
}

fn find_bin_index(value: f64, edges: &[f64], n_total_bins: usize) -> usize {
    let pos = edges.partition_point(|&e| e <= value);
    let idx = if pos == 0 { 0 } else { pos - 1 };
    idx.min(n_total_bins - 1)
}
