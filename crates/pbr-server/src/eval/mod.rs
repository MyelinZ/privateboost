//! Per-tree quality evaluation, run by the aggregator's own round loop.
//!
//! Sessions whose `dataset_id` has an entry under the config's
//! `[eval.datasets]` are scored at every tree boundary: the round loop
//! evaluates the just-finished model against that dataset's held-out split
//! and writes one `paperSimTreeMetrics` document to Firestore off the hot
//! path: spawned, never awaited inline, so a failed write costs one metric row
//! rather than a stalled round.
//!
//! The held-out split is the last 20% of a public benchmark CSV committed to
//! this repo: contributed by no client, never trained on, disjoint from all
//! client data. Every metric written from here is therefore a score against
//! that public split and not a measurement of any client's records, which is
//! also why the aggregator may hold it in the clear. Clients submit
//! secret shares only, and the aggregator sees no record or label of theirs
//! either way.

pub mod metrics;
pub mod firestore;

use anyhow::Context;
use firestore::FirestoreWriter;
use metrics::TreeMetrics;
use pbr_core::read_csv;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// One `paperSimTreeMetrics` row. `analysis/fetch.py` reads exactly these
/// fields, so their names and set are a binding contract; this struct's
/// `Serialize` impl is the single place they are produced, via the
/// `firestore` crate's serializer.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TreeMetricDoc {
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "treeIdx")]
    pub tree_idx: u32,
    #[serde(flatten)]
    pub metrics: TreeMetrics,
    #[serde(rename = "nTest")]
    pub n_test: usize,
    #[serde(rename = "thresholdUsed")]
    pub threshold_used: f64,
    /// Observation time. Written as a Firestore TIMESTAMP value via
    /// `firestore::serialize_as_timestamp`, so `analysis/fetch.py` reads it
    /// typed rather than parsing a string.
    #[serde(with = "::firestore::serialize_as_timestamp")]
    pub ts: chrono::DateTime<chrono::Utc>,
}

/// The labelled rows one dataset's learning curve is scored against.
#[derive(Debug)]
pub struct HeldOutSplit {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
}

impl HeldOutSplit {
    /// The last 20% of the CSV at `path` (the `pbr-e2e` test's own 80/20
    /// rule, so per-tree curves stay comparable with its final-model AUC
    /// gate). The first 80% must be exactly what that dataset's clients
    /// train on, or the held-out rows overlap training rows.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let data = read_csv(path, "target")
            .with_context(|| format!("eval test csv {}", path.display()))?;
        let split_idx = (data.features.len() as f64 * 0.8) as usize;
        let split = Self {
            features: data.features[split_idx..].to_vec(),
            targets: data.targets[split_idx..].to_vec(),
        };
        anyhow::ensure!(
            !split.targets.is_empty(),
            "held-out split of {} is empty",
            path.display()
        );
        Ok(split)
    }

    /// Feature count of the split's rows (0 only for an empty split, which
    /// `load` refuses to produce).
    pub fn width(&self) -> usize {
        self.features.first().map_or(0, Vec::len)
    }
}

/// Everything `run_session` needs to score sessions: one held-out split per
/// configured dataset and the sink the documents go to. One instance is
/// shared by every session of the process.
pub struct Evaluator {
    pub splits: HashMap<String, HeldOutSplit>,
    pub sink: Arc<dyn MetricSink>,
}

impl Evaluator {
    /// Load every configured split and build the Firestore sink. Any failure
    /// is a hard error surfaced from `serve`: a configured `[eval]` means
    /// this run exists to produce the learning curve, so a half-working
    /// evaluator must stop startup rather than train silently unmeasured.
    pub async fn from_config(cfg: &crate::agg_config::EvalConfig) -> anyhow::Result<Self> {
        let mut splits = HashMap::new();
        for (dataset_id, path) in &cfg.datasets {
            splits.insert(dataset_id.clone(), HeldOutSplit::load(path)?);
        }
        let sink = FirestoreWriter::from_config(
            cfg.project_id.clone(),
            cfg.service_account_path.clone(),
        )
        .await?;
        Ok(Self {
            splits,
            sink: Arc::new(sink),
        })
    }
}

/// One `paperSimTreeMetrics` row for `model`'s newest tree, the one
/// `pbr_core::Aggregator::finish_round` just pushed, so
/// `model.trees` is never empty here and `tree_idx` is its last index.
/// `ts` is the aggregator's clock at scoring time.
pub fn score_newest_tree(
    model: &pbr_core::Model,
    split: &HeldOutSplit,
    session_id: &str,
) -> TreeMetricDoc {
    TreeMetricDoc {
        session_id: session_id.to_string(),
        tree_idx: (model.trees.len() - 1) as u32,
        metrics: metrics::evaluate(model, &split.features, &split.targets),
        n_test: split.targets.len(),
        threshold_used: metrics::THRESHOLD,
        ts: chrono::Utc::now(),
    }
}

/// A destination for one per-tree metrics document. The real implementation
/// (`FirestoreWriter` in `firestore.rs`) writes to Firestore; tests collect
/// the documents in memory. A write error is surfaced so the caller can log
/// and continue rather than abort. Object-safe (`async_trait`) so the round
/// loop holds it as `Arc<dyn MetricSink>`.
#[tonic::async_trait]
pub trait MetricSink: Send + Sync {
    async fn write(&self, doc: &TreeMetricDoc) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> TreeMetricDoc {
        TreeMetricDoc {
            session_id: "sess-xyz".to_string(),
            tree_idx: 7,
            metrics: TreeMetrics {
                auc: 0.88,
                accuracy: 0.81,
                precision: 0.79,
                recall: 0.83,
                f1: 0.81,
                logloss: 0.42,
            },
            n_test: 61,
            threshold_used: 0.5,
            ts: "2026-07-18T12:34:56Z".parse().unwrap(),
        }
    }

    #[test]
    fn document_has_exactly_the_contract_field_set() {
        let doc = firestore::tree_metric_document(&sample_doc())
            .expect("TreeMetricDoc serializes to a Firestore document");
        assert!(
            doc.name.is_empty(),
            "create rejects a preset Document.name alongside generate_document_id"
        );
        let mut keys: Vec<&String> = doc.fields.keys().collect();
        keys.sort();
        let mut expected = vec![
            "sessionId",
            "treeIdx",
            "auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "logloss",
            "nTest",
            "thresholdUsed",
            "ts",
        ];
        expected.sort_unstable();
        assert_eq!(
            keys, expected,
            "paperSimTreeMetrics field set is a binding contract with analysis/fetch.py"
        );
    }

    #[test]
    fn document_uses_firestore_value_kinds() {
        use ::gcloud_sdk::google::firestore::v1::value::ValueType;

        let doc = firestore::tree_metric_document(&sample_doc())
            .expect("TreeMetricDoc serializes to a Firestore document");
        let kind_of = |field: &str| doc.fields[field].value_type.clone().unwrap();
        assert!(matches!(kind_of("sessionId"), ValueType::StringValue(_)));
        assert!(matches!(kind_of("treeIdx"), ValueType::IntegerValue(_)));
        assert!(matches!(kind_of("nTest"), ValueType::IntegerValue(_)));
        assert!(matches!(kind_of("auc"), ValueType::DoubleValue(_)));
        assert!(matches!(kind_of("thresholdUsed"), ValueType::DoubleValue(_)));
        assert!(matches!(kind_of("ts"), ValueType::TimestampValue(_)));
    }

    fn write_csv(dir: &tempfile::TempDir, rows: usize) -> std::path::PathBuf {
        let path = dir.path().join("t.csv");
        let mut s = String::from("f1,f2,target\n");
        for i in 0..rows {
            s.push_str(&format!("{}.0,0.5,{}\n", i, i % 2));
        }
        std::fs::write(&path, s).unwrap();
        path
    }

    #[test]
    fn held_out_split_is_the_last_20_percent() {
        let dir = tempfile::tempdir().unwrap();
        let split = HeldOutSplit::load(&write_csv(&dir, 10)).unwrap();
        // 10 rows -> split_idx 8 -> the file's last two rows.
        assert_eq!(split.targets.len(), 2);
        assert_eq!(split.features[0][0], 8.0);
        assert_eq!(split.width(), 2);
    }

    #[test]
    fn an_empty_held_out_split_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = HeldOutSplit::load(&write_csv(&dir, 0))
            .expect_err("a header-only csv has no held-out rows");
        assert!(err.to_string().contains("empty"), "got: {err}");
    }
}
