//! What one training session is, as opposed to what the cluster is.
//!
//! A `SessionSpec` carries what varies per session: the dataset, the boosting
//! hyperparameters, the round-close policy. Cluster-wide concerns (shareholder
//! endpoints, the Shamir threshold, auth, TLS, FCM) stay in `AggregatorConfig`,
//! since they belong to the deployment and cannot differ between two sessions
//! sharing one process.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// The datasets this cluster accepts sessions for, and each one's feature
/// width. A session is refused unless its dataset appears here, but only via
/// `is_some()`: the stored width is never compared against anything, so this
/// table cannot catch a dataset/width mismatch. The actual feature count is
/// learned later, in the stats phase (`SessionState::n_features`).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(transparent)]
pub struct DatasetTable(BTreeMap<String, u32>);

impl DatasetTable {
    pub fn from_pairs(pairs: Vec<(String, u32)>) -> Self {
        Self(pairs.into_iter().collect())
    }
    pub fn n_features(&self, dataset_id: &str) -> Option<u32> {
        self.0.get(dataset_id).copied()
    }
    pub fn ids(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
}

/// Upper bound on `submission_window_ms`. `await_submissions` gives up only
/// after `GIVE_UP_WINDOWS` windows, so the real per-round ceiling is ten times
/// this: a ten-hour budget on how long a round that never reaches
/// `min_clients` can pin a share pool, a task and internal-plane connections.
const MAX_SUBMISSION_WINDOW_MS: u64 = 60 * 60 * 1000;

/// Upper bound on `n_trees`. Unlike depth or bin count an oversized tree count
/// triggers no large allocation, so this bounds wall-clock time instead: every
/// tree spends at least one submission window per depth. The reference
/// heart_disease session in `pbr-e2e` uses 15 trees.
const MAX_N_TREES: usize = 1_000;

/// Upper bound on `max_depth`, the dangerous dimension:
/// `pbr_core::Aggregator::compute_splits` allocates one gradient and hessian
/// histogram (`n_features * n_bins` floats each) per active node, and the
/// active-node count doubles every level. This caps a round at 4096 node
/// histograms, bounding per-round memory even at `MAX_N_BINS`. The reference
/// session trains to depth 3.
const MAX_MAX_DEPTH: usize = 12;

/// Upper bound on `n_bins`, which is `uint32` on the wire: a request near
/// `u32::MAX` would ask `FeatureStats::to_bins` for a multi-gigabyte edge
/// vector per feature, and every histogram in `compute_splits` scales with it,
/// all before a single client submits. 1024 keeps both in the tens of KB.
const MAX_N_BINS: usize = 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionSpec {
    pub dataset_id: String,
    pub title: String,
    pub n_trees: usize,
    pub max_depth: usize,
    pub n_bins: usize,
    pub learning_rate: f64,
    pub lambda: f64,
    pub min_clients: usize,
    pub target_clients: usize,
    pub submission_window_ms: u64,
}

impl SessionSpec {
    pub fn submission_window(&self) -> Duration {
        Duration::from_millis(self.submission_window_ms)
    }

    /// Reject a spec that could never train: an unknown dataset, or a
    /// round-close policy with no reachable close condition.
    pub fn validate(&self, datasets: &DatasetTable) -> anyhow::Result<()> {
        anyhow::ensure!(
            datasets.n_features(&self.dataset_id).is_some(),
            "unknown dataset {}; this cluster accepts: {}",
            self.dataset_id,
            datasets.ids().join(", ")
        );
        self.validate_bounds()
    }

    /// The dataset-independent bounds every spec must satisfy. `validate` runs
    /// these plus the dataset table; `AggregatorHandle::create_session` runs
    /// only these, which is how a dataset-less session gets past the table
    /// check `validate` would fail it on.
    pub fn validate_bounds(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.min_clients >= 1, "min_clients must be >= 1");
        anyhow::ensure!(
            self.target_clients >= self.min_clients,
            "target_clients ({}) must be >= min_clients ({})",
            self.target_clients,
            self.min_clients
        );
        anyhow::ensure!(
            self.n_trees >= 1 && self.n_trees <= MAX_N_TREES,
            "n_trees must be in 1..={MAX_N_TREES}, got {}",
            self.n_trees
        );
        anyhow::ensure!(
            self.max_depth >= 1 && self.max_depth <= MAX_MAX_DEPTH,
            "max_depth must be in 1..={MAX_MAX_DEPTH}, got {}",
            self.max_depth
        );
        anyhow::ensure!(
            self.n_bins >= 2 && self.n_bins <= MAX_N_BINS,
            "n_bins must be in 2..={MAX_N_BINS}, got {}",
            self.n_bins
        );
        anyhow::ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0 && self.learning_rate <= 1.0,
            "learning_rate must be finite and in (0.0, 1.0], got {}",
            self.learning_rate
        );
        anyhow::ensure!(
            self.lambda.is_finite() && self.lambda >= 0.0,
            "lambda must be finite and >= 0.0, got {}",
            self.lambda
        );
        anyhow::ensure!(self.submission_window_ms > 0, "submission_window_ms must be > 0");
        anyhow::ensure!(
            self.submission_window_ms <= MAX_SUBMISSION_WINDOW_MS,
            "submission_window_ms ({}) exceeds the maximum of {MAX_SUBMISSION_WINDOW_MS}",
            self.submission_window_ms
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn datasets() -> DatasetTable {
        DatasetTable::from_pairs(vec![("heart_disease".into(), 13), ("pima_diabetes".into(), 8)])
    }

    fn spec(dataset: &str) -> SessionSpec {
        SessionSpec {
            dataset_id: dataset.into(),
            title: "t".into(),
            n_trees: 15,
            max_depth: 3,
            n_bins: 10,
            learning_rate: 0.15,
            lambda: 2.0,
            min_clients: 10,
            target_clients: 237,
            submission_window_ms: 5_000,
        }
    }

    #[test]
    fn a_spec_naming_an_unknown_dataset_is_rejected() {
        let err = spec("no_such_dataset").validate(&datasets()).expect_err(
            "a session over a dataset the cluster does not know cannot be trained",
        );
        assert!(err.to_string().contains("no_such_dataset"));
    }

    #[test]
    fn a_spec_over_a_known_dataset_validates() {
        spec("heart_disease").validate(&datasets()).expect("known dataset");
    }

    #[test]
    fn target_below_min_is_rejected() {
        let mut s = spec("heart_disease");
        s.min_clients = 20;
        s.target_clients = 5;
        let err = s.validate(&datasets()).expect_err(
            "a round could never close on target before settling for min",
        );
        assert!(err.to_string().contains("target_clients"));
    }

    #[test]
    fn a_zero_submission_window_is_rejected() {
        let mut s = spec("heart_disease");
        s.submission_window_ms = 0;
        assert!(s.validate(&datasets()).is_err(), "a zero window gives clients no time to submit");
    }

    #[test]
    fn a_submission_window_above_the_ceiling_is_rejected() {
        let mut s = spec("heart_disease");
        s.submission_window_ms = MAX_SUBMISSION_WINDOW_MS + 1;
        let err = s.validate(&datasets()).expect_err(
            "a round give-up budget of GIVE_UP_WINDOWS times an unbounded window could pin \
             cluster resources indefinitely",
        );
        assert!(err.to_string().contains("submission_window_ms"));
    }

    #[test]
    fn a_submission_window_at_the_ceiling_validates() {
        let mut s = spec("heart_disease");
        s.submission_window_ms = MAX_SUBMISSION_WINDOW_MS;
        s.validate(&datasets()).expect("the ceiling itself is still an allowed window");
    }

    #[test]
    fn n_trees_above_the_ceiling_is_rejected() {
        let mut s = spec("heart_disease");
        s.n_trees = MAX_N_TREES + 1;
        let err = s
            .validate(&datasets())
            .expect_err("far more trees than this system is characterised for must be refused");
        assert!(err.to_string().contains("n_trees"));
    }

    #[test]
    fn n_trees_at_the_ceiling_validates() {
        let mut s = spec("heart_disease");
        s.n_trees = MAX_N_TREES;
        s.validate(&datasets()).expect("the ceiling itself is still an allowed tree count");
    }

    #[test]
    fn max_depth_above_the_ceiling_is_rejected() {
        let mut s = spec("heart_disease");
        s.max_depth = MAX_MAX_DEPTH + 1;
        let err = s.validate(&datasets()).expect_err(
            "a depth this far beyond what this system is characterised for must be refused before \
             compute_splits doubles its active-node histogram count that many more times",
        );
        assert!(err.to_string().contains("max_depth"));
    }

    #[test]
    fn max_depth_at_the_ceiling_validates() {
        let mut s = spec("heart_disease");
        s.max_depth = MAX_MAX_DEPTH;
        s.validate(&datasets()).expect("the ceiling itself is still an allowed depth");
    }

    #[test]
    fn n_bins_above_the_ceiling_is_rejected() {
        let mut s = spec("heart_disease");
        s.n_bins = MAX_N_BINS + 1;
        let err = s.validate(&datasets()).expect_err(
            "an oversized bin count must be refused before FeatureStats::to_bins allocates a huge \
             edge vector",
        );
        assert!(err.to_string().contains("n_bins"));
    }

    #[test]
    fn n_bins_at_the_ceiling_validates() {
        let mut s = spec("heart_disease");
        s.n_bins = MAX_N_BINS;
        s.validate(&datasets()).expect("the ceiling itself is still an allowed bin count");
    }

    #[test]
    fn a_non_positive_learning_rate_is_rejected() {
        let mut s = spec("heart_disease");
        s.learning_rate = -1.0;
        let err = s
            .validate(&datasets())
            .expect_err("a non-positive rate cannot learn");
        assert!(err.to_string().contains("learning_rate"));
    }

    #[test]
    fn a_learning_rate_above_one_is_rejected() {
        let mut s = spec("heart_disease");
        s.learning_rate = 1.5;
        let err = s
            .validate(&datasets())
            .expect_err("a rate above 1 is outside anything this system has been characterised for");
        assert!(err.to_string().contains("learning_rate"));
    }

    #[test]
    fn a_nan_learning_rate_is_rejected() {
        let mut s = spec("heart_disease");
        s.learning_rate = f64::NAN;
        assert!(
            s.validate(&datasets()).is_err(),
            "NaN fails every ordered comparison, so the (0.0, 1.0] range check alone would silently \
             pass it; is_finite() is what actually rejects it"
        );
    }

    #[test]
    fn a_negative_lambda_is_rejected() {
        let mut s = spec("heart_disease");
        s.lambda = -100.0;
        let err = s
            .validate(&datasets())
            .expect_err("a negative L2 penalty would invert the regularisation");
        assert!(err.to_string().contains("lambda"));
    }

    #[test]
    fn a_nan_lambda_is_rejected() {
        let mut s = spec("heart_disease");
        s.lambda = f64::NAN;
        assert!(
            s.validate(&datasets()).is_err(),
            "NaN fails every ordered comparison, so the >= 0.0 check alone would silently pass it; \
             is_finite() is what actually rejects it"
        );
    }

    #[test]
    fn a_sane_learning_rate_and_lambda_validate() {
        let s = spec("heart_disease"); // learning_rate 0.15, lambda 2.0
        s.validate(&datasets())
            .expect("both new bounds are satisfied by the values every other test relies on");
    }

    fn base_aggregator_config_toml() -> String {
        r#"
listen = "127.0.0.1:7000"
internal_shareholder_endpoints = ["http://127.0.0.1:7101"]
client_shareholder_endpoints = ["http://127.0.0.1:7201"]
threshold = 1
state_path = ":memory:"

[auth]
issuer = "https://issuer.local"
audience = "pbr"
static_keys = []
"#
        .to_string()
    }

    #[test]
    fn dataset_table_parses_pairs_from_toml() {
        let table: DatasetTable =
            toml::from_str("heart_disease = 13\npima_diabetes = 8\n").expect("valid [datasets] body");
        assert_eq!(table.n_features("heart_disease"), Some(13));
        assert_eq!(table.n_features("pima_diabetes"), Some(8));
    }

    #[test]
    fn a_datasets_section_parses_through_the_full_aggregator_config() {
        let toml = format!(
            "{}\n[datasets]\nheart_disease = 13\npima_diabetes = 8\n",
            base_aggregator_config_toml()
        );
        let cfg: crate::agg_config::AggregatorConfig =
            toml::from_str(&toml).expect("a trailing [datasets] section must parse");
        assert_eq!(cfg.datasets.n_features("heart_disease"), Some(13));
        assert_eq!(cfg.datasets.n_features("pima_diabetes"), Some(8));
    }

    #[test]
    fn no_datasets_section_yields_an_empty_table() {
        let cfg: crate::agg_config::AggregatorConfig = toml::from_str(&base_aggregator_config_toml())
            .expect("a config with no [datasets] section must still parse");
        assert!(
            cfg.datasets.ids().is_empty(),
            "every committed deploy config omits [datasets] and relies on getting an empty table, \
             not a parse error"
        );
    }
}
