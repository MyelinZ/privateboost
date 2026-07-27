use crate::aggregator::DatasetTable;
use crate::config::{AuthConfig, TlsConfig};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone)]
pub struct AggregatorConfig {
    pub listen: SocketAddr,
    /// One internal (localhost-only) endpoint per shareholder;
    /// `internal_shareholder_endpoints[i]` serves Shamir evaluation point
    /// x = i + 1, matching the client fan-out convention.
    pub internal_shareholder_endpoints: Vec<String>,
    /// One CLIENT-FACING endpoint per shareholder, same x = i + 1 ordering
    /// as `internal_shareholder_endpoints` but DISTINCT from it: this is
    /// where clients submit shares, not where the aggregator gathers from.
    /// Handed to clients verbatim via `EnrollSession`'s `SessionConfig`.
    pub client_shareholder_endpoints: Vec<String>,
    pub threshold: usize,
    pub auth: AuthConfig,
    /// Round-open push notifications via FCM HTTP v1. Absent (as in the
    /// `deploy/local` and e2e configs, which carry no `[fcm]` section)
    /// disables notify entirely: the tick never spawns and no push is sent.
    #[serde(default)]
    pub fcm: Option<FcmConfig>,
    /// TLS for the CLIENT-FACING listener only. Absent leaves it plaintext;
    /// the internal loopback plane is never wrapped in TLS.
    #[serde(default)]
    pub tls: Option<TlsConfig>,
    /// Datasets this cluster accepts sessions for (`id = n_features`). Empty
    /// by default, which only affects `CreateSession`: a spec whose
    /// `dataset_id` is not in this table is rejected.
    #[serde(default)]
    pub datasets: DatasetTable,
    /// Bearer token guarding the admin plane. `None` disables `CreateSession`
    /// entirely, the safe default for a cluster that should not be
    /// administered remotely.
    #[serde(default)]
    pub admin_token: Option<String>,
    /// Where the session list is persisted, as a SQLite database, so a
    /// restart re-serves its history (in-flight sessions demoted to `Failed`).
    /// Mandatory: a config omitting it fails to parse. `":memory:"` gives an
    /// ephemeral store backed by no file, which the tests use.
    pub state_path: PathBuf,
    /// Per-tree quality evaluation written to Firestore. Absent (every
    /// committed config except the fleet deploy) disables scoring entirely;
    /// see `crate::eval`. Unlike `[fcm]`, a configured-but-unbuildable
    /// evaluator fails startup: a run with `[eval]` set exists to produce
    /// the learning curve, so it must not train while silently unmeasured.
    #[serde(default)]
    pub eval: Option<EvalConfig>,
}

fn default_interval_minutes() -> u64 {
    15
}

#[derive(Deserialize, Clone)]
pub struct FcmConfig {
    pub project_id: String,
    /// A service-account JSON key file; `None` uses Application Default
    /// Credentials (see `crate::fcm::FcmSender::from_config`).
    #[serde(default)]
    pub service_account_path: Option<String>,
    /// Minutes between round-open pushes to one account, both the floor the
    /// notify tick enforces and the TTL stamped on each message, so an
    /// undelivered push expires once the next eligible one would supersede it.
    /// 15 matches the app's WorkManager cadence and Android's rare-bucket
    /// high-priority quota.
    #[serde(default = "default_interval_minutes")]
    pub interval_minutes: u64,
}

#[derive(Deserialize, Clone)]
pub struct EvalConfig {
    pub project_id: String,
    /// A service-account JSON key file; `None` uses Application Default
    /// Credentials (see `crate::eval::firestore::FirestoreWriter`).
    #[serde(default)]
    pub service_account_path: Option<String>,
    /// dataset_id -> held-out test CSV. A session is scored iff its
    /// `dataset_id` has an entry. Each CSV's last 20% is the held-out
    /// split; its first 80% must be exactly what that dataset's clients
    /// train on, or held-out rows overlap training rows and the curve is
    /// meaningless. A quoted empty key (`"" = ...`) matches a dataset-less
    /// session, one created with an empty `dataset_id` as the in-process tests
    /// do.
    pub datasets: BTreeMap<String, PathBuf>,
}

impl AggregatorConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let cfg = toml::from_str::<AggregatorConfig>(&std::fs::read_to_string(path)?)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validates the cluster-level fields only: endpoint counts, the Shamir
    /// threshold, and a non-empty `[eval.datasets]` when
    /// `[eval]` is set. Per-session training parameters live in
    /// `SessionSpec` and are validated at session creation by
    /// `SessionSpec::validate`.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.internal_shareholder_endpoints.is_empty() {
            anyhow::bail!("internal_shareholder_endpoints must not be empty");
        }
        if self.client_shareholder_endpoints.len() != self.internal_shareholder_endpoints.len() {
            anyhow::bail!(
                "client_shareholder_endpoints must have one entry per internal_shareholder_endpoints \
                 entry (got {} vs {})",
                self.client_shareholder_endpoints.len(),
                self.internal_shareholder_endpoints.len()
            );
        }
        if self.threshold == 0 || self.threshold > self.internal_shareholder_endpoints.len() {
            anyhow::bail!(
                "threshold must be in 1..={}",
                self.internal_shareholder_endpoints.len()
            );
        }
        if let Some(eval) = &self.eval
            && eval.datasets.is_empty()
        {
            anyhow::bail!("eval.datasets must not be empty when [eval] is configured");
        }
        if let Some(fcm) = &self.fcm {
            anyhow::ensure!(
                fcm.interval_minutes > 0,
                "fcm.interval_minutes must be greater than 0"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn minimal_config() -> AggregatorConfig {
        AggregatorConfig {
            listen: "127.0.0.1:0".parse().unwrap(),
            internal_shareholder_endpoints: vec!["http://127.0.0.1:1".into()],
            client_shareholder_endpoints: vec!["http://127.0.0.1:2".into()],
            threshold: 2,
            auth: AuthConfig {
                issuer: "i".into(),
                audience: "a".into(),
                static_keys: vec![],
                google_jwks_url: None,
            },
            fcm: None,
            tls: None,
            datasets: DatasetTable::default(),
            admin_token: None,
            state_path: ":memory:".into(),
            eval: None,
        }
    }

    #[test]
    fn parses_toml_and_validates() {
        let toml = r#"
listen = "127.0.0.1:7000"
internal_shareholder_endpoints = ["http://127.0.0.1:7101", "http://127.0.0.1:7102"]
client_shareholder_endpoints = ["http://127.0.0.1:7201", "http://127.0.0.1:7202"]
threshold = 2
state_path = ":memory:"

[auth]
issuer = "https://issuer.local"
audience = "pbr"
static_keys = [{ kid = "k1", public_key_pem_path = "/tmp/k.pub.pem" }]
"#;
        let cfg: AggregatorConfig = toml::from_str(toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.internal_shareholder_endpoints.len(), 2);
        assert_eq!(cfg.threshold, 2);
        assert_eq!(cfg.auth.google_jwks_url, None);
    }

    #[test]
    fn google_jwks_url_parses_when_present() {
        let toml = r#"
listen = "127.0.0.1:7000"
internal_shareholder_endpoints = ["http://127.0.0.1:7101"]
client_shareholder_endpoints = ["http://127.0.0.1:7201"]
threshold = 1
state_path = ":memory:"

[auth]
issuer = "https://securetoken.google.com/pboost-test-12345"
audience = "pboost-test-12345"
static_keys = []
google_jwks_url = "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"
"#;
        let cfg: AggregatorConfig = toml::from_str(toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            cfg.auth.google_jwks_url.as_deref(),
            Some(
                "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"
            )
        );
    }

    #[test]
    fn rejects_threshold_above_shareholder_count() {
        let cfg = minimal_config();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn fcm_section_absent_by_default_present_when_configured() {
        let base = r#"
listen = "127.0.0.1:7000"
internal_shareholder_endpoints = ["http://127.0.0.1:7101"]
client_shareholder_endpoints = ["http://127.0.0.1:7201"]
threshold = 1
state_path = ":memory:"

[auth]
issuer = "https://issuer.local"
audience = "pbr"
static_keys = []
"#;
        let cfg: AggregatorConfig = toml::from_str(base).unwrap();
        assert!(
            cfg.fcm.is_none(),
            "a config with no [fcm] section must leave notify disabled"
        );

        let with_fcm = format!(
            "{base}\n[fcm]\nproject_id = \"pboost-test-12345\"\nservice_account_path = \"/tmp/sa.json\"\n"
        );
        let cfg: AggregatorConfig = toml::from_str(&with_fcm).unwrap();
        let fcm = cfg.fcm.expect("[fcm] section must parse");
        assert_eq!(fcm.project_id, "pboost-test-12345");
        assert_eq!(fcm.service_account_path.as_deref(), Some("/tmp/sa.json"));
        assert_eq!(
            fcm.interval_minutes, 15,
            "interval_minutes must default to 15 when absent"
        );

        let with_fcm_adc = format!("{base}\n[fcm]\nproject_id = \"pboost-test-12345\"\n");
        let cfg: AggregatorConfig = toml::from_str(&with_fcm_adc).unwrap();
        let fcm = cfg
            .fcm
            .expect("[fcm] section must parse without service_account_path");
        assert_eq!(fcm.service_account_path, None);
        assert_eq!(
            fcm.interval_minutes, 15,
            "interval_minutes must default to 15 when absent"
        );

        let with_interval = format!(
            "{base}\n[fcm]\nproject_id = \"p\"\ninterval_minutes = 30\n"
        );
        let cfg: AggregatorConfig = toml::from_str(&with_interval).unwrap();
        assert_eq!(cfg.fcm.expect("[fcm] must parse").interval_minutes, 30);
    }

    #[test]
    fn state_path_parses_to_the_configured_path() {
        let toml = r#"
listen = "127.0.0.1:7000"
internal_shareholder_endpoints = ["http://127.0.0.1:7101"]
client_shareholder_endpoints = ["http://127.0.0.1:7201"]
threshold = 1
state_path = "/var/lib/pbr/sessions.sqlite"

[auth]
issuer = "https://issuer.local"
audience = "pbr"
static_keys = []
"#;
        let cfg: AggregatorConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.state_path, PathBuf::from("/var/lib/pbr/sessions.sqlite"));
    }

    #[test]
    fn a_config_without_state_path_fails_to_parse() {
        // The session store is always on, so there is no in-memory fallback to
        // default to: a config that names no path is a configuration mistake,
        // not a silently ephemeral cluster.
        let toml = r#"
listen = "127.0.0.1:7000"
internal_shareholder_endpoints = ["http://127.0.0.1:7101"]
client_shareholder_endpoints = ["http://127.0.0.1:7201"]
threshold = 1

[auth]
issuer = "https://issuer.local"
audience = "pbr"
static_keys = []
"#;
        assert!(
            toml::from_str::<AggregatorConfig>(toml).is_err(),
            "a missing state_path must be a parse error, not a default"
        );
    }

    /// `minimal_config()` itself fails `validate()` (its threshold of 2
    /// exceeds its single shareholder, on purpose, for
    /// `rejects_threshold_above_shareholder_count`);
    /// `rejects_an_eval_section_with_no_datasets` needs a config that
    /// validates before its `[eval]` section is broken.
    fn valid_config() -> AggregatorConfig {
        AggregatorConfig {
            threshold: 1,
            ..minimal_config()
        }
    }

    #[test]
    fn eval_section_absent_by_default_present_when_configured() {
        let base = r#"
listen = "127.0.0.1:7000"
internal_shareholder_endpoints = ["http://127.0.0.1:7101"]
client_shareholder_endpoints = ["http://127.0.0.1:7201"]
threshold = 1
state_path = ":memory:"

[auth]
issuer = "https://issuer.local"
audience = "pbr"
static_keys = []
"#;
        let cfg: AggregatorConfig = toml::from_str(base).unwrap();
        assert!(
            cfg.eval.is_none(),
            "a config with no [eval] section must leave evaluation disabled"
        );

        let with_eval = format!(
            "{base}\n[eval]\nproject_id = \"pboost-test-12345\"\n\
             service_account_path = \"/secrets/firestore-sa.json\"\n\n\
             [eval.datasets]\nheart_disease = \"/data/heart_disease.csv\"\n"
        );
        let cfg: AggregatorConfig = toml::from_str(&with_eval).unwrap();
        let eval = cfg.eval.expect("[eval] section must parse");
        assert_eq!(eval.project_id, "pboost-test-12345");
        assert_eq!(
            eval.service_account_path.as_deref(),
            Some("/secrets/firestore-sa.json")
        );
        assert_eq!(
            eval.datasets.get("heart_disease"),
            Some(&PathBuf::from("/data/heart_disease.csv"))
        );

        let with_eval_adc = format!(
            "{base}\n[eval]\nproject_id = \"pboost-test-12345\"\n\n\
             [eval.datasets]\nheart_disease = \"/data/heart_disease.csv\"\n"
        );
        let cfg: AggregatorConfig = toml::from_str(&with_eval_adc).unwrap();
        let eval = cfg
            .eval
            .expect("[eval] section must parse without service_account_path");
        assert_eq!(eval.service_account_path, None);
    }

    #[test]
    fn rejects_an_eval_section_with_no_datasets() {
        let mut cfg = valid_config();
        cfg.eval = Some(EvalConfig {
            project_id: "p".into(),
            service_account_path: None,
            datasets: std::collections::BTreeMap::new(),
        });
        let err = cfg
            .validate()
            .expect_err("an [eval] section that scores nothing is a config mistake");
        assert!(err.to_string().contains("eval.datasets"));
    }

    #[test]
    fn rejects_a_zero_interval_minutes() {
        let mut cfg = valid_config();
        cfg.fcm = Some(FcmConfig {
            project_id: "p".into(),
            service_account_path: None,
            interval_minutes: 0,
        });
        let err = cfg
            .validate()
            .expect_err("a zero interval_minutes defeats the notify throttle");
        assert!(err.to_string().contains("fcm.interval_minutes"));
    }
}
