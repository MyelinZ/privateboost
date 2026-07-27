//! Operator CLI for `AdminService`: create, list, and delete sessions on a
//! running aggregator.
//! The admin bearer token is never accepted as a flag: it comes from
//! `PBR_ADMIN_TOKEN` only, so it cannot land in shell history or `ps` output.

use clap::{Parser, Subcommand};
use pbr_proto::v1::{CreateSessionRequest, DeleteSessionRequest, ListSessionsRequest};
use pbr_proto::v1::admin_service_client::AdminServiceClient;
use std::time::Duration;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

/// Matches `client_endpoint` in `pbr-client`: generous yet finite, so a
/// blackholed aggregator fails the connect instead of hanging.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Matches `client_endpoint` in `pbr-client`, so an unresponsive but connected
/// aggregator bounds `create_session` to one timeout.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Parser)]
#[command(name = "pbr-admin")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Schedule a new training session on an aggregator.
    CreateSession(CreateSessionArgs),
    /// Delete a session by id; a live session's round loop is aborted.
    DeleteSession(DeleteSessionArgs),
    /// List every session the aggregator hosts.
    ListSessions(ListSessionsArgs),
}

impl Command {
    fn connect_args(&self) -> &ConnectArgs {
        match self {
            Command::CreateSession(a) => &a.connect,
            Command::DeleteSession(a) => &a.connect,
            Command::ListSessions(a) => &a.connect,
        }
    }
}

/// Connection flags every subcommand takes, flattened so `--aggregator` and
/// `--ca-cert` keep their existing names and positions.
#[derive(clap::Args)]
struct ConnectArgs {
    /// Aggregator base URL, e.g. https://aggregator.example:42800
    #[arg(long)]
    aggregator: String,

    /// Path to a PEM CA certificate to pin for TLS (use an https:// aggregator
    /// URL). Omitted leaves the connection plaintext (loopback testing only).
    #[arg(long)]
    ca_cert: Option<std::path::PathBuf>,
}

#[derive(clap::Args)]
struct DeleteSessionArgs {
    #[command(flatten)]
    connect: ConnectArgs,

    /// Id of the session to delete (discover ids with list-sessions).
    #[arg(long)]
    session_id: String,
}

#[derive(clap::Args)]
struct ListSessionsArgs {
    #[command(flatten)]
    connect: ConnectArgs,
}

#[derive(clap::Args)]
struct CreateSessionArgs {
    #[command(flatten)]
    connect: ConnectArgs,

    /// Dataset id; must be one this aggregator's [datasets] table accepts.
    #[arg(long)]
    dataset: String,

    /// Session title shown alongside its id; purely descriptive.
    #[arg(long, default_value = "")]
    title: String,

    /// Number of boosting rounds. Default matches the heart_disease
    /// reference configuration `pbr-e2e` trains and gates on.
    #[arg(long, default_value_t = 15)]
    trees: u32,

    /// Max tree depth. Default matches the heart_disease reference
    /// configuration.
    #[arg(long, default_value_t = 3)]
    depth: u32,

    /// Histogram bins per feature. Default matches the heart_disease
    /// reference configuration.
    #[arg(long, default_value_t = 10)]
    bins: u32,

    /// Learning rate. Default matches the heart_disease reference
    /// configuration.
    #[arg(long, default_value_t = 0.15)]
    lr: f64,

    /// L2 regularization. Default matches the heart_disease reference
    /// configuration.
    #[arg(long, default_value_t = 2.0)]
    lambda: f64,

    /// Round-close target: a round waits for submissions until at least
    /// this many distinct clients have contributed, then closes. It is not
    /// the privacy floor; each shareholder enforces its own min_clients,
    /// fixed at deployment, which no session parameter can lower. Set this
    /// below a shareholder's floor and the round closes anyway, then the
    /// gather fails once reconstruction comes up short. There is no
    /// default: choose it deliberately, matched to what the deployed
    /// shareholders require.
    #[arg(long)]
    min_clients: u32,

    /// Client commitments a round closes early upon reaching, without
    /// waiting for the submission deadline. Defaults to --min-clients.
    #[arg(long)]
    target_clients: Option<u32>,

    /// Milliseconds a round waits for submissions before applying the
    /// min_clients/target_clients close policy.
    #[arg(long, default_value_t = 5000)]
    window_ms: u64,
}

impl CreateSessionArgs {
    /// Maps the flags this subcommand carries onto the wire request. Argument
    /// bounds (dataset validity, window limits, live-session cap) are the
    /// server's job, not this CLI's, which only maps types.
    fn to_request(&self) -> anyhow::Result<CreateSessionRequest> {
        Ok(CreateSessionRequest {
            dataset_id: self.dataset.clone(),
            title: self.title.clone(),
            n_trees: self.trees,
            max_depth: self.depth,
            n_bins: self.bins,
            learning_rate: self.lr,
            lambda: self.lambda,
            min_clients: self.min_clients,
            target_clients: self.target_clients.unwrap_or(self.min_clients),
            submission_window_ms: self.window_ms,
        })
    }
}

/// `var` is the raw `PBR_ADMIN_TOKEN` value (`None` if unset). A missing or
/// empty value is a clear, named error, never an empty `Bearer` header sent
/// to the server.
fn admin_token_from(var: Option<String>) -> anyhow::Result<String> {
    match var {
        Some(token) if !token.is_empty() => Ok(token),
        _ => anyhow::bail!("PBR_ADMIN_TOKEN must be set to a non-empty admin bearer token"),
    }
}

/// Installs the ring provider as rustls's process-wide default, exactly
/// once, matching `pbr-client`'s `ensure_crypto_provider`: the dependency
/// tree links both `ring` and `aws-lc-rs`, so rustls cannot pick a default on
/// its own and `ClientTlsConfig` would otherwise panic.
fn ensure_crypto_provider() {
    static INSTALL_CRYPTO: std::sync::Once = std::sync::Once::new();
    INSTALL_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Connects to `aggregator`, pinning TLS to `ca_pem` when given (the URL must
/// then be `https://`); omitting it leaves the connection plaintext.
async fn connect(aggregator: &str, ca_pem: Option<&[u8]>) -> anyhow::Result<Channel> {
    let mut endpoint = Endpoint::from_shared(aggregator.to_string())?
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT);
    if let Some(ca) = ca_pem {
        ensure_crypto_provider();
        endpoint = endpoint
            .tls_config(ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca)))?;
    }
    Ok(endpoint.connect().await?)
}

/// Wrap `msg` in a request carrying the admin bearer token.
fn with_auth<T>(msg: T, token: &str) -> anyhow::Result<tonic::Request<T>> {
    let mut req = tonic::Request::new(msg);
    req.metadata_mut()
        .insert("authorization", format!("Bearer {token}").parse()?);
    Ok(req)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let token = admin_token_from(std::env::var("PBR_ADMIN_TOKEN").ok())?;

    let conn = args.command.connect_args();
    let ca_pem = conn.ca_cert.as_ref().map(std::fs::read).transpose()?;
    let channel = connect(&conn.aggregator, ca_pem.as_deref()).await?;
    let mut client = AdminServiceClient::new(channel);

    match &args.command {
        Command::CreateSession(a) => {
            let summary = client
                .create_session(with_auth(a.to_request()?, &token)?)
                .await?
                .into_inner();
            println!("{}", summary.session_id);
            println!("{:?}", summary.phase());
        }
        Command::DeleteSession(a) => {
            client
                .delete_session(with_auth(
                    DeleteSessionRequest {
                        session_id: a.session_id.clone(),
                    },
                    &token,
                )?)
                .await?;
            println!("deleted {}", a.session_id);
        }
        Command::ListSessions(_) => {
            let list = client
                .list_sessions(with_auth(ListSessionsRequest {}, &token)?)
                .await?
                .into_inner();
            // One row per session: id, phase, dataset (blank for a
            // dataset-less session), RFC 3339 creation time.
            for s in &list.sessions {
                let created = s
                    .created_at
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "-".into());
                println!(
                    "{}\t{:?}\t{}\t{}",
                    s.session_id,
                    s.phase(),
                    s.dataset_id,
                    created
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_map_to_the_request_with_documented_defaults() {
        let args = Args::parse_from([
            "pbr-admin",
            "create-session",
            "--aggregator",
            "https://example:42800",
            "--dataset",
            "heart_disease",
            "--min-clients",
            "10",
        ]);
        let Command::CreateSession(a) = args.command else {
            panic!("must parse as create-session");
        };
        let req = a.to_request().expect("valid args");
        assert_eq!(req.dataset_id, "heart_disease");
        assert_eq!(req.n_trees, 15, "default matches the paper's configuration");
        assert_eq!(req.max_depth, 3);
        assert_eq!(req.n_bins, 10);
        assert!((req.learning_rate - 0.15).abs() < 1e-9);
    }

    #[test]
    fn target_clients_defaults_to_min_clients() {
        let args = Args::parse_from([
            "pbr-admin",
            "create-session",
            "--aggregator",
            "https://example:42800",
            "--dataset",
            "heart_disease",
            "--min-clients",
            "10",
        ]);
        let Command::CreateSession(a) = args.command else {
            panic!("must parse as create-session");
        };
        let req = a.to_request().expect("valid args");
        assert_eq!(req.min_clients, 10);
        assert_eq!(req.target_clients, 10);
    }

    #[test]
    fn omitting_min_clients_is_a_parse_failure() {
        let result = Args::try_parse_from([
            "pbr-admin",
            "create-session",
            "--aggregator",
            "https://example:42800",
            "--dataset",
            "heart_disease",
        ]);
        assert!(
            result.is_err(),
            "min_clients must be required, not defaulted"
        );
    }

    #[test]
    fn a_missing_admin_token_is_a_clear_error_not_an_empty_bearer() {
        let err = admin_token_from(None).expect_err("must not send an empty bearer");
        assert!(err.to_string().contains("PBR_ADMIN_TOKEN"));
    }

    #[test]
    fn an_empty_admin_token_is_also_rejected() {
        let err = admin_token_from(Some(String::new()))
            .expect_err("empty string must not become a Bearer");
        assert!(err.to_string().contains("PBR_ADMIN_TOKEN"));
    }

    #[test]
    fn delete_session_requires_a_session_id() {
        let result = Args::try_parse_from([
            "pbr-admin",
            "delete-session",
            "--aggregator",
            "https://example:42800",
        ]);
        assert!(result.is_err(), "delete-session without --session-id must not parse");
    }

    #[test]
    fn delete_session_parses_its_flags() {
        let args = Args::parse_from([
            "pbr-admin",
            "delete-session",
            "--aggregator",
            "https://example:42800",
            "--session-id",
            "sess-1",
        ]);
        let Command::DeleteSession(a) = args.command else {
            panic!("must parse as delete-session");
        };
        assert_eq!(a.session_id, "sess-1");
        assert_eq!(a.connect.aggregator, "https://example:42800");
    }

    #[test]
    fn list_sessions_parses_with_connection_flags_only() {
        let args = Args::parse_from([
            "pbr-admin",
            "list-sessions",
            "--aggregator",
            "https://example:42800",
        ]);
        let Command::ListSessions(a) = args.command else {
            panic!("must parse as list-sessions");
        };
        assert_eq!(a.connect.aggregator, "https://example:42800");
        assert!(a.connect.ca_cert.is_none());
    }
}
