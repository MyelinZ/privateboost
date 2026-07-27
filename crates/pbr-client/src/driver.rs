//! Autonomous polling client driver.
//!
//! [`ClientSession`] carries one client through a session one
//! [`ClientSession::step`] at a time: poll, compute, best-effort fan-out
//! submit, exactly once per call, with no sleeping or looping inside `step`.
//! That is the seam a silent wake or a foreground timer drives.
//! [`run_collecting`] loops it to completion and returns wire byte totals;
//! [`run_to_completion`] is the shim for callers that want neither the cadence
//! nor the totals.
//!
//! One dead shareholder must not abort a client. A round counts as delivered
//! once `threshold` shareholders acknowledge it, since reconstruction needs no
//! more shares than that, and falling short is logged rather than fatal: the
//! client polls on to the next round, as the aggregator's own loop does. It
//! never asks a client to retry a round.

use crate::rpc::{Bearer, Shareholders};
use crate::submit::DeliveryReport;
use crate::wire_metrics::WireCounters;
use pbr_proto::convert::{edges_to_bin_config, model_from_proto, split_from_proto};
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_proto::v1::poll_session_response::Body;
use pbr_proto::v1::{
    ClientPlatform as WireClientPlatform, EnrollRequest, ListSessionsRequest, PollSessionRequest,
    RegisterDeviceRequest, SessionPhase as WireSessionPhase,
};
use pbr_core::{Client, Loss, RoundContext as CoreRoundContext};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tonic::Request;
use tonic::transport::Channel;

/// Fallback poll cadence when the server does not supply `next_poll_after`.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Floor on the poll cadence so a zero/near-zero server hint cannot spin.
const MIN_POLL_INTERVAL: Duration = Duration::from_millis(10);
/// Consecutive transient `step` failures tolerated before a session is
/// aborted. A single timed-out poll on a flaky radio must not end a whole
/// participation; a persistent fault still aborts within a few polls.
const MAX_CONSECUTIVE_FAILURES: u32 = 3;

static CLIENT_SEQ: AtomicU64 = AtomicU64::new(0);

/// Salts the per-round commitment hash (`pbr_core::commit`). Unlinkability
/// comes from a fresh OS-random nonce each round, so this need not be globally
/// unique.
fn next_client_id() -> String {
    let seq = CLIENT_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("pbr-client-{}-{seq}", std::process::id())
}

/// Connect an aggregator channel with the same deadlines the shareholder
/// fan-out uses, and attach a [`Bearer`] interceptor stamping `token` on every
/// RPC over it.
///
/// The interceptor pins the header for the channel's lifetime, which is only
/// correct because a session's token cannot change once the channel is built.
/// Refreshing a token mid-session would need a per-request insert instead.
async fn connect_aggregator(
    agg_endpoint: &str,
    token: &str,
    ca_pem: Option<&[u8]>,
    counters: Arc<WireCounters>,
) -> anyhow::Result<
    AggregatorServiceClient<tonic::service::interceptor::InterceptedService<Channel, Bearer>>,
> {
    let endpoint = crate::rpc::client_endpoint(agg_endpoint, ca_pem)?;
    let channel = crate::rpc::connect_counted(endpoint, counters).await?;
    Ok(AggregatorServiceClient::with_interceptor(
        channel,
        Bearer::new(token)?,
    ))
}

/// Which kind of round a wire round id denotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundKind {
    /// The one stats round that opens a session (Gaussian bin definition).
    Stats,
    /// A per-tree, per-depth gradient/hessian round.
    Gradient,
}

/// A wire round id decoded into its kind and, for a gradient round, the tree
/// and depth it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedRound {
    pub kind: RoundKind,
    pub tree_idx: Option<u32>,
    pub depth: Option<u32>,
}

/// Must match `pbr_server::aggregator::context`, or a decoded round is
/// labelled with the wrong tree and depth in telemetry.
const STATS_ROUND_ID: u64 = 1;
const COMPLETED_DEPTH_SENTINEL: u32 = u32::MAX;

/// Decode a wire round id into its kind and, for a gradient round, its tree and
/// depth. The packing is `gradient_round_id(tree, depth) =
/// ((tree + 1) << 32) | depth`, with stats pinned at 1. Returns `None` for the
/// completed sentinel, which carries no contribution to summarise. This is the
/// single source of truth: the bridge does not re-derive tree and depth in
/// Dart.
pub fn decode_round_id(round_id: u64) -> Option<DecodedRound> {
    if round_id == STATS_ROUND_ID {
        return Some(DecodedRound {
            kind: RoundKind::Stats,
            tree_idx: None,
            depth: None,
        });
    }
    let depth = (round_id & 0xFFFF_FFFF) as u32;
    if depth == COMPLETED_DEPTH_SENTINEL {
        return None;
    }
    Some(DecodedRound {
        kind: RoundKind::Gradient,
        tree_idx: Some(((round_id >> 32) - 1) as u32),
        depth: Some(depth),
    })
}

/// The result of exactly one [`ClientSession::step`] call.
#[derive(Debug)]
pub enum StepOutcome {
    /// A round's context was polled, computed, and (best-effort) submitted.
    Submitted {
        round_id: u64,
        /// From enrolment. Carried so mobile telemetry keys each round on the
        /// id the aggregator's per-tree metrics carry, making the two
        /// collections joinable.
        session_id: String,
        report: DeliveryReport,
        round_kind: RoundKind,
        /// Microseconds waiting on the `PollSession` RPC that surfaced this
        /// context. Microseconds throughout, because a small batch's totals are
        /// often under 1 ms, which `as_millis` would floor to 0.
        poll_us: u64,
        /// Microseconds computing this round's shares, summed over the device's
        /// whole batch.
        compute_us: u64,
        /// Microseconds on the best-effort shareholder fan-out submit.
        submit_us: u64,
        /// Bytes written over this whole step, across the aggregator channel
        /// and the fan-out. Below TLS, so ciphertext on an `https://` endpoint.
        tx_bytes: u64,
        /// Bytes read over this whole step.
        rx_bytes: u64,
        /// The contribution reached fewer than `threshold` shareholders. The
        /// client moves on regardless; the bridge marks the round
        /// `BelowThreshold`.
        below_threshold: bool,
        /// How many records this one client folded into its commitment. The
        /// same on every `Submitted` a session emits.
        n_records: u32,
    },
    /// No new round context; the caller waits `next_poll_after` before
    /// stepping again. Telemetry records it as an idle check-in, so there are
    /// no compute or submit fields.
    NothingNew {
        next_poll_after: Duration,
        session_id: String,
        /// As `Submitted::poll_us`.
        poll_us: u64,
    },
    /// The aggregator reports the session `COMPLETED`; no further steps.
    Completed,
    /// The aggregator reports the session `FAILED`; no further steps.
    Failed,
}

/// One client's live session, driven one [`step`](Self::step) at a time.
pub struct ClientSession {
    agg: AggregatorServiceClient<tonic::service::interceptor::InterceptedService<Channel, Bearer>>,
    fanout: Shareholders,
    client: Client,
    session_id: String,
    threshold: usize,
    hide_path: bool,
    n_records: u32,
    last_seen: u64,
    /// Shared by the aggregator channel and the whole fan-out, so a round's
    /// delta is all the ciphertext this session moved.
    wire: Arc<WireCounters>,
}

/// Everything one client needs to join a session and contribute its records.
pub struct SessionParams {
    /// Aggregator base URL (e.g. `http://127.0.0.1:42800`).
    pub agg_endpoint: String,
    /// Client-facing shareholder URLs, ordered by evaluation point x = 1, 2,
    /// and so on. `None` bootstraps them from the enroll response, so a caller
    /// needs only `agg_endpoint` and `token`.
    pub shareholder_endpoints: Option<Vec<String>>,
    pub token: String,
    /// This device's `(features, label)` records. One protocol client
    /// contributes the whole slice under a single commitment per round, so
    /// per-round message size is independent of the batch size and only compute
    /// scales with it. `enroll_at` rejects a ragged batch.
    pub records: Vec<(Vec<f64>, f64)>,
    /// Both the client's own Shamir threshold and the acknowledgements a round
    /// needs to count as delivered. `None` bootstraps it from the enroll
    /// response.
    pub threshold: Option<usize>,
    /// Submit gradient shares for every active node at the current depth
    /// (path hiding), rather than only for the client's true node.
    pub hide_path: bool,
    /// CA certificate pinned on every connection. When set, all endpoints must
    /// be `https://`; `None` leaves them plaintext.
    pub ca_pem: Option<Vec<u8>>,
    /// Which aggregator session to join. `None` joins the aggregator's only
    /// session.
    pub session_id: Option<String>,
}

impl ClientSession {
    /// From round watermark 0, no round polled yet.
    pub async fn enroll(params: SessionParams) -> anyhow::Result<Self> {
        Self::enroll_at(params, 0).await
    }

    /// Enroll and connect the fan-out, resuming from a round watermark.
    ///
    /// `step` polls with `last_seen_round_id` and surfaces only a round whose
    /// id differs, so passing an already-submitted round's watermark means it
    /// is not offered again. Persist the watermark only *after* a successful
    /// `step`: persisting early and then losing the device before the submit
    /// completes resumes past a round this device never contributed to,
    /// silently shrinking that round's contributor count.
    ///
    /// The enroll precedes the fan-out connect because with
    /// `shareholder_endpoints` unset the target list comes from the enroll
    /// response. A resume still enrolls in full, taking a fresh commitment
    /// identity and fresh connections: the watermark restores the session's
    /// position, not its identity.
    pub async fn enroll_at(params: SessionParams, last_seen_round_id: u64) -> anyhow::Result<Self> {
        let SessionParams {
            agg_endpoint,
            shareholder_endpoints,
            token,
            records,
            threshold,
            hide_path,
            ca_pem,
            session_id: requested_session_id,
        } = params;
        anyhow::ensure!(
            !records.is_empty(),
            "a client session requires at least one record"
        );
        // A ragged batch indexes out of bounds inside `compute_stat_shares` and
        // the gradient rounds, panicking across the FRB bridge instead of
        // returning the documented infallible error.
        let width = records[0].0.len();
        if let Some((idx, (features, _))) = records
            .iter()
            .enumerate()
            .find(|(_, (f, _))| f.len() != width)
        {
            anyhow::bail!(
                "record {idx} has {} features, but record 0 has {width} features: \
                 every record in a batch must have the same feature count",
                features.len()
            );
        }
        let n_records = records.len() as u32;

        let wire = Arc::new(WireCounters::default());
        let mut agg =
            connect_aggregator(&agg_endpoint, &token, ca_pem.as_deref(), wire.clone()).await?;

        let req = Request::new(EnrollRequest {
            session_id: requested_session_id.unwrap_or_default(),
        });
        let enrolled = agg.enroll_session(req).await?.into_inner();
        // Authoritative: an empty selector still yields a concrete id here,
        // used for every later poll and submit.
        let session_id = enrolled.session_id;
        anyhow::ensure!(
            !session_id.is_empty(),
            "enrolled session_id must not be empty"
        );

        let shareholder_endpoints =
            shareholder_endpoints.unwrap_or_else(|| enrolled.shareholder_endpoints.clone());
        let threshold = threshold.unwrap_or(enrolled.threshold as usize);
        let n_parties = shareholder_endpoints.len();
        anyhow::ensure!(
            n_parties > 0,
            "at least one shareholder endpoint is required"
        );

        let fanout = Shareholders::connect_best_effort(
            &shareholder_endpoints,
            token.clone(),
            ca_pem.as_deref(),
            wire.clone(),
        )?;
        let client = Client::new_batch(next_client_id(), records, n_parties, threshold, None);

        Ok(Self {
            agg,
            fanout,
            client,
            session_id,
            threshold,
            hide_path,
            n_records,
            last_seen: last_seen_round_id,
            wire,
        })
    }

    /// Snapshot [`WireCounters::tx`] and [`WireCounters::rx`] around a step for
    /// that round's ciphertext cost across all its sockets.
    pub fn wire_counters(&self) -> &WireCounters {
        &self.wire
    }

    /// The round watermark, updated as soon as `step` observes a new round.
    /// Persisting it after a successful `step` lets a caller resume via
    /// [`Self::enroll_at`] without repeating that round.
    pub fn last_seen(&self) -> u64 {
        self.last_seen
    }

    /// The session this client enrolled under. A [`Self::last_seen`] watermark
    /// means nothing outside it, since every session restarts its round ids, so
    /// a resuming caller must persist and pass back both.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Exactly one `PollSession`, plus compute and best-effort submit if it
    /// turned up a new context. Never sleeps and never loops: the caller
    /// decides when to step again.
    pub async fn step(&mut self) -> anyhow::Result<StepOutcome> {
        let req = Request::new(PollSessionRequest {
            last_seen_round_id: self.last_seen,
            session_id: self.session_id.clone(),
        });
        // Snapshot before the poll so a Submitted step's delta spans the whole
        // step: the poll that pulled the context and the submit that followed.
        let tx_before = self.wire.tx();
        let rx_before = self.wire.rx();
        let poll_start = std::time::Instant::now();
        let resp = self.agg.poll_session(req).await?.into_inner();
        let poll_us = poll_start.elapsed().as_micros() as u64;
        let phase = resp.phase();
        if phase == WireSessionPhase::Failed {
            return Ok(StepOutcome::Failed);
        }

        let Some(Body::Ctx(ctx)) = resp.body else {
            let sleep_for = resp
                .next_poll_after
                .and_then(|d| Duration::try_from(d).ok())
                .unwrap_or(DEFAULT_POLL_INTERVAL)
                .max(MIN_POLL_INTERVAL);
            return Ok(StepOutcome::NothingNew {
                next_poll_after: sleep_for,
                session_id: self.session_id.clone(),
                poll_us,
            });
        };
        self.last_seen = ctx.round_id;

        if phase == WireSessionPhase::Completed {
            return Ok(StepOutcome::Completed);
        }

        let round_id = ctx.round_id;
        let session_id = self.session_id.clone();
        let (report, compute_us, submit_us, round_kind) = if ctx.depth == u32::MAX {
            let compute_start = std::time::Instant::now();
            let shares = self.client.compute_stat_shares()?;
            let compute_us = compute_start.elapsed().as_micros() as u64;
            let submit_start = std::time::Instant::now();
            let report = self.fanout.submit_stats(shares, &session_id).await;
            let submit_us = submit_start.elapsed().as_micros() as u64;
            (report, compute_us, submit_us, RoundKind::Stats)
        } else {
            let core_ctx = CoreRoundContext {
                bins: ctx
                    .bin_edges
                    .iter()
                    .cloned()
                    .map(edges_to_bin_config)
                    .collect(),
                model: model_from_proto(
                    ctx.model
                        .clone()
                        .ok_or_else(|| anyhow::anyhow!("gradient round context missing model"))?,
                )?,
                splits: ctx
                    .splits_so_far
                    .iter()
                    .map(|(id, s)| (*id as usize, split_from_proto(*s, *id)))
                    .collect::<BTreeMap<_, _>>(),
                round_id: ctx.round_id,
                depth: ctx.depth as usize,
            };
            let compute_start = std::time::Instant::now();
            let shares =
                self.client
                    .compute_gradient_shares(&core_ctx, &Loss::Logistic, self.hide_path)?;
            let compute_us = compute_start.elapsed().as_micros() as u64;
            let submit_start = std::time::Instant::now();
            let report = self.fanout.submit_gradients(shares, &session_id).await;
            let submit_us = submit_start.elapsed().as_micros() as u64;
            (report, compute_us, submit_us, RoundKind::Gradient)
        };

        let below_threshold = report.accepted < self.threshold;
        if below_threshold {
            tracing::warn!(
                round_id,
                accepted = report.accepted,
                attempted = report.attempted,
                threshold = self.threshold,
                "round contribution delivered to fewer than threshold shareholders"
            );
        }

        let tx_bytes = self.wire.tx() - tx_before;
        let rx_bytes = self.wire.rx() - rx_before;

        Ok(StepOutcome::Submitted {
            round_id,
            session_id: self.session_id.clone(),
            report,
            round_kind,
            poll_us,
            compute_us,
            submit_us,
            tx_bytes,
            rx_bytes,
            below_threshold,
            n_records: self.n_records,
        })
    }
}

/// The platform a device reports in [`register_device`]. `Unspecified` is not
/// representable: the aggregator rejects it, and a device knows what it runs
/// on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientPlatform {
    Android,
    Ios,
}

/// Register this device's FCM token so a round opening can wake it. Opens its
/// own short-lived connection rather than requiring an enrolled session, since
/// registration happens at sign-in and on token refresh, before any session
/// exists to join.
///
/// `ca_pem` pins TLS as [`SessionParams::ca_pem`] does: `Some` requires an
/// `https://` endpoint, `None` leaves it plaintext. A device participating over
/// TLS must register over TLS too.
pub async fn register_device(
    agg_endpoint: &str,
    token: String,
    fcm_token: String,
    platform: ClientPlatform,
    ca_pem: Option<Vec<u8>>,
) -> anyhow::Result<()> {
    // Outside any session, so its wire bytes are not measured.
    let mut agg = connect_aggregator(
        agg_endpoint,
        &token,
        ca_pem.as_deref(),
        Arc::new(WireCounters::default()),
    )
    .await?;
    let platform = match platform {
        ClientPlatform::Android => WireClientPlatform::Android,
        ClientPlatform::Ios => WireClientPlatform::Ios,
    };
    let req = Request::new(RegisterDeviceRequest {
        fcm_token,
        platform: platform.into(),
    });
    agg.register_device(req).await?;
    Ok(())
}

/// One `EnrollSession` RPC, then drop the connection. The app's join action
/// uses it so the notify tick learns of the device's interest immediately; the
/// wake loop's later full enroll refreshes the same row. `ca_pem` pins TLS as
/// [`register_device`] does.
pub async fn enroll_session_only(
    agg_endpoint: &str,
    token: String,
    session_id: String,
    ca_pem: Option<Vec<u8>>,
) -> anyhow::Result<()> {
    let mut agg = connect_aggregator(
        agg_endpoint,
        &token,
        ca_pem.as_deref(),
        Arc::new(WireCounters::default()),
    )
    .await?;
    agg.enroll_session(Request::new(EnrollRequest { session_id }))
        .await?;
    Ok(())
}

/// Which phase a hosted session (see [`SessionSummary`]) is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    /// Waiting for enough clients to submit the Gaussian binning stats round.
    StatsPending,
    /// Growing trees: the per-tree, per-depth gradient rounds.
    Training,
    Completed,
    Failed,
}

/// Enough for a device to choose whether to join, before enrolling.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: String,
    pub phase: SessionPhase,
    /// The feature width records submitted to this session must match.
    pub n_features: u32,
    /// Which dataset a device selects its records by; empty for a dataset-less
    /// session. `n_features` cross-checks the width, it does not select.
    pub dataset_id: String,
    /// Epoch milliseconds on the aggregator's own wall clock: orders sessions
    /// from one aggregator, not across hosts.
    pub created_at_ms: u64,
}

/// Every session the aggregator is hosting, so a device can choose one before
/// enrolling. Opens its own short-lived connection, like [`register_device`],
/// and pins TLS the same way.
pub async fn list_sessions(
    agg_endpoint: &str,
    token: &str,
    ca_pem: Option<&[u8]>,
) -> anyhow::Result<Vec<SessionSummary>> {
    let mut agg = connect_aggregator(
        agg_endpoint,
        token,
        ca_pem,
        Arc::new(WireCounters::default()),
    )
    .await?;
    let req = Request::new(ListSessionsRequest {});
    let list = agg.list_sessions(req).await?.into_inner();
    list.sessions
        .into_iter()
        .map(|s| {
            let phase = match s.phase() {
                WireSessionPhase::StatsPending => SessionPhase::StatsPending,
                WireSessionPhase::Training => SessionPhase::Training,
                WireSessionPhase::Completed => SessionPhase::Completed,
                WireSessionPhase::Failed => SessionPhase::Failed,
                WireSessionPhase::Unspecified => anyhow::bail!(
                    "aggregator reported session {} with no phase",
                    s.session_id
                ),
            };
            // A hostile aggregator could send a pre-epoch or far-future
            // timestamp, so clamp and saturate rather than wrap.
            let created_at_ms = s
                .created_at
                .map(|t| match u64::try_from(t.seconds) {
                    Ok(secs) => secs
                        .saturating_mul(1000)
                        .saturating_add(t.nanos.max(0) as u64 / 1_000_000),
                    Err(_) => 0,
                })
                .unwrap_or(0);
            Ok(SessionSummary {
                session_id: s.session_id,
                phase,
                n_features: s.n_features,
                dataset_id: s.dataset_id,
                created_at_ms,
            })
        })
        .collect()
}

/// Session-total wire byte tallies for one `run_collecting` run.
/// `total_*` is the whole session (`wire_counters()` start→end: handshakes,
/// polls, and submits). `submit_*` is the sum of per-round `Submitted`
/// tx/rx deltas only. `total − submit` is thus handshake + idle-poll traffic.
pub struct WireRun {
    pub total_tx: u64,
    pub total_rx: u64,
    pub submit_tx: u64,
    pub submit_rx: u64,
    pub n_rounds: u64,
}

/// Drive a session to completion, returning its wire byte totals (`WireRun`).
pub async fn run_collecting(
    params: SessionParams,
    mut on_progress: impl FnMut(&StepOutcome),
) -> anyhow::Result<WireRun> {
    let mut session = ClientSession::enroll(params).await?;
    let mut consecutive_failures: u32 = 0;
    let mut submit_tx: u64 = 0;
    let mut submit_rx: u64 = 0;
    let mut n_rounds: u64 = 0;
    loop {
        let outcome = match session.step().await {
            Ok(outcome) => {
                consecutive_failures = 0;
                outcome
            }
            // Step errors are treated as transient, a timed-out poll on a
            // flaky radio, so one network blip does not abandon the whole
            // participation.
            Err(e) => {
                consecutive_failures += 1;
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    return Err(e.context(format!(
                        "aborting session after {consecutive_failures} consecutive step failures"
                    )));
                }
                tracing::warn!(
                    error = %e,
                    consecutive_failures,
                    "step failed; retrying after a transient error"
                );
                tokio::time::sleep(DEFAULT_POLL_INTERVAL).await;
                continue;
            }
        };
        on_progress(&outcome);
        match outcome {
            StepOutcome::Completed => {
                return Ok(WireRun {
                    total_tx: session.wire_counters().tx(),
                    total_rx: session.wire_counters().rx(),
                    submit_tx,
                    submit_rx,
                    n_rounds,
                });
            }
            StepOutcome::Failed => anyhow::bail!("aggregator session failed"),
            StepOutcome::NothingNew { next_poll_after, .. } => {
                tokio::time::sleep(next_poll_after).await;
            }
            StepOutcome::Submitted { tx_bytes, rx_bytes, .. } => {
                submit_tx += tx_bytes;
                submit_rx += rx_bytes;
                n_rounds += 1;
            }
        }
    }
}

/// Enroll, then poll, compute and submit until the aggregator reports the
/// session complete. `on_progress` observes every step outcome, `NothingNew`
/// polls included, so a caller need not reimplement the loop to watch it. Pass
/// `|_| {}` to run unobserved.
pub async fn run_to_completion(
    params: SessionParams,
    on_progress: impl FnMut(&StepOutcome),
) -> anyhow::Result<()> {
    run_collecting(params, on_progress).await.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_stats_round() {
        let d = decode_round_id(1).expect("stats round has a summary");
        assert_eq!(d.kind, RoundKind::Stats);
        assert_eq!(d.tree_idx, None);
        assert_eq!(d.depth, None);
    }

    #[test]
    fn decode_gradient_round() {
        // gradient_round_id(tree = 1, depth = 0) = ((1 + 1) << 32) | 0.
        let d = decode_round_id((1 + 1) << 32).expect("gradient round has a summary");
        assert_eq!(d.kind, RoundKind::Gradient);
        assert_eq!(d.tree_idx, Some(1));
        assert_eq!(d.depth, Some(0));

        // A deeper round on the third tree: gradient_round_id(tree = 2, depth = 3).
        let d = decode_round_id(((2 + 1) << 32) | 3).expect("gradient round has a summary");
        assert_eq!(d.kind, RoundKind::Gradient);
        assert_eq!(d.tree_idx, Some(2));
        assert_eq!(d.depth, Some(3));
    }

    #[test]
    fn decode_completed_sentinel_is_none() {
        // completed = ((n_trees + 1) << 32) | u32::MAX carries no contribution.
        let completed = ((1u64 + 1) << 32) | u32::MAX as u64;
        assert!(decode_round_id(completed).is_none());
    }
}
