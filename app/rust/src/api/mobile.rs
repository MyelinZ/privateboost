//! The FRB surface: the flat, self-contained entry points codegen generates
//! Dart bindings from. Running a device's records through a session to
//! completion (`participate`, `participate_batch`), advancing a resumed session
//! by one round (`step_session`), listing hosted sessions (`list_sessions`),
//! and FCM registration (`register_device`).
//!
//! Every entry point here is infallible across the bridge: none panics or
//! returns `Err`, and any failure is folded into the result's `last_error`, so
//! Dart never handles a bridge-level exception for a network or protocol
//! fault.

use crate::frb_generated::StreamSink;
use pbr_client::driver::{
    decode_round_id, run_to_completion, ClientPlatform as CoreClientPlatform, ClientSession,
    RoundKind as CoreRoundKind, SessionParams, SessionPhase as CoreSessionPhase,
    SessionSummary as CoreSessionSummary, StepOutcome,
};
use std::sync::Mutex;

/// FRB generates a Dart enum whose `.name` yields `stats` or `gradient`, which
/// the analysis pipeline keys on.
pub enum RoundKind {
    Stats,
    Gradient,
}

/// How a round's submission landed; `.name` yields `submitted` or
/// `belowThreshold`. Reaching fewer than the Shamir threshold of shareholders
/// is recorded, not fatal.
pub enum RoundOutcome {
    Submitted,
    BelowThreshold,
}

/// One submitted round's mobile-side measurements, one per round, each
/// becoming a `paperSimRoundMetrics` document. `tree_idx` and `depth` are
/// decoded from `round_id` in Rust; `last_error` is a per-round note.
pub struct RoundSummary {
    pub round_id: u64,
    /// Keys these metrics on the same session as the aggregator's per-tree
    /// quality metrics.
    pub session_id: String,
    /// How many records the one client folded into its commitment.
    pub n_records: u32,
    pub round_kind: RoundKind,
    pub tree_idx: Option<u32>,
    pub depth: Option<u32>,
    pub poll_us: u64,
    pub compute_us: u64,
    pub submit_us: u64,
    pub tx_bytes: u64,
    pub rx_bytes: u64,
    pub n_peers_attempted: u32,
    pub n_peers_accepted: u32,
    pub outcome: RoundOutcome,
    pub last_error: Option<String>,
}

/// The stream `participate` emits round summaries into. FRB turns a
/// `StreamSink` parameter into a Dart `Stream<T>` return, and a function cannot
/// return both a value and a stream, so the sink is registered here instead of
/// threaded through `participate`. Registered once per isolate; newest wins.
static ROUND_SUMMARY_SINK: Mutex<Option<StreamSink<RoundSummary>>> = Mutex::new(None);

/// Subscribe before calling `participate`: each submitted round then arrives
/// as one [`RoundSummary`]. The sink auto-closes when Dart drops the stream.
pub fn create_round_summary_stream(sink: StreamSink<RoundSummary>) {
    if let Ok(mut slot) = ROUND_SUMMARY_SINK.lock() {
        *slot = Some(sink);
    }
}

/// Telemetry must never break participation, so a poisoned lock, an
/// unregistered sink or a closed stream is swallowed here.
fn emit_round_summary(summary: RoundSummary) {
    if let Ok(slot) = ROUND_SUMMARY_SINK.lock() {
        if let Some(sink) = slot.as_ref() {
            let _ = sink.add(summary);
        }
    }
}

/// A `step_session` wake whose poll found no new context. Leaner than
/// [`RoundSummary`]: nothing was computed or submitted, so there is no round
/// id, no wire detail, and no timing beyond the poll.
pub struct IdlePoll {
    /// Keys the check-in document on the same session as a submitted round.
    pub session_id: String,
    /// Wall time spent waiting on the `PollSession` RPC, in microseconds.
    pub poll_us: u64,
}

/// `Some` only for a `Submitted` step whose round id decodes to a real round;
/// anything else, including the completed sentinel, has nothing to emit.
fn round_summary_from(outcome: &StepOutcome) -> Option<RoundSummary> {
    let StepOutcome::Submitted {
        round_id,
        session_id,
        report,
        round_kind,
        poll_us,
        compute_us,
        submit_us,
        tx_bytes,
        rx_bytes,
        below_threshold,
        n_records,
    } = outcome
    else {
        return None;
    };
    let decoded = decode_round_id(*round_id)?;
    Some(RoundSummary {
        round_id: *round_id,
        session_id: session_id.clone(),
        n_records: *n_records,
        round_kind: match round_kind {
            CoreRoundKind::Stats => RoundKind::Stats,
            CoreRoundKind::Gradient => RoundKind::Gradient,
        },
        tree_idx: decoded.tree_idx,
        depth: decoded.depth,
        poll_us: *poll_us,
        compute_us: *compute_us,
        submit_us: *submit_us,
        tx_bytes: *tx_bytes,
        rx_bytes: *rx_bytes,
        n_peers_attempted: report.attempted as u32,
        n_peers_accepted: report.accepted as u32,
        outcome: if *below_threshold {
            RoundOutcome::BelowThreshold
        } else {
            RoundOutcome::Submitted
        },
        last_error: None,
    })
}

/// `Some` only for `NothingNew`: a `Submitted` step has a [`RoundSummary`]
/// instead, and the terminal outcomes have nothing to report.
fn idle_poll_from(outcome: &StepOutcome) -> Option<IdlePoll> {
    let StepOutcome::NothingNew {
        session_id,
        poll_us,
        ..
    } = outcome
    else {
        return None;
    };
    Some(IdlePoll {
        session_id: session_id.clone(),
        poll_us: *poll_us,
    })
}

/// FRB init hook, called once during `RustLib.init()`: installs Android
/// logging and FRB's panic handler, so a Rust panic surfaces as a Dart
/// `PanicException` rather than aborting the process.
#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Routed to logcat first, so it wins the global logger slot.
    #[cfg(target_os = "android")]
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug)
            .with_tag("pboost_rust"),
    );
    flutter_rust_bridge::setup_default_user_utils();
}

/// Plain fields: no proto types cross the bridge.
pub struct ParticipationResult {
    pub completed: bool,
    pub rounds_submitted: u32,
    pub last_error: Option<String>,
    /// Microseconds of share computation, summed over every `Submitted` step.
    pub compute_us: u64,
    /// Microseconds of shareholder fan-out, summed the same way.
    pub submit_us: u64,
}

/// Drive this device through one full training session: enroll, then poll,
/// compute and submit until it completes. Shareholder endpoints and the Shamir
/// threshold are learned from the enroll response.
///
/// `session_id` picks the session, `None` the aggregator's only one. `ca_pem`
/// pins TLS for an `https://` endpoint; `None` connects in plaintext, the
/// loopback default. Dart chooses by scheme and supplies the bundled CA.
pub async fn participate(
    agg_endpoint: String,
    id_token: String,
    session_id: Option<String>,
    feature_row: Vec<f64>,
    label: f64,
    hide_path: bool,
    ca_pem: Option<Vec<u8>>,
) -> ParticipationResult {
    let mut rounds_submitted: u32 = 0;
    let mut compute_us_total: u64 = 0;
    let mut submit_us_total: u64 = 0;
    let result = run_to_completion(
        SessionParams {
            agg_endpoint,
            shareholder_endpoints: None,
            token: id_token,
            records: vec![(feature_row, label)],
            threshold: None,
            hide_path,
            // Pins the CA for the aggregator channel and the fan-out alike.
            ca_pem,
            session_id,
        },
        |outcome| {
            if let StepOutcome::Submitted {
                compute_us,
                submit_us,
                ..
            } = outcome
            {
                rounds_submitted += 1;
                compute_us_total += compute_us;
                submit_us_total += submit_us;
            }
            if let Some(summary) = round_summary_from(outcome) {
                emit_round_summary(summary);
            }
        },
    )
    .await;

    match result {
        Ok(()) => ParticipationResult {
            completed: true,
            rounds_submitted,
            last_error: None,
            compute_us: compute_us_total,
            submit_us: submit_us_total,
        },
        Err(e) => ParticipationResult {
            completed: false,
            rounds_submitted,
            last_error: Some(e.to_string()),
            compute_us: compute_us_total,
            submit_us: submit_us_total,
        },
    }
}

/// One protocol client contributes the device's whole record slice under one
/// commitment per round. Per-round detail arrives on the summary stream.
pub struct BatchParticipationResult {
    /// The batch size: how many records this device submitted.
    pub records: u32,
    /// 1 if the client drove its session to `Completed`, else 0.
    pub completed: u32,
    /// Rounds this client submitted.
    pub rounds_total: u32,
    /// The client's error, if it failed rather than completed.
    pub last_error: Option<String>,
}

/// Drive this device's whole record slice through one session as one protocol
/// client: `records` is folded into a single commitment per round, shared once
/// rather than once per record, so per-round message size is independent of
/// batch size and only compute scales with it. Submitted rounds go to the same
/// summary stream `participate` feeds.
///
/// `session_id` and `ca_pem` behave as in [`participate`].
pub async fn participate_batch(
    agg_endpoint: String,
    id_token: String,
    session_id: Option<String>,
    records: Vec<(Vec<f64>, f64)>,
    hide_path: bool,
    ca_pem: Option<Vec<u8>>,
) -> BatchParticipationResult {
    let records_total = records.len() as u32;
    let mut rounds_submitted: u32 = 0;
    let result = run_to_completion(
        SessionParams {
            agg_endpoint,
            shareholder_endpoints: None,
            token: id_token,
            records,
            threshold: None,
            hide_path,
            ca_pem,
            session_id,
        },
        |outcome| {
            if matches!(outcome, StepOutcome::Submitted { .. }) {
                rounds_submitted += 1;
            }
            if let Some(summary) = round_summary_from(outcome) {
                emit_round_summary(summary);
            }
        },
    )
    .await;

    match result {
        Ok(()) => BatchParticipationResult {
            records: records_total,
            completed: 1,
            rounds_total: rounds_submitted,
            last_error: None,
        },
        Err(e) => BatchParticipationResult {
            records: records_total,
            completed: 0,
            rounds_total: rounds_submitted,
            last_error: Some(e.to_string()),
        },
    }
}

/// Plain fields: no proto types cross the bridge.
pub struct RegisterResult {
    pub ok: bool,
    pub last_error: Option<String>,
}

/// Its own enum, so no pbr-client type crosses the bridge.
pub enum ClientPlatform {
    Android,
    Ios,
}

/// Register this device's FCM token, bound to the identity `id_token` proves.
/// Called on sign-in and on every token refresh. A device participating over
/// TLS must register over TLS, so Dart threads the same `ca_pem` through
/// both.
pub async fn register_device(
    agg_endpoint: String,
    id_token: String,
    fcm_token: String,
    platform: ClientPlatform,
    ca_pem: Option<Vec<u8>>,
) -> RegisterResult {
    let platform = match platform {
        ClientPlatform::Android => CoreClientPlatform::Android,
        ClientPlatform::Ios => CoreClientPlatform::Ios,
    };
    match pbr_client::driver::register_device(&agg_endpoint, id_token, fcm_token, platform, ca_pem)
        .await
    {
        Ok(()) => RegisterResult {
            ok: true,
            last_error: None,
        },
        Err(e) => RegisterResult {
            ok: false,
            last_error: Some(e.to_string()),
        },
    }
}

/// Plain data across the bridge.
pub struct EnrollResult {
    pub ok: bool,
    pub last_error: Option<String>,
}

/// Records this device's interest, since the notify tick only wakes enrolled
/// devices. Best-effort from the join action: the wake loop's own enroll
/// refreshes the same row, so a failure here delays only the first push.
pub async fn enroll_session(
    agg_endpoint: String,
    id_token: String,
    session_id: String,
    ca_pem: Option<Vec<u8>>,
) -> EnrollResult {
    match pbr_client::driver::enroll_session_only(&agg_endpoint, id_token, session_id, ca_pem)
        .await
    {
        Ok(()) => EnrollResult {
            ok: true,
            last_error: None,
        },
        Err(e) => EnrollResult {
            ok: false,
            last_error: Some(e.to_string()),
        },
    }
}

/// Which phase a hosted session is in, as reported by `list_sessions`.
pub enum SessionPhase {
    StatsPending,
    Training,
    Completed,
    Failed,
}

/// One session the app can choose to join, as reported by `list_sessions`.
pub struct SessionSummary {
    pub session_id: String,
    pub phase: SessionPhase,
    pub n_features: u32,
    /// Which dataset a device selects its records by; empty for a
    /// dataset-less session. `n_features` cross-checks this id's expected
    /// width and is never a substitute selector.
    pub dataset_id: String,
}

fn session_summary_from(core: CoreSessionSummary) -> SessionSummary {
    SessionSummary {
        session_id: core.session_id,
        phase: match core.phase {
            CoreSessionPhase::StatsPending => SessionPhase::StatsPending,
            CoreSessionPhase::Training => SessionPhase::Training,
            CoreSessionPhase::Completed => SessionPhase::Completed,
            CoreSessionPhase::Failed => SessionPhase::Failed,
        },
        n_features: core.n_features,
        dataset_id: core.dataset_id,
    }
}

/// Plain fields: no proto types cross the bridge.
pub struct SessionListResult {
    pub sessions: Vec<SessionSummary>,
    pub last_error: Option<String>,
}

/// Every session the aggregator is hosting, so the app can offer a choice
/// before joining one. `ca_pem` behaves as in [`participate`]; on failure the
/// session list comes back empty.
pub async fn list_sessions(
    agg_endpoint: String,
    id_token: String,
    ca_pem: Option<Vec<u8>>,
) -> SessionListResult {
    match pbr_client::driver::list_sessions(&agg_endpoint, &id_token, ca_pem.as_deref()).await {
        Ok(sessions) => SessionListResult {
            sessions: sessions.into_iter().map(session_summary_from).collect(),
            last_error: None,
        },
        Err(e) => SessionListResult {
            sessions: Vec::new(),
            last_error: Some(e.to_string()),
        },
    }
}

/// What one `step_session` wake observed.
pub enum RoundStepOutcome {
    /// A round was polled, computed and submitted; `summary` has the detail.
    Submitted,
    /// No new round was open yet; try again after the server's poll hint.
    NothingNew,
    /// The aggregator reports the session complete; no further steps.
    Completed,
    /// The aggregator reports the session failed; no further steps.
    Failed,
    /// The step could not be completed; see `last_error`.
    Error,
}

/// Outcome of one `step_session` call: exactly one wake's worth of progress
/// on a resumable session, crossing the FRB bridge as plain fields.
pub struct RoundStepResult {
    pub outcome: RoundStepOutcome,
    /// The round id this step observed; 0 for `NothingNew`, `Error`, or a
    /// `Completed`/`Failed` outcome with no new round context.
    pub round_id: u64,
    /// The session this wake enrolled under; `None` only if `enroll_at` itself
    /// failed. Persist it alongside `last_seen_round_id` and pass both back: a
    /// watermark means nothing outside its own session. Without the pair, a
    /// device resuming after an aggregator restart can find the new session's
    /// round id equal to its stale watermark and park on `NothingNew`
    /// forever.
    pub session_id: Option<String>,
    /// The watermark to persist for the next wake. It advances only on
    /// `Submitted`; every other outcome, `Error` included, returns the input
    /// unchanged. `ClientSession` advances its own watermark as soon as it
    /// polls a context, before compute and submit, so a step can still fail
    /// after that point: reporting the advanced value would let a caller
    /// persist a round this device never finished, and skip it next wake.
    pub last_seen_round_id: u64,
    pub summary: Option<RoundSummary>,
    /// Only for `NothingNew`: the idle poll's session id and wall time, for a
    /// check-in document. `summary` is its `Submitted` counterpart.
    pub idle: Option<IdlePoll>,
    pub last_error: Option<String>,
}

/// Advance a resumed session by exactly one round: enroll at
/// `last_seen_round_id`, take one step, and report what happened plus the
/// watermark to persist. The seam a background wake drives: one wake, one
/// step, no polling loop inside.
///
/// `session_id` and `ca_pem` behave as in [`participate`]; an enroll or step
/// failure arrives as `outcome: Error` with `last_error` set.
pub async fn step_session(
    agg_endpoint: String,
    id_token: String,
    session_id: Option<String>,
    records: Vec<(Vec<f64>, f64)>,
    hide_path: bool,
    ca_pem: Option<Vec<u8>>,
    last_seen_round_id: u64,
) -> RoundStepResult {
    let mut session = match ClientSession::enroll_at(
        SessionParams {
            agg_endpoint,
            shareholder_endpoints: None,
            token: id_token,
            records,
            threshold: None,
            hide_path,
            ca_pem,
            session_id,
        },
        last_seen_round_id,
    )
    .await
    {
        Ok(session) => session,
        Err(e) => {
            return RoundStepResult {
                outcome: RoundStepOutcome::Error,
                round_id: 0,
                session_id: None,
                last_seen_round_id,
                summary: None,
                idle: None,
                last_error: Some(e.to_string()),
            };
        }
    };

    let step_result = session.step().await;
    // `session.last_seen()` may already be advanced past `last_seen_round_id`
    // here: `ClientSession::step` sets it as soon as it polls a new round's
    // context, before computing or submitting for it. Only report that
    // advance when the step actually finished (`Ok`); see
    // `watermark_to_report`.
    let reported_last_seen =
        watermark_to_report(&step_result, last_seen_round_id, session.last_seen());
    let session_id = Some(session.session_id().to_string());

    match step_result {
        Ok(outcome) => {
            let round_id = match &outcome {
                StepOutcome::Submitted { round_id, .. } => *round_id,
                _ => 0,
            };
            let step_outcome = match &outcome {
                StepOutcome::Submitted { .. } => RoundStepOutcome::Submitted,
                StepOutcome::NothingNew { .. } => RoundStepOutcome::NothingNew,
                StepOutcome::Completed => RoundStepOutcome::Completed,
                StepOutcome::Failed => RoundStepOutcome::Failed,
            };
            let summary = round_summary_from(&outcome);
            let idle = idle_poll_from(&outcome);
            RoundStepResult {
                outcome: step_outcome,
                round_id,
                session_id,
                last_seen_round_id: reported_last_seen,
                summary,
                idle,
                last_error: None,
            }
        }
        Err(e) => RoundStepResult {
            outcome: RoundStepOutcome::Error,
            round_id: 0,
            session_id,
            last_seen_round_id: reported_last_seen,
            summary: None,
            idle: None,
            last_error: Some(e.to_string()),
        },
    }
}

/// On a successful step, whatever its outcome, the round `last_seen()` just
/// advanced to; `NothingNew` and `Completed` leave that equal to
/// `prior_last_seen` anyway. On `Err`, `prior_last_seen` unchanged: a round the
/// device did not finish contributing to must stay eligible next wake, even
/// though `ClientSession::step` already bumped its internal watermark when it
/// polled that context.
fn watermark_to_report(
    step_result: &anyhow::Result<StepOutcome>,
    prior_last_seen: u64,
    polled_last_seen: u64,
) -> u64 {
    if step_result.is_ok() {
        polled_last_seen
    } else {
        prior_last_seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pbr_client::submit::DeliveryReport;

    fn submitted(round_id: u64, kind: CoreRoundKind, accepted: usize, below: bool) -> StepOutcome {
        StepOutcome::Submitted {
            round_id,
            session_id: "test-session".to_string(),
            report: DeliveryReport {
                attempted: 3,
                accepted,
            },
            round_kind: kind,
            poll_us: 5,
            compute_us: 250,
            submit_us: 12,
            tx_bytes: 900,
            rx_bytes: 1500,
            below_threshold: below,
            n_records: 1,
        }
    }

    #[test]
    fn maps_a_submitted_gradient_round() {
        // gradient_round_id(tree = 1, depth = 0).
        let s = round_summary_from(&submitted((1 + 1) << 32, CoreRoundKind::Gradient, 3, false))
            .expect("a submitted round maps to a summary");
        assert!(matches!(s.round_kind, RoundKind::Gradient));
        assert_eq!(s.tree_idx, Some(1));
        assert_eq!(s.depth, Some(0));
        assert_eq!(s.compute_us, 250);
        assert_eq!(s.tx_bytes, 900);
        assert_eq!(s.n_peers_attempted, 3);
        assert_eq!(s.n_peers_accepted, 3);
        assert!(matches!(s.outcome, RoundOutcome::Submitted));
        assert!(s.last_error.is_none());
    }

    #[test]
    fn carries_the_batch_size() {
        // A submitted step's own n_records (the one client's batch size) must
        // survive the StepOutcome -> RoundSummary mapping, or per-round
        // metrics cannot be attributed to the right batch size.
        let s = round_summary_from(&submitted(1, CoreRoundKind::Stats, 3, false))
            .expect("a submitted round maps to a summary");
        assert_eq!(s.n_records, 1);
    }

    #[test]
    fn maps_the_stats_round_with_no_tree_or_depth() {
        let s = round_summary_from(&submitted(1, CoreRoundKind::Stats, 3, false)).unwrap();
        assert!(matches!(s.round_kind, RoundKind::Stats));
        assert_eq!(s.tree_idx, None);
        assert_eq!(s.depth, None);
    }

    #[test]
    fn marks_a_below_threshold_round() {
        let s = round_summary_from(&submitted(1, CoreRoundKind::Stats, 1, true)).unwrap();
        assert_eq!(s.n_peers_accepted, 1);
        assert!(matches!(s.outcome, RoundOutcome::BelowThreshold));
    }

    #[test]
    fn no_summary_for_non_submitted_steps() {
        assert!(round_summary_from(&StepOutcome::Completed).is_none());
        assert!(round_summary_from(&StepOutcome::Failed).is_none());
    }

    fn nothing_new(session_id: &str, poll_us: u64) -> StepOutcome {
        StepOutcome::NothingNew {
            next_poll_after: std::time::Duration::from_secs(1),
            session_id: session_id.to_string(),
            poll_us,
        }
    }

    #[test]
    fn idle_poll_from_a_nothing_new_step() {
        let idle = idle_poll_from(&nothing_new("sess-idle", 42))
            .expect("a NothingNew step carries idle-poll fields");
        assert_eq!(idle.session_id, "sess-idle");
        assert_eq!(idle.poll_us, 42);
    }

    #[test]
    fn no_idle_poll_for_submitted_or_terminal_steps() {
        assert!(idle_poll_from(&submitted(1, CoreRoundKind::Stats, 3, false)).is_none());
        assert!(idle_poll_from(&StepOutcome::Completed).is_none());
        assert!(idle_poll_from(&StepOutcome::Failed).is_none());
    }

    #[test]
    fn watermark_advances_on_a_successful_step() {
        assert_eq!(watermark_to_report(&Ok(StepOutcome::Completed), 5, 42), 42);
    }

    #[test]
    fn watermark_holds_the_prior_value_when_the_step_fails() {
        // `ClientSession::step` can advance its internal watermark upon
        // polling a new round's context, then still return `Err` if
        // computing or converting that round's context fails afterward
        // (the submit itself is infallible best-effort). A caller must not
        // persist that advanced value for a round this device never
        // finished contributing to, or the next wake would skip it
        // entirely, reproducing the exact hazard `enroll_at`'s own doc
        // warns about, through a different door.
        let failed_step: anyhow::Result<StepOutcome> =
            Err(anyhow::anyhow!("share computation failed"));
        assert_eq!(watermark_to_report(&failed_step, 5, 42), 5);
    }

    #[tokio::test]
    async fn participate_fails_gracefully_against_a_dead_endpoint() {
        // No cluster listening on this port: `participate` must still
        // return (never panic), with completed == false and an error
        // message rather than propagating a bridge-level exception.
        let result = participate(
            "http://127.0.0.1:1".to_string(),
            "fake-token".to_string(),
            None,
            vec![1.0, 2.0],
            0.0,
            true,
            None,
        )
        .await;
        assert!(!result.completed);
        assert_eq!(result.rounds_submitted, 0);
        assert!(result.last_error.is_some());
    }

    #[tokio::test]
    async fn register_device_fails_gracefully_against_a_dead_endpoint() {
        // No cluster listening on this port: `register_device` must still
        // return (never panic), with ok == false and an error message
        // rather than propagating a bridge-level exception.
        let result = register_device(
            "http://127.0.0.1:1".to_string(),
            "fake-token".to_string(),
            "fake-fcm-token".to_string(),
            ClientPlatform::Android,
            None,
        )
        .await;
        assert!(!result.ok);
        assert!(result.last_error.is_some());
    }

    #[tokio::test]
    async fn enroll_session_fails_gracefully_against_a_dead_endpoint() {
        let r = enroll_session(
            "http://127.0.0.1:1".into(),
            "token".into(),
            "sess".into(),
            None,
        )
        .await;
        assert!(!r.ok);
        assert!(r.last_error.is_some());
    }

    mod cluster {
        use super::*;
        use pbr_client::jwt::mint;
        use pbr_server::agg_config::AggregatorConfig;
        use pbr_server::aggregator::{
            serve as serve_aggregator, DatasetTable, RunningAggregator, SessionSpec,
        };
        use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
        use pbr_server::shareholder::{serve as serve_shareholder, RunningShareholder};

        const ISS: &str = "https://test-issuer.local";
        const AUD: &str = "pbr";
        const KID: &str = "test-1";
        const PRIV: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../crates/pbr-server/tests/fixtures/test_key.pem"
        ));

        fn auth_cfg() -> AuthConfig {
            AuthConfig {
                issuer: ISS.into(),
                audience: AUD.into(),
                static_keys: vec![StaticKey {
                    kid: KID.into(),
                    public_key_pem_path: concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../crates/pbr-server/tests/fixtures/test_key.pub.pem"
                    )
                    .into(),
                }],
                google_jwks_url: None,
            }
        }

        fn shareholder_cfg(x: u64) -> ShareholderConfig {
            ShareholderConfig {
                x_coord: x,
                min_clients: 1,
                listen: "127.0.0.1:0".parse().unwrap(),
                internal_listen: "127.0.0.1:0".parse().unwrap(),
                auth: auth_cfg(),
                tls: None,
            }
        }

        struct Cluster {
            agg_url: String,
            agg_handle: pbr_server::aggregator::AggregatorHandle,
            sh_handles: Vec<pbr_server::shareholder::ShutdownHandle>,
        }

        impl Cluster {
            fn shutdown(self) {
                self.agg_handle.shutdown();
                for h in self.sh_handles {
                    h.shutdown();
                }
            }
        }

        /// Three shareholders and an aggregator on loopback, min_clients = 1,
        /// one depth-1 tree: the smallest session a bridge test can complete.
        async fn start_cluster() -> Cluster {
            start_cluster_with(1, 1).await
        }

        /// The cluster above, but with the aggregator's round-close gates
        /// (`min_clients`, `target_clients`) chosen by the caller. A batch of
        /// `target_clients` concurrent clients closes every round on its own
        /// once all of them submit.
        async fn start_cluster_with(min_clients: usize, target_clients: usize) -> Cluster {
            let mut client_eps = Vec::new();
            let mut internal_eps = Vec::new();
            let mut sh_handles = Vec::new();
            for x in 1..=3u64 {
                let RunningShareholder {
                    client_addr: addr,
                    internal_addr: internal,
                    handle: h,
                } = serve_shareholder(shareholder_cfg(x)).await.unwrap();
                client_eps.push(format!("http://{addr}"));
                internal_eps.push(format!("http://{internal}"));
                sh_handles.push(h);
            }

            let RunningAggregator {
                addr: agg_addr,
                handle: agg_handle,
            } = serve_aggregator(AggregatorConfig {
                listen: "127.0.0.1:0".parse().unwrap(),
                internal_shareholder_endpoints: internal_eps,
                client_shareholder_endpoints: client_eps,
                threshold: 2,
                auth: auth_cfg(),
                eval: None,
                fcm: None,
                tls: None,
                datasets: DatasetTable::default(),
                admin_token: None,
                state_path: ":memory:".into(),
            })
            .await
            .unwrap();

            // The aggregator boots hosting nothing; create the dataset-less
            // session these bridge tests drive with an empty selector.
            agg_handle
                .create_session(SessionSpec {
                    dataset_id: String::new(),
                    title: "bridge test".into(),
                    n_trees: 1,
                    max_depth: 1,
                    n_bins: 8,
                    learning_rate: 0.3,
                    lambda: 1.0,
                    min_clients,
                    target_clients,
                    submission_window_ms: 2_000,
                })
                .expect("the session under test must be created");

            Cluster {
                agg_url: format!("http://{agg_addr}"),
                agg_handle,
                sh_handles,
            }
        }

        /// `participate` against a real 1-client, 3-shareholder cluster: this
        /// proves the bridge logic (not just `run_to_completion` itself,
        /// already covered by `pbr-client`'s own tests) actually drives a
        /// session to completion and reports it correctly.
        #[tokio::test]
        async fn participate_completes_against_local_cluster() {
            let cluster = start_cluster().await;
            let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

            let result = participate(
                cluster.agg_url.clone(),
                token,
                None,
                vec![2.0, 1.0],
                1.0,
                true,
                None,
            )
            .await;

            assert!(
                result.completed,
                "participate should complete: {:?}",
                result.last_error
            );
            assert!(result.last_error.is_none());
            assert!(
                result.rounds_submitted >= 1,
                "expected at least one submitted round (stats + gradient)"
            );

            cluster.shutdown();
        }

        /// A device's whole slice goes to the aggregator as one client, so the
        /// session sees a single contributor no matter how many records the
        /// device holds.
        #[tokio::test(flavor = "multi_thread")]
        async fn participate_batch_contributes_as_a_single_client() {
            let cluster = start_cluster().await;
            let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

            let records: Vec<(Vec<f64>, f64)> = (0..5)
                .map(|i| (vec![i as f64, (4 - i) as f64], (i % 2) as f64))
                .collect();

            let result =
                participate_batch(cluster.agg_url.clone(), token, None, records, true, None).await;

            assert_eq!(result.records, 5, "the batch size is 5 records");
            assert_eq!(
                result.completed, 1,
                "one client drives the whole batch to completion: {:?}",
                result.last_error
            );
            assert!(
                result.last_error.is_none(),
                "the client should not error: {:?}",
                result.last_error
            );
            assert!(
                result.rounds_total >= 1,
                "the client submits at least the stats round, got {}",
                result.rounds_total
            );

            cluster.shutdown();
        }

        /// The app can enumerate sessions before choosing one to join.
        #[tokio::test(flavor = "multi_thread")]
        async fn list_sessions_returns_the_hosted_session() {
            let cluster = start_cluster().await;
            let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

            let result = list_sessions(cluster.agg_url.clone(), token, None).await;

            assert!(result.last_error.is_none(), "{:?}", result.last_error);
            assert_eq!(result.sessions.len(), 1, "the created session is hosted");
            let s = &result.sessions[0];
            assert!(!s.session_id.is_empty(), "a session always has an id");
            assert!(
                matches!(s.phase, SessionPhase::StatsPending),
                "a fresh session with no contributions yet is StatsPending"
            );
            assert_eq!(
                s.dataset_id, "",
                "this dataset-less session names no dataset; a device must never guess one for it"
            );

            cluster.shutdown();
        }

        /// One wake advances the session by at most one round and reports the
        /// watermark to persist. Each wake re-enrolls from scratch (mirroring a
        /// killed-and-restarted background isolate), so this loops calling
        /// Steps with the previous watermark until it observes a submitted
        /// round. The round loop opens round 1 in a task separate from the
        /// listener, so an early wake legitimately sees `NothingNew`; only a
        /// hang is a failure.
        #[tokio::test(flavor = "multi_thread")]
        async fn step_session_advances_one_round_and_reports_the_watermark() {
            let cluster = start_cluster().await;
            let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

            let mut last_seen = 0u64;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            let result = loop {
                assert!(
                    std::time::Instant::now() < deadline,
                    "no round submitted in time"
                );
                let result = step_session(
                    cluster.agg_url.clone(),
                    token.clone(),
                    None,
                    vec![(vec![2.0, 1.0], 1.0)],
                    true,
                    None,
                    last_seen,
                )
                .await;
                assert!(result.last_error.is_none(), "{:?}", result.last_error);
                // The session id must be present on every outcome, not only
                // `Submitted`: a caller has no other reliable way to learn
                // which session a `NothingNew` wake resumed, and needs the
                // id to persist alongside the watermark for the next wake.
                assert!(
                    result.session_id.as_deref().is_some_and(|s| !s.is_empty()),
                    "session id must be present even on a non-Submitted outcome"
                );
                last_seen = result.last_seen_round_id;
                if matches!(result.outcome, RoundStepOutcome::Submitted) {
                    break result;
                }
                assert!(
                    matches!(result.outcome, RoundStepOutcome::NothingNew),
                    "expected NothingNew while waiting for round 1 to open"
                );
                let expected_session_id = result.session_id.clone().unwrap();
                assert!(
                    result
                        .idle
                        .as_ref()
                        .is_some_and(|idle| idle.session_id == expected_session_id),
                    "a NothingNew step reports idle-poll fields keyed on this session"
                );
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            };

            assert_eq!(
                result.last_seen_round_id, result.round_id,
                "a submitted step's watermark is the round it just observed"
            );
            assert!(
                result.summary.is_some(),
                "a submitted step reports a round summary"
            );
            assert!(
                result.idle.is_none(),
                "a submitted step reports no idle-poll fields"
            );

            cluster.shutdown();
        }

        /// `register_device` against a real aggregator: proves the bridge
        /// fn (not just the RPC handler, covered by `pbr-server`'s own
        /// tests) actually reaches the aggregator and reports success.
        #[tokio::test]
        async fn register_device_completes_against_local_cluster() {
            let cluster = start_cluster().await;
            let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

            let result = register_device(
                cluster.agg_url.clone(),
                token,
                "fcm-token-abc".to_string(),
                ClientPlatform::Android,
                None,
            )
            .await;

            assert!(
                result.ok,
                "register_device should succeed: {:?}",
                result.last_error
            );
            assert!(result.last_error.is_none());

            cluster.shutdown();
        }
    }
}
