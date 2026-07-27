//! Aggregator orchestrator: the round loop (`run_session`) and the server
//! bootstrap (`serve`). The client-facing gRPC service lives in `service`,
//! device and enrollment state with the notify tick in `notify`, the
//! internal-plane round-lifecycle RPCs in `internal`, the gather-side share
//! sources in `gather`, wire round ids and published contexts in `context`.
//!
//! # Round loop
//!
//! The loop owns a `pbr_core::Aggregator` over [`SourceSlot`] share sources
//! and drives each round in strict sequence:
//!
//! 1. `OpenRound` on every shareholder, which resets their pools; fewer than
//!    `threshold` acceptances fails the round before anything is published;
//! 2. publish a `RoundContext` for `PollSession` to serve;
//! 3. wait the submission window, polling `ListCommitments` counts: close
//!    early once every shareholder holds `target_clients` commitments, or at
//!    the deadline once the best threshold-combination overlap reaches
//!    `min_clients` (extended up to `GIVE_UP_WINDOWS` windows);
//! 4. `CloseRound`, which freezes the pools. The gather reads only
//!    shareholders that both opened and confirmed: one that never opened
//!    still Acks a stale close, and an unconfirmed pool may still be
//!    accepting submissions;
//! 5. gather: one prefetched `GrpcShareSource` snapshot per confirmed
//!    shareholder, then `define_bins`/`compute_splits`, all inside
//!    `spawn_blocking` because they drive `block_on`;
//! 6. advance to the next context, or finish the tree.
//!
//! The next `OpenRound` waits for the current gather: it resets the pools,
//! destroying shares the gather still needs.

mod context;
mod gather;
mod internal;
mod manager;
mod notify;
mod service;
mod spec;

pub use notify::DeviceRow;
pub use spec::{DatasetTable, SessionSpec};

pub(crate) use internal::internal_endpoint;

use crate::admin;
use crate::agg_config::AggregatorConfig;
use crate::auth;
use crate::eval::Evaluator;
use crate::fcm::FcmSender;
use context::{
    STATS_DEPTH_SENTINEL, STATS_ROUND_ID, completed_ctx, gradient_ctx, gradient_round_id, stats_ctx,
};
use gather::{SourceSlot, snapshot_slot};
use internal::{
    await_submissions, close_round_all, close_round_and_end_session_best_effort,
    connect_all_internal, end_session_all, open_round_all,
};
use manager::SessionManager;
use pbr_proto::convert::bin_config_to_edges;
use pbr_proto::v1::admin_service_server::{AdminService, AdminServiceServer};
use pbr_proto::v1::aggregator_service_server::AggregatorServiceServer;
use pbr_proto::v1::{
    Ack, CreateSessionRequest, DeleteSessionRequest, ListSessionsRequest, SessionList,
    SessionPhase, SessionSummary, SharePhase,
};
use pbr_core::{Aggregator, BinMethod};
use service::{AggregatorSvc, SessionState, SharedSession, fail_session, publish, session_list};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// Split-search bounds, matching `pbr-core`'s benchmark binary so distributed
/// runs stay comparable with its single-process results.
const MIN_GAIN: f64 = 0.0;
const MIN_CHILD_WEIGHT: f64 = 1.0;

/// Cap on live (non-terminal) sessions this process hosts at once. Each holds
/// its own tasks and internal-plane connections, and nothing removes a session
/// automatically (`DeleteSession` is an operator action), so unbounded
/// `CreateSession` calls would accumulate them. A safety rail, not a deployment
/// knob; public so tests assert against the real limit.
pub const MAX_LIVE_SESSIONS: usize = 16;

async fn run_session(
    cfg: AggregatorConfig,
    spec: SessionSpec,
    session: SharedSession,
    session_id: String,
    eval: Option<Arc<Evaluator>>,
) -> anyhow::Result<()> {
    let window = spec.submission_window();
    let endpoints = cfg.internal_shareholder_endpoints.clone();

    let mut internals = connect_all_internal(&endpoints, cfg.threshold).await?;

    let opened = open_round_all(
        &mut internals,
        STATS_ROUND_ID,
        STATS_DEPTH_SENTINEL,
        SharePhase::Stats,
        &session_id,
        cfg.threshold,
    )
    .await?;
    publish(
        &session,
        SessionPhase::StatsPending,
        stats_ctx(window, &session_id),
    );
    await_submissions(
        &mut internals,
        SharePhase::Stats,
        0,
        &session_id,
        cfg.threshold,
        spec.target_clients,
        spec.min_clients,
        window,
    )
    .await?;
    let closed = close_round_all(
        &mut internals,
        STATS_ROUND_ID,
        STATS_DEPTH_SENTINEL,
        SharePhase::Stats,
        &session_id,
        cfg.threshold,
        &opened,
    )
    .await?;

    // Only confirmed-closed shareholders are read: an unconfirmed pool may
    // still be accepting submissions. The snapshots drive `block_on`, which
    // forces spawn_blocking.
    let (mut agg, bins) = {
        let endpoints = endpoints.clone();
        let session_id_for_gather = session_id.clone();
        let (n_bins, threshold, min_clients, lr, lambda) = (
            spec.n_bins,
            cfg.threshold,
            spec.min_clients,
            spec.learning_rate,
            spec.lambda,
        );
        tokio::task::spawn_blocking(move || {
            let handle = tokio::runtime::Handle::current();
            let sources: Vec<SourceSlot> = endpoints
                .iter()
                .zip(&closed)
                .map(|(ep, &frozen)| {
                    if frozen {
                        snapshot_slot(ep, SharePhase::Stats, 0, &session_id_for_gather, &handle)
                    } else {
                        SourceSlot::Dead
                    }
                })
                .collect();
            let mut agg = Aggregator::new(
                sources,
                n_bins,
                threshold,
                min_clients,
                lr,
                lambda,
                BinMethod::Gaussian,
            )?;
            let bins = agg.define_bins()?;
            anyhow::Ok((agg, bins))
        })
        .await??
    };

    {
        let mut s = session.lock().unwrap();
        s.bin_edges = bins.iter().map(bin_config_to_edges).collect();
        s.n_features = bins.len() as u32;
        s.phase = SessionPhase::Training;
    }
    tracing::info!(
        n_features = bins.len(),
        "stats phase complete, bins defined"
    );

    // The [eval.datasets] split for this dataset, if its width matches the
    // bins just defined. Nothing cross-checks a split against the dataset
    // table, so a mis-sized CSV reaches here and is dropped rather than
    // scoring against another session's features.
    let session_split = eval.as_ref().and_then(|e| {
        let split = e.splits.get(&spec.dataset_id)?;
        if split.width() != bins.len() {
            tracing::error!(
                dataset_id = %spec.dataset_id,
                split_width = split.width(),
                n_features = bins.len(),
                "eval split width does not match this session's features; not scoring it"
            );
            return None;
        }
        Some(split)
    });

    for _ in 0..spec.n_trees {
        loop {
            // (tree, depth) come from the aggregator's own state; the loop
            // never tracks them separately.
            let pctx = agg.round_context();
            let depth = pctx.depth;
            if depth >= spec.max_depth {
                break;
            }
            let round_id = gradient_round_id(pctx.round_id, depth);

            let opened = open_round_all(
                &mut internals,
                round_id,
                depth as u32,
                SharePhase::Gradient,
                &session_id,
                cfg.threshold,
            )
            .await?;
            publish(
                &session,
                SessionPhase::Training,
                gradient_ctx(&pctx, round_id, window, &session_id),
            );
            await_submissions(
                &mut internals,
                SharePhase::Gradient,
                depth as u32,
                &session_id,
                cfg.threshold,
                spec.target_clients,
                spec.min_clients,
                window,
            )
            .await?;
            let closed = close_round_all(
                &mut internals,
                round_id,
                depth as u32,
                SharePhase::Gradient,
                &session_id,
                cfg.threshold,
                &opened,
            )
            .await?;

            let (next_agg, more_splits) = {
                let endpoints = endpoints.clone();
                let session_id_for_gather = session_id.clone();
                tokio::task::spawn_blocking(move || {
                    let handle = tokio::runtime::Handle::current();
                    for ((slot, ep), &frozen) in agg
                        .shareholders_mut()
                        .iter_mut()
                        .zip(&endpoints)
                        .zip(&closed)
                    {
                        *slot = if frozen {
                            snapshot_slot(
                                ep,
                                SharePhase::Gradient,
                                depth,
                                &session_id_for_gather,
                                &handle,
                            )
                        } else {
                            SourceSlot::Dead
                        };
                    }
                    let more = agg.compute_splits(MIN_GAIN, MIN_CHILD_WEIGHT)?;
                    anyhow::Ok((agg, more))
                })
                .await??
            };
            agg = next_agg;
            if !more_splits {
                break;
            }
        }
        agg.finish_round();
        tracing::info!(trees = agg.model().trees.len(), "tree finished");
        if let (Some(eval), Some(split)) = (&eval, session_split) {
            let doc = crate::eval::score_newest_tree(agg.model(), split, &session_id);
            let sink = eval.sink.clone();
            // Never awaited: a hung Firestore call must cost one metric row,
            // not a stalled round for every client.
            tokio::spawn(async move {
                match sink.write(&doc).await {
                    Ok(()) => tracing::info!(
                        tree_idx = doc.tree_idx,
                        auc = doc.metrics.auc,
                        "wrote per-tree metric"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        tree_idx = doc.tree_idx,
                        "per-tree metric write failed; continuing"
                    ),
                }
            });
        }
    }

    publish(
        &session,
        SessionPhase::Completed,
        completed_ctx(&agg, spec.n_trees, &session_id),
    );
    end_session_all(&mut internals, &session_id).await;
    tracing::info!("session completed");
    Ok(())
}

/// Upsert one session's row. A failed write is logged and ignored: whether
/// history survives a restart must not decide whether training proceeds.
async fn checkpoint(manager: &SessionManager, session: &SharedSession) {
    if let Err(e) = manager.persist(session).await {
        tracing::error!(error = %e, "failed to persist session state");
    }
}

/// Delete one session's row. Logged-not-fatal like `checkpoint`: a delete that
/// never reaches disk leaves one stale row, which resurfaces on the next
/// restart carrying the last phase written and goes away on a repeat delete.
async fn checkpoint_removal(manager: &SessionManager, session_id: &str) {
    if let Err(e) = manager.persist_removal(session_id).await {
        tracing::error!(error = %e, %session_id, "failed to delete session row");
    }
}

/// Upsert one session's row, then delete it again if the session has since left
/// the serving map. Every upsert site must go through this: it is what stops a
/// `DeleteSession` racing an upsert from leaving a resurrected row that reloads
/// on the next restart.
///
/// Both interleavings converge to no row. If the handler's map-remove precedes
/// this check, `get` returns `None` and this delete runs after the upsert; if
/// the check precedes it, the handler's own row-delete runs after this upsert.
/// The map read serializes on the map mutex, the `DELETE` on the
/// single-connection store, and both deletes are idempotent.
async fn checkpoint_and_compensate(manager: &SessionManager, session: &SharedSession) {
    checkpoint(manager, session).await;
    let session_id = session.lock().unwrap().session_id.clone();
    if manager.get(&session_id).is_none() {
        checkpoint_removal(manager, &session_id).await;
    }
}

/// Mint a session id and register it before its round loop exists, so it
/// resolves (`resolve`, `ListSessions`) the instant this returns; then spawn
/// `run_session` and attach the loop's abort handle. A concurrent
/// `DeleteSession` in that window aborts the loop instead of re-registering
/// the session.
///
/// `cap` bounds live sessions: `SessionManager::try_insert_new` counts and
/// registers under one lock acquisition, so concurrent callers cannot each
/// observe room and together overshoot. Returns `None`, spawning nothing, at
/// the cap.
fn spawn_session(
    cfg: AggregatorConfig,
    spec: SessionSpec,
    manager: &SessionManager,
    eval: Option<Arc<Evaluator>>,
    cap: usize,
) -> Option<(String, SharedSession)> {
    let session_id = Uuid::new_v4().to_string();
    let session: SharedSession = Arc::new(Mutex::new(SessionState {
        phase: SessionPhase::StatsPending,
        ctx: None,
        bin_edges: Vec::new(),
        // From the dataset table so a device's pre-join width check works while
        // the session is still StatsPending; the stats gather overwrites it
        // with the measured width. A dataset-less session has no entry: 0.
        n_features: cfg.datasets.n_features(&spec.dataset_id).unwrap_or(0),
        session_id: session_id.clone(),
        spec: spec.clone(),
        created_at: SystemTime::now(),
    }));
    if !manager.try_insert_new(cap, session_id.clone(), session.clone()) {
        return None;
    }

    let loop_session = session.clone();
    let loop_session_id = session_id.clone();
    let close_endpoints = cfg.internal_shareholder_endpoints.clone();
    let loop_manager = manager.clone();
    let round_loop = tokio::spawn(async move {
        // The creation upsert runs here, at the loop's first await, because the
        // sync `create_session` chain cannot await. It lands before the terminal
        // write below, so the store records create then outcome in order even
        // if the two race a restart.
        checkpoint_and_compensate(&loop_manager, &loop_session).await;
        if let Err(e) = run_session(cfg, spec, loop_session.clone(), loop_session_id, eval)
            .await
        {
            tracing::error!(error = %e, "aggregator session failed");
            fail_session(&loop_session);
            // Best-effort: stop shareholders accepting submissions for a failed
            // session and free its pools now rather than at the idle sweep.
            close_round_and_end_session_best_effort(&close_endpoints, &loop_session).await;
        }
        // Reached on both outcomes, with the phase already published. A
        // `DeleteSession` can land during run_session's final EndSession
        // broadcast, since it does not abort a terminal loop, so this terminal
        // upsert needs the same re-check as the creation site.
        checkpoint_and_compensate(&loop_manager, &loop_session).await;
    });

    let abort_handle = round_loop.abort_handle();
    if !manager.attach_round_loop(&session_id, abort_handle.clone()) {
        // Deleted between registration and this attach: stop the just-spawned
        // loop rather than resurrecting the session. DeleteSession already
        // checkpointed the removal.
        abort_handle.abort();
    }
    // A panic resolves the JoinHandle to a panicked JoinError without running
    // run_session's Err path, leaving the session published in its last
    // non-terminal phase forever. This monitor owns the handle (the manager
    // keeps only an abort handle) and fails the session instead.
    let monitor_session = session.clone();
    let monitor_manager = manager.clone();
    tokio::spawn(async move {
        if let Err(e) = round_loop.await
            && e.is_panic()
        {
            tracing::error!("aggregator round loop panicked; marking session failed");
            fail_session(&monitor_session);
            checkpoint_and_compensate(&monitor_manager, &monitor_session).await;
        }
    });

    Some((session_id, session))
}

/// The shared dependencies of a session spawn, so the admin `CreateSession`
/// RPC and the in-process `AggregatorHandle::create_session` both create
/// sessions through `spawn_session` under one `MAX_LIVE_SESSIONS` cap. They
/// differ only in how they validate the spec (`create` vs
/// `create_dataset_unchecked`) and how they surface errors.
#[derive(Clone)]
struct SessionFactory {
    cfg: AggregatorConfig,
    manager: SessionManager,
    eval: Option<Arc<Evaluator>>,
}

/// Why a `SessionFactory::create*` call produced no session.
enum CreateError {
    Invalid(anyhow::Error),
    CapReached { live: usize },
}

impl SessionFactory {
    /// The admin `CreateSession` semantics: reject a spec whose `dataset_id`
    /// is not in this cluster's dataset table, then reject one that fails the
    /// hyperparameter/round-close bounds, then spawn under the cap.
    fn create(&self, spec: SessionSpec) -> Result<(String, SharedSession), CreateError> {
        spec.validate(&self.cfg.datasets)
            .map_err(CreateError::Invalid)?;
        self.spawn(spec)
    }

    /// The sole way to create a dataset-less session (`dataset_id` empty),
    /// which `create` refuses because the empty id is never in the dataset
    /// table: skips the table check and runs `SessionSpec::validate_bounds`
    /// alone. A non-empty `dataset_id` delegates to `create`, so this is
    /// "unchecked" only for the one shape `create` cannot produce.
    fn create_dataset_unchecked(
        &self,
        spec: SessionSpec,
    ) -> Result<(String, SharedSession), CreateError> {
        if !spec.dataset_id.is_empty() {
            return self.create(spec);
        }
        spec.validate_bounds().map_err(CreateError::Invalid)?;
        self.spawn(spec)
    }

    fn spawn(&self, spec: SessionSpec) -> Result<(String, SharedSession), CreateError> {
        spawn_session(
            self.cfg.clone(),
            spec,
            &self.manager,
            self.eval.clone(),
            MAX_LIVE_SESSIONS,
        )
        .ok_or_else(|| CreateError::CapReached {
            live: self.manager.live_count(),
        })
    }
}

fn session_summary(session_id: String, session: &SharedSession) -> SessionSummary {
    let s = session.lock().unwrap();
    SessionSummary {
        session_id,
        phase: s.phase as i32,
        n_features: s.n_features,
        dataset_id: s.spec.dataset_id.clone(),
        created_at: Some(prost_types::Timestamp::from(s.created_at)),
    }
}

/// The admin plane, authenticated by `admin::interceptor`'s static bearer
/// token and never by the device identity provider.
struct AdminSvc {
    factory: SessionFactory,
}

#[tonic::async_trait]
impl AdminService for AdminSvc {
    async fn create_session(
        &self,
        req: Request<CreateSessionRequest>,
    ) -> Result<Response<SessionSummary>, Status> {
        let msg = req.into_inner();
        let spec = SessionSpec {
            dataset_id: msg.dataset_id,
            title: msg.title,
            n_trees: msg.n_trees as usize,
            max_depth: msg.max_depth as usize,
            n_bins: msg.n_bins as usize,
            learning_rate: msg.learning_rate,
            lambda: msg.lambda,
            min_clients: msg.min_clients as usize,
            target_clients: msg.target_clients as usize,
            submission_window_ms: msg.submission_window_ms,
        };
        let (session_id, session) = self.factory.create(spec).map_err(|e| match e {
            CreateError::Invalid(err) => Status::invalid_argument(err.to_string()),
            CreateError::CapReached { live } => Status::resource_exhausted(format!(
                "cluster already hosts {live} live sessions (limit {MAX_LIVE_SESSIONS})"
            )),
        })?;
        Ok(Response::new(session_summary(session_id, &session)))
    }

    async fn list_sessions(
        &self,
        _req: Request<ListSessionsRequest>,
    ) -> Result<Response<SessionList>, Status> {
        Ok(Response::new(session_list(&self.factory.manager)))
    }

    async fn delete_session(
        &self,
        req: Request<DeleteSessionRequest>,
    ) -> Result<Response<Ack>, Status> {
        let session_id = req.into_inner().session_id;
        let Some(removed) = self.factory.manager.remove(&session_id) else {
            return Err(Status::not_found(format!("no session {session_id}")));
        };
        // Only a live session's loop is aborted: a terminal loop may still be
        // running its own final EndSession broadcast (Completed is published
        // before end_session_all), which an abort would cut short with no
        // replacement cleanup. The panic monitor sees a cancelled join, not a
        // panicked one, and does nothing.
        if removed.was_live
            && let Some(handle) = &removed.round_loop
        {
            handle.abort();
        }
        // The row goes before the broadcast: once this RPC returns a crash must
        // not resurrect the session from a stale row, and the broadcast below
        // can block on connect timeouts against dead shareholders.
        checkpoint_removal(&self.factory.manager, &session_id).await;
        if removed.was_live {
            close_round_and_end_session_best_effort(
                &self.factory.cfg.internal_shareholder_endpoints,
                &removed.state,
            )
            .await;
        }
        tracing::info!(%session_id, was_live = removed.was_live, "session deleted by admin");
        Ok(Response::new(Ack {}))
    }
}

pub struct RunningAggregator {
    pub addr: SocketAddr,
    pub handle: AggregatorHandle,
}

pub struct AggregatorHandle {
    server_tx: oneshot::Sender<()>,
    manager: SessionManager,
    /// Kept only for its Drop: aborts the Firebase key-refresh task when this
    /// handle drops, so the refresh loop cannot outlive the aggregator.
    #[allow(dead_code)]
    refresh: crate::config::RefreshHandle,
    /// Aborts the notify tick on shutdown so it cannot outlive the aggregator.
    notify_loop: Option<tokio::task::AbortHandle>,
    factory: SessionFactory,
}

impl AggregatorHandle {
    pub fn shutdown(self) {
        let _ = self.server_tx.send(());
        if let Some(h) = &self.notify_loop {
            h.abort();
        }
        self.manager.abort_all();
    }

    /// The devices table as stored; read-only introspection.
    pub async fn registered_devices(&self) -> anyhow::Result<Vec<DeviceRow>> {
        notify::all_devices(self.manager.store()).await
    }

    /// The enrollments table as stored, `(session_id, uid)` pairs.
    pub async fn enrollments(&self) -> anyhow::Result<Vec<(String, String)>> {
        notify::all_enrollments(self.manager.store()).await
    }

    /// Create a session in-process, through the same `spawn_session` path and
    /// cap as the admin `CreateSession` RPC. A non-empty `dataset_id` gets the
    /// RPC's full validation; only an empty one skips the dataset table, which
    /// is how a dataset-less session gets created.
    pub fn create_session(&self, spec: SessionSpec) -> anyhow::Result<SessionSummary> {
        let (session_id, session) =
            self.factory.create_dataset_unchecked(spec).map_err(|e| match e {
                CreateError::Invalid(err) => err,
                CreateError::CapReached { live } => anyhow::anyhow!(
                    "cluster already hosts {live} live sessions (limit {MAX_LIVE_SESSIONS})"
                ),
            })?;
        Ok(session_summary(session_id, &session))
    }
}

/// Bind the client-facing listener (port 0 supported) and start the JWT-gated
/// `AggregatorService`, plus `AdminService` when `cfg.admin_token` is set. The
/// process boots hosting no session; a configured `[eval]` is built here and
/// any failure aborts startup.
pub async fn serve(cfg: AggregatorConfig) -> anyhow::Result<RunningAggregator> {
    let eval = match &cfg.eval {
        Some(ecfg) => Some(Arc::new(Evaluator::from_config(ecfg).await?)),
        None => None,
    };
    serve_with_eval(cfg, eval).await
}

/// `serve` with the evaluator supplied directly, `cfg.eval` ignored: the seam
/// integration tests use to inject a fake `MetricSink` with no live Firestore.
pub async fn serve_with_eval(
    cfg: AggregatorConfig,
    eval: Option<Arc<Evaluator>>,
) -> anyhow::Result<RunningAggregator> {
    cfg.validate()?;
    let crate::config::RefreshingVerifier { verifier, refresh } =
        cfg.auth.build_and_refresh_verifier().await?;
    // The SQLite store at `state_path` reloads whatever history a previous
    // process wrote, in-flight sessions demoted to Failed; a missing file boots
    // an empty store, and a failed open aborts startup.
    let manager = SessionManager::load(&cfg.state_path).await?;
    let listener = tokio::net::TcpListener::bind(cfg.listen).await?;
    let addr = listener.local_addr()?;
    let (server_tx, server_rx) = oneshot::channel();

    // Absent `[fcm]` disables notify entirely. A configured but unbuildable
    // sender (no ADC on this machine, say) is logged and disables notify for
    // this run rather than failing the aggregator: push is a nicety, not on the
    // critical path to a trained model.
    let (fcm_sender, floor_secs): (Option<FcmSender>, i64) = match &cfg.fcm {
        Some(fcm_cfg) => {
            let floor_secs = (fcm_cfg.interval_minutes * 60) as i64;
            let sender = match FcmSender::from_config(
                fcm_cfg.project_id.clone(),
                fcm_cfg.service_account_path.clone(),
                Duration::from_secs(fcm_cfg.interval_minutes * 60),
            )
            .await
            {
                Ok(sender) => Some(sender),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "FCM configured but sender init failed; round-open push disabled for this run"
                    );
                    None
                }
            };
            (sender, floor_secs)
        }
        None => (None, 0),
    };

    // Built before the router so both the `AdminSvc` and the returned handle
    // can hold a clone.
    let factory = SessionFactory {
        cfg: cfg.clone(),
        manager: manager.clone(),
        eval,
    };

    // One process-wide notify loop, re-planning every NOTIFY_TICK_PERIOD, so
    // push stays off every round loop's critical path.
    let notify_loop = fcm_sender.map(|fcm| {
        let manager = manager.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(notify::NOTIFY_TICK_PERIOD).await;
                let sessions: Vec<(String, SessionPhase)> = manager
                    .summaries()
                    .into_iter()
                    .map(|s| (s.session_id, s.phase))
                    .collect();
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                match notify::run_notify_tick(manager.store(), &sessions, &fcm, now, floor_secs)
                    .await
                {
                    Ok(0) => {}
                    Ok(n) => tracing::info!(notified = n, "round_open notify tick"),
                    Err(e) => tracing::warn!(error = %e, "notify tick failed"),
                }
            }
        })
        .abort_handle()
    });

    let svc = AggregatorServiceServer::with_interceptor(
        AggregatorSvc {
            manager: manager.clone(),
            threshold: cfg.threshold as u32,
            n_parties: cfg.internal_shareholder_endpoints.len() as u32,
            client_shareholder_endpoints: cfg.client_shareholder_endpoints.clone(),
        },
        auth::interceptor(verifier),
    );
    // `[tls]` wraps the client-facing listener; absent it the listener is
    // plaintext. The aggregator has no internal listener of its own.
    let mut client_builder = tonic::transport::Server::builder();
    if let Some(tls) = &cfg.tls {
        client_builder = client_builder.tls_config(tls.server_tls_config()?)?;
    }
    let mut client_router = client_builder.add_service(svc);

    // Without a configured token `AdminService` is never added to the router,
    // so its RPCs are unreachable rather than merely rejecting. The token value
    // is never logged.
    match &cfg.admin_token {
        Some(token) => {
            tracing::info!(
                "admin plane enabled: CreateSession, ListSessions, DeleteSession are reachable"
            );
            let admin_svc = AdminServiceServer::with_interceptor(
                AdminSvc {
                    factory: factory.clone(),
                },
                admin::interceptor(token.clone()),
            );
            client_router = client_router.add_service(admin_svc);
        }
        None => tracing::info!("admin plane disabled: no admin_token configured"),
    }

    // The accept loop's Result would otherwise vanish with the dropped
    // JoinHandle, leaving the process up with a dead listener that surfaces
    // only minutes later as a misleading "did not reach min_clients".
    tokio::spawn(async move {
        let result = client_router
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = server_rx.await;
                },
            )
            .await;
        if let Err(e) = result {
            tracing::error!(error = %e, "aggregator client-facing gRPC server exited with error");
        }
    });

    Ok(RunningAggregator {
        addr,
        handle: AggregatorHandle {
            server_tx,
            manager,
            refresh,
            notify_loop,
            factory,
        },
    })
}
