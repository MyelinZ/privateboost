//! Internal-plane control of the shareholder round lifecycle: connecting to
//! every shareholder, the OpenRound/CloseRound broadcasts, submission-window
//! polling, and the overlap math that decides when a round may close.
//!
//! # Early-close race handling
//!
//! Early close needs the intersection across every shareholder to reach
//! `target_clients`. The deadline path instead uses the best
//! threshold-combination overlap, the rule
//! `pbr_core::Aggregator::select_shareholders` applies at gather time, so the
//! close counts exactly the clients the gather can reconstruct and a dead
//! shareholder cannot hold the session below `min_clients`. A client's
//! per-round contribution to one shareholder is a single atomic
//! `SubmitGradientBatch`, so a counted commitment stands for that client's
//! whole per-node contribution there; the grace delay before CloseRound only
//! gives batches still in flight to other shareholders a moment to land.

use super::context::STATS_DEPTH_SENTINEL;
use super::service::SharedSession;
use pbr_proto::v1::shareholder_internal_client::ShareholderInternalClient;
use pbr_proto::v1::{
    CloseRoundRequest, EndSessionRequest, ListCommitmentsRequest, OpenRoundRequest, SharePhase,
};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};
use tonic::transport::{Channel, Endpoint};

/// A round keeps waiting past its deadline until `min_clients` is met; after
/// this many submission windows in total the session fails.
const GIVE_UP_WINDOWS: u32 = 10;

/// Bounds one `connect_internal` attempt, so a black-holed link cannot hang it.
const INTERNAL_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Per-RPC deadline on open, close and list_commitments. A shareholder that
/// accepts the connection but never answers (SIGSTOP, a poisoned lock, a
/// black-holed link) fails within this budget instead of hanging the round
/// loop, which is what lets `await_submissions`'s give-up deadline fire. Kept
/// well under the smallest give-up budget (window x GIVE_UP_WINDOWS).
const INTERNAL_RPC_TIMEOUT: Duration = Duration::from_secs(10);

pub(super) type Internals = Vec<ShareholderInternalClient<Channel>>;

/// An internal-plane `Endpoint` carrying both budgets. `grpc_share_source`
/// builds its gather channel from this same helper, so the post-close
/// snapshot and get_sums reads carry identical deadlines.
pub(crate) fn internal_endpoint(endpoint: &str) -> anyhow::Result<Endpoint> {
    Ok(Endpoint::from_shared(endpoint.to_string())?
        .connect_timeout(INTERNAL_CONNECT_TIMEOUT)
        .timeout(INTERNAL_RPC_TIMEOUT))
}

async fn connect_internal(endpoint: &str) -> anyhow::Result<ShareholderInternalClient<Channel>> {
    let ep = internal_endpoint(endpoint)?;
    // Small retry budget so a cluster whose processes start in arbitrary
    // order still comes up before the caller gives up on this endpoint.
    let mut last_err = None;
    for _ in 0..20 {
        match ep.connect().await {
            Ok(c) => return Ok(ShareholderInternalClient::new(c)),
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        }
    }
    anyhow::bail!(
        "failed to connect to shareholder internal endpoint {endpoint}: {}",
        last_err.expect("at least one attempt")
    )
}

/// Connect to every shareholder's internal endpoint, tolerating a startup
/// outage down to `threshold` reachable ones. An unreachable shareholder gets a
/// lazy placeholder client whose RPCs fail until it comes up, so being absent
/// at startup behaves exactly like dying a second after it: the round calls and
/// the gather already treat a shareholder they cannot reach as dead and route
/// around it. The placeholder also preserves the positional alignment between
/// `internals`, the opened/closed masks and `endpoints` that the round loop
/// relies on. Fewer than `threshold` reachable fails the session outright,
/// since no reconstructable subset could ever be frozen.
pub(super) async fn connect_all_internal(
    endpoints: &[String],
    threshold: usize,
) -> anyhow::Result<Internals> {
    let mut internals = Internals::with_capacity(endpoints.len());
    let mut reachable = 0usize;
    for ep in endpoints {
        match connect_internal(ep).await {
            Ok(client) => {
                reachable += 1;
                internals.push(client);
            }
            Err(e) => {
                tracing::warn!(
                    endpoint = %ep,
                    error = %e,
                    "shareholder unreachable at startup; using a placeholder whose RPCs fail \
                     until it comes up, and routing around it down to threshold"
                );
                internals.push(ShareholderInternalClient::new(
                    internal_endpoint(ep)?.connect_lazy(),
                ));
            }
        }
    }
    anyhow::ensure!(
        reachable >= threshold,
        "only {reachable} of {} shareholders reachable at startup (threshold {threshold}); \
         no reconstructable subset can be frozen",
        endpoints.len(),
    );
    Ok(internals)
}

/// Best-effort CloseRound and EndSession broadcast after a session has failed.
/// `run_session` returning early would otherwise leave the last round open on
/// the shareholders, still accepting submissions for a session that no longer
/// exists, and leave its pools allocated until `SESSION_IDLE_TTL` sweeps them.
///
/// The close is derived from the published context, whose round_id, depth and
/// session_id are what the loop opened with, so it either matches the daemon's
/// open round or hits the idempotent "already closed" path. `EndSession` goes
/// out even if no round was ever published, since freeing a pool needs only the
/// session id. Failures are logged, not retried: a down shareholder has nothing
/// to close, and the next session's OpenRound resets any pool this misses.
pub(super) async fn close_round_and_end_session_best_effort(
    endpoints: &[String],
    session: &SharedSession,
) {
    let (session_id, ctx) = {
        let s = session.lock().unwrap();
        (s.session_id.clone(), s.ctx.clone())
    };
    let close_req = ctx.map(|ctx| {
        let phase = if ctx.depth == STATS_DEPTH_SENTINEL {
            SharePhase::Stats
        } else {
            SharePhase::Gradient
        };
        CloseRoundRequest {
            round_id: ctx.round_id,
            depth: ctx.depth,
            phase: phase as i32,
            session_id: ctx.session_id,
        }
    });
    for ep in endpoints {
        let channel = match internal_endpoint(ep) {
            Ok(e) => e.connect_lazy(),
            Err(e) => {
                tracing::warn!(
                    endpoint = %ep,
                    error = %e,
                    "best-effort session-failure cleanup: bad endpoint"
                );
                continue;
            }
        };
        let mut client = ShareholderInternalClient::new(channel);
        if let Some(req) = &close_req
            && let Err(e) = client.close_round(req.clone()).await
        {
            tracing::warn!(
                endpoint = %ep,
                error = %e,
                "best-effort CloseRound on session failure did not land"
            );
        }
        if let Err(e) = client
            .end_session(EndSessionRequest {
                session_id: session_id.clone(),
            })
            .await
        {
            tracing::warn!(
                endpoint = %ep,
                error = %e,
                session_id,
                "best-effort EndSession on session failure did not land"
            );
        }
    }
}

/// Open the round on every shareholder and report which accepted. Fewer than
/// `threshold` opens means no reconstructable subset can ever be confirmed
/// frozen, so this fails before the context is published or a submission
/// window is entered, rather than burning the give-up budget on a doomed round.
pub(super) async fn open_round_all(
    internals: &mut Internals,
    round_id: u64,
    depth: u32,
    phase: SharePhase,
    session_id: &str,
    threshold: usize,
) -> anyhow::Result<Vec<bool>> {
    let mut opened = vec![false; internals.len()];
    for (i, client) in internals.iter_mut().enumerate() {
        match client
            .open_round(OpenRoundRequest {
                round_id,
                depth,
                phase: phase as i32,
                session_id: session_id.to_string(),
            })
            .await
        {
            Ok(_) => opened[i] = true,
            Err(e) => tracing::warn!(round_id, depth, error = %e, "open_round failed"),
        }
    }
    let ok = opened.iter().filter(|&&o| o).count();
    anyhow::ensure!(
        ok >= threshold,
        "open_round succeeded on only {ok} of {} shareholders (threshold {threshold}); \
         failing the round before it can enter a submission window it could never leave",
        opened.len()
    );
    Ok(opened)
}

/// Close the round on every shareholder and report which are safe to gather
/// from: one must have opened this round and had its close confirmed. A
/// shareholder that never opened still Acks a close for it (the daemon's
/// idempotent "already closed" case), so confirmation alone overstates which
/// pools are frozen.
///
/// The gather treats anything outside this mask as dead for the round: its pool
/// may still be accepting submissions, and gathering from an unfrozen pool
/// reintroduces the List-to-Sums race. Fewer than `threshold` qualifying
/// shareholders fails the round.
pub(super) async fn close_round_all(
    internals: &mut Internals,
    round_id: u64,
    depth: u32,
    phase: SharePhase,
    session_id: &str,
    threshold: usize,
    opened: &[bool],
) -> anyhow::Result<Vec<bool>> {
    let mut confirmed = vec![false; internals.len()];
    for (i, client) in internals.iter_mut().enumerate() {
        match client
            .close_round(CloseRoundRequest {
                round_id,
                depth,
                phase: phase as i32,
                session_id: session_id.to_string(),
            })
            .await
        {
            Ok(_) => confirmed[i] = true,
            Err(e) => tracing::warn!(round_id, depth, error = %e, "close_round failed"),
        }
    }
    for (c, &o) in confirmed.iter_mut().zip(opened) {
        *c = *c && o;
    }
    let ok = confirmed.iter().filter(|&&c| c).count();
    anyhow::ensure!(
        ok >= threshold,
        "close_round confirmed-and-opened on only {ok} of {} shareholders (threshold {threshold}); \
         refusing to gather from a possibly-unfrozen pool",
        confirmed.len()
    );
    Ok(confirmed)
}

/// Tell every shareholder the session is over so it can free the pool.
/// Best-effort: an unreachable shareholder just keeps the pool until the
/// periodic idle sweep reclaims it, so this never fails the session.
pub(super) async fn end_session_all(internals: &mut Internals, session_id: &str) {
    for client in internals.iter_mut() {
        if let Err(e) = client
            .end_session(EndSessionRequest {
                session_id: session_id.to_string(),
            })
            .await
        {
            tracing::warn!(error = %e, session_id, "end_session failed; pool will age out");
        }
    }
}

/// Fetch each shareholder's current commitment set for `(phase, depth)`;
/// an unreachable shareholder contributes an empty set.
async fn list_commitment_sets(
    internals: &mut Internals,
    phase: SharePhase,
    depth: u32,
    session_id: &str,
) -> Vec<BTreeSet<Vec<u8>>> {
    let mut sets = Vec::with_capacity(internals.len());
    for client in internals.iter_mut() {
        match client
            .list_commitments(ListCommitmentsRequest {
                phase: phase as i32,
                depth,
                session_id: session_id.to_string(),
            })
            .await
        {
            Ok(resp) => sets.push(resp.into_inner().commitments.into_iter().collect()),
            Err(e) => {
                tracing::warn!(error = %e, "list_commitments failed during submission window");
                sets.push(BTreeSet::new());
            }
        }
    }
    sets
}

/// Size of the intersection across every shareholder.
fn intersect_all(sets: &[BTreeSet<Vec<u8>>]) -> usize {
    let Some((first, rest)) = sets.split_first() else {
        return 0;
    };
    first
        .iter()
        .filter(|c| rest.iter().all(|s| s.contains(*c)))
        .count()
}

/// Largest intersection over any `threshold`-sized subset of shareholders, the
/// overlap rule `pbr_core::Aggregator::select_shareholders` applies at gather
/// time, so a deadline close never counts clients the gather cannot
/// reconstruct.
fn best_threshold_overlap(sets: &[BTreeSet<Vec<u8>>], threshold: usize) -> usize {
    let n = sets.len();
    if threshold == 0 || threshold > n {
        return 0;
    }
    let mut best = 0;
    for mask in 1u32..(1 << n) {
        if mask.count_ones() as usize != threshold {
            continue;
        }
        let members: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let count = sets[members[0]]
            .iter()
            .filter(|c| members[1..].iter().all(|&i| sets[i].contains(*c)))
            .count();
        best = best.max(count);
    }
    best
}

/// Wait out the submission window for the open round and return the number of
/// usable client commitments once it may be closed.
///
/// A commitment counted here implies that client's full per-node contribution
/// is present on that shareholder, because it arrives as one atomic
/// `SubmitGradientBatch`, so a count-triggered close cannot split a
/// contribution. The grace sleep is a participation nicety, not a correctness
/// requirement: it gives batches still in flight to other shareholders a moment
/// to land everywhere.
#[allow(clippy::too_many_arguments)]
pub(super) async fn await_submissions(
    internals: &mut Internals,
    phase: SharePhase,
    depth: u32,
    session_id: &str,
    threshold: usize,
    target_clients: usize,
    min_clients: usize,
    window: Duration,
) -> anyhow::Result<usize> {
    let poll_every = (window / 20).clamp(Duration::from_millis(10), Duration::from_millis(250));
    let grace = poll_every.max(Duration::from_millis(100));
    let started = Instant::now();
    let give_up = window.saturating_mul(GIVE_UP_WINDOWS);
    let n = loop {
        let sets = list_commitment_sets(internals, phase, depth, session_id).await;
        let everywhere = intersect_all(&sets);
        if everywhere >= target_clients {
            break everywhere;
        }
        let elapsed = started.elapsed();
        if elapsed >= window {
            let best = best_threshold_overlap(&sets, threshold);
            if best >= min_clients {
                break best;
            }
            if elapsed >= give_up {
                anyhow::bail!(
                    "round (phase {phase:?}, depth {depth}) did not reach min_clients={min_clients} \
                     within {GIVE_UP_WINDOWS} submission windows (best overlap {best})"
                );
            }
        }
        tokio::time::sleep(poll_every).await;
    };
    tokio::time::sleep(grace).await;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregator::service::SessionState;
    use crate::aggregator::SessionSpec;
    use crate::test_support::{RecordingServer, WedgedServer};
    use pbr_proto::v1::{RoundContext, SessionPhase};
    use std::sync::{Arc, Mutex};
    use std::time::SystemTime;

    /// Field values are arbitrary: no round-lifecycle test reads them.
    fn test_spec() -> SessionSpec {
        SessionSpec {
            dataset_id: "d".into(),
            title: "t".into(),
            n_trees: 1,
            max_depth: 1,
            n_bins: 4,
            learning_rate: 0.1,
            lambda: 1.0,
            min_clients: 1,
            target_clients: 1,
            submission_window_ms: 1000,
        }
    }

    fn shareholder_test_cfg(x_coord: u64) -> crate::config::ShareholderConfig {
        crate::config::ShareholderConfig {
            x_coord,
            min_clients: 1,
            listen: "127.0.0.1:0".parse().unwrap(),
            internal_listen: "127.0.0.1:0".parse().unwrap(),
            auth: crate::config::AuthConfig {
                issuer: "https://test-issuer.local".into(),
                audience: "pbr".into(),
                static_keys: vec![crate::config::StaticKey {
                    kid: "test-1".into(),
                    public_key_pem_path: concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/tests/fixtures/test_key.pub.pem"
                    )
                    .into(),
                }],
                google_jwks_url: None,
            },
            tls: None,
        }
    }

    /// A client pointed at a reserved-then-closed port, so every RPC on it
    /// fails: an unreachable shareholder.
    fn dead_internal_client() -> ShareholderInternalClient<Channel> {
        let dead_port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        ShareholderInternalClient::new(
            tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{dead_port}"))
                .unwrap()
                .connect_lazy(),
        )
    }

    #[tokio::test]
    async fn close_round_all_requires_threshold_confirmations() {
        let crate::shareholder::RunningShareholder {
            internal_addr: internal,
            handle: h,
            ..
        } = crate::shareholder::serve(shareholder_test_cfg(1))
            .await
            .unwrap();
        let live = ShareholderInternalClient::connect(format!("http://{internal}"))
            .await
            .unwrap();
        let dead = dead_internal_client();
        let mut internals: Internals = vec![live, dead];

        // Both count as opened: this test is about close confirmation alone.
        // `close_round_all_masks_out_unopened` covers the open mask.
        let opened = vec![true, true];

        let res = close_round_all(&mut internals, 1, 0, SharePhase::Gradient, "", 2, &opened).await;
        assert!(
            res.is_err(),
            "close_round_all must not succeed when a required CloseRound failed"
        );

        let confirmed = close_round_all(&mut internals, 1, 0, SharePhase::Gradient, "", 1, &opened)
            .await
            .unwrap();
        assert_eq!(confirmed, vec![true, false]);
        h.shutdown();
    }

    #[tokio::test]
    async fn open_round_below_threshold_fails_fast() {
        let crate::shareholder::RunningShareholder {
            internal_addr: internal,
            handle: h,
            ..
        } = crate::shareholder::serve(shareholder_test_cfg(1))
            .await
            .unwrap();
        let live = ShareholderInternalClient::connect(format!("http://{internal}"))
            .await
            .unwrap();
        let dead = dead_internal_client();
        let mut internals: Internals = vec![live, dead];

        // Only the live shareholder can open, so this must fail before any
        // submission window, well under the give-up budget a window-based
        // failure would burn.
        let started = Instant::now();
        let res = open_round_all(&mut internals, 1, 0, SharePhase::Gradient, "", 2).await;
        assert!(
            res.is_err(),
            "open_round_all must not succeed below threshold"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "below-threshold open must fail fast, not wait out give-up windows"
        );

        let opened = open_round_all(&mut internals, 1, 0, SharePhase::Gradient, "", 1)
            .await
            .unwrap();
        assert_eq!(opened, vec![true, false]);
        h.shutdown();
    }

    #[tokio::test]
    async fn close_round_all_masks_out_unopened_shareholders() {
        // Only the first is told to open. The second's CloseRound Acks via the
        // idempotent "already closed" path, which alone would wrongly mark it a
        // live gather source.
        let crate::shareholder::RunningShareholder {
            internal_addr: internal1,
            handle: h1,
            ..
        } = crate::shareholder::serve(shareholder_test_cfg(1))
            .await
            .unwrap();
        let crate::shareholder::RunningShareholder {
            internal_addr: internal2,
            handle: h2,
            ..
        } = crate::shareholder::serve(shareholder_test_cfg(2))
            .await
            .unwrap();
        let mut opener = ShareholderInternalClient::connect(format!("http://{internal1}"))
            .await
            .unwrap();
        opener
            .open_round(OpenRoundRequest {
                round_id: 1,
                depth: 0,
                phase: SharePhase::Gradient as i32,
                session_id: String::new(),
            })
            .await
            .unwrap();

        let client1 = ShareholderInternalClient::connect(format!("http://{internal1}"))
            .await
            .unwrap();
        let client2 = ShareholderInternalClient::connect(format!("http://{internal2}"))
            .await
            .unwrap();
        let mut internals: Internals = vec![client1, client2];
        let opened = vec![true, false];

        let confirmed = close_round_all(&mut internals, 1, 0, SharePhase::Gradient, "", 1, &opened)
            .await
            .unwrap();
        assert_eq!(
            confirmed,
            vec![true, false],
            "a shareholder that never opened the round must be masked out even though its \
             CloseRound Acked"
        );
        h1.shutdown();
        h2.shutdown();
    }

    #[test]
    fn overlap_counts() {
        let set = |ids: &[u8]| -> BTreeSet<Vec<u8>> { ids.iter().map(|&i| vec![i]).collect() };
        let sets = vec![set(&[1, 2, 3]), set(&[2, 3, 4]), set(&[3, 4, 5])];
        assert_eq!(intersect_all(&sets), 1); // only {3} everywhere
        assert_eq!(best_threshold_overlap(&sets, 2), 2); // {2,3} or {3,4}
        assert_eq!(best_threshold_overlap(&sets, 3), 1);
        assert_eq!(best_threshold_overlap(&sets, 4), 0);
        // A dead shareholder (empty set) blocks the all-intersection but
        // not the threshold overlap.
        let with_dead = vec![set(&[1, 2, 3]), set(&[1, 2, 3]), set(&[])];
        assert_eq!(intersect_all(&with_dead), 0);
        assert_eq!(best_threshold_overlap(&with_dead, 2), 3);
    }

    /// An unresponsive-but-connected shareholder must not hang the round loop.
    /// The per-RPC deadline makes `list_commitment_sets` return an empty set for
    /// it instead of blocking forever; without the deadline the outer timeout
    /// below trips.
    #[tokio::test]
    async fn internal_rpc_times_out_against_a_wedged_shareholder() {
        let server = WedgedServer::spawn().await;

        // The server speaks HTTP/2, so the connection establishes; the deadline
        // bites only on the wedged RPC.
        let mut internals: Internals = vec![
            connect_internal(&format!("http://{}", server.addr))
                .await
                .unwrap(),
        ];

        let started = Instant::now();
        let sets = tokio::time::timeout(
            Duration::from_secs(20),
            list_commitment_sets(&mut internals, SharePhase::Gradient, 0, ""),
        )
        .await
        .expect("list_commitment_sets must return once the per-RPC deadline fires, not hang");
        assert_eq!(sets.len(), 1);
        assert!(
            sets[0].is_empty(),
            "a wedged shareholder contributes an empty commitment set"
        );
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the internal-plane RPC deadline must fire well within the give-up budget"
        );

        server.shutdown().await;
    }

    /// A URL on a reserved-then-closed port: connecting always fails, standing
    /// in for a shareholder down at startup.
    fn dead_endpoint_url() -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://127.0.0.1:{}", l.local_addr().unwrap().port())
    }

    /// A shareholder unreachable at startup must not fail the session while
    /// `threshold` others are up, the same tolerance the round loop applies
    /// mid-session. Below threshold, startup still fails.
    #[tokio::test]
    async fn startup_tolerates_dead_shareholders_down_to_threshold() {
        let crate::shareholder::RunningShareholder {
            internal_addr: internal,
            handle: h,
            ..
        } = crate::shareholder::serve(shareholder_test_cfg(1))
            .await
            .unwrap();
        let endpoints = vec![format!("http://{internal}"), dead_endpoint_url()];

        let internals = connect_all_internal(&endpoints, 1).await.unwrap();
        assert_eq!(
            internals.len(),
            endpoints.len(),
            "every endpoint keeps a positional slot even when unreachable"
        );

        assert!(
            connect_all_internal(&endpoints, 2).await.is_err(),
            "startup must fail when fewer than threshold shareholders are reachable"
        );
        h.shutdown();
    }

    /// On session failure both `CloseRound` and `EndSession` must reach every
    /// shareholder. Pool state cannot tell the two apart (`end_session` removes
    /// the pool, and `close_round` against a removed pool is an idempotent Ack),
    /// so this asserts on the RPCs a [`RecordingServer`] received: it fails if
    /// the close broadcast is ever dropped while end_session keeps firing.
    #[tokio::test]
    async fn best_effort_close_broadcasts_close_round_and_end_session() {
        let server = RecordingServer::spawn().await;
        let ep = format!("http://{}", server.addr);

        // A session whose published context describes the round the
        // aggregator left open.
        let session: SharedSession = Arc::new(Mutex::new(SessionState {
            phase: SessionPhase::Training,
            ctx: Some(RoundContext {
                tree_idx: 0,
                depth: 2,
                round_id: 7,
                active_node_ids: Vec::new(),
                splits_so_far: Default::default(),
                bin_edges: Vec::new(),
                model: None,
                submission_deadline: None,
                session_id: "s".into(),
            }),
            bin_edges: Vec::new(),
            n_features: 0,
            session_id: "s".into(),
            spec: test_spec(),
            created_at: SystemTime::now(),
        }));

        close_round_and_end_session_best_effort(&[ep], &session).await;

        {
            let calls = server.calls.lock().unwrap();
            let close_round = calls
                .close_round
                .as_ref()
                .expect("close_round_and_end_session_best_effort must send CloseRound");
            assert_eq!(close_round.round_id, 7, "must close the published round");
            assert_eq!(close_round.depth, 2);
            assert_eq!(close_round.phase, SharePhase::Gradient as i32);
            assert_eq!(close_round.session_id, "s");
            assert!(
                calls.end_session.is_some(),
                "close_round_and_end_session_best_effort must send EndSession"
            );
        }

        server.shutdown().await;
    }

    /// On session failure the aggregator also ends the session on every
    /// shareholder, so the pool is freed immediately rather than sitting until
    /// `SESSION_IDLE_TTL` sweeps it. Observed via `list_commitments`: it
    /// succeeds while the pool exists and fails with `FailedPrecondition` once
    /// the pool is gone.
    #[tokio::test]
    async fn best_effort_cleanup_also_frees_the_shareholder_pool() {
        let crate::shareholder::RunningShareholder {
            internal_addr: internal,
            handle: h,
            ..
        } = crate::shareholder::serve(shareholder_test_cfg(1))
            .await
            .unwrap();
        let ep = format!("http://{internal}");
        let mut client = ShareholderInternalClient::connect(ep.clone())
            .await
            .unwrap();
        client
            .open_round(OpenRoundRequest {
                round_id: 7,
                depth: 2,
                phase: SharePhase::Gradient as i32,
                session_id: "s".into(),
            })
            .await
            .unwrap();

        // The pool exists: list_commitments succeeds.
        client
            .list_commitments(ListCommitmentsRequest {
                phase: SharePhase::Gradient as i32,
                depth: 2,
                session_id: "s".into(),
            })
            .await
            .expect("pool exists before cleanup");

        let session: SharedSession = Arc::new(Mutex::new(SessionState {
            phase: SessionPhase::Training,
            ctx: Some(RoundContext {
                tree_idx: 0,
                depth: 2,
                round_id: 7,
                active_node_ids: Vec::new(),
                splits_so_far: Default::default(),
                bin_edges: Vec::new(),
                model: None,
                submission_deadline: None,
                session_id: "s".into(),
            }),
            bin_edges: Vec::new(),
            n_features: 0,
            session_id: "s".into(),
            spec: test_spec(),
            created_at: SystemTime::now(),
        }));

        close_round_and_end_session_best_effort(&[ep], &session).await;

        // The pool is gone: the same list_commitments now fails.
        let res = client
            .list_commitments(ListCommitmentsRequest {
                phase: SharePhase::Gradient as i32,
                depth: 2,
                session_id: "s".into(),
            })
            .await;
        assert!(
            res.is_err(),
            "the session's pool must be freed by the best-effort EndSession broadcast"
        );
        h.shutdown();
    }
}
