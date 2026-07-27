//! The open/close round state machine: which submissions the currently
//! accepted round admits, what a close freezes, and what a duplicate or
//! new-session OpenRound does to the pool.

use crate::{
    AUD, ISS, KID, PRIV, STATS_DEPTH_SENTINEL, bearer, cfg, close_round, gradient_batch, open_round,
};
use pbr_client::jwt::mint;
use pbr_proto::convert::{commitment_to_bytes, share_to_proto};
use pbr_proto::v1::shareholder_internal_client::ShareholderInternalClient;
use pbr_proto::v1::shareholder_service_client::ShareholderServiceClient;
use pbr_proto::v1::{
    CloseRoundRequest, EndSessionRequest, ListCommitmentsRequest, OpenRoundRequest, SharePhase,
    StatsShareSubmission,
};
use pbr_server::shareholder::{RunningShareholder, serve};
use pbr_core::Client;
use tonic::Request;
use tonic::transport::Channel;

async fn client(addr: std::net::SocketAddr) -> ShareholderServiceClient<Channel> {
    ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap()
}

async fn internal(addr: std::net::SocketAddr) -> ShareholderInternalClient<Channel> {
    ShareholderInternalClient::connect(format!("http://{addr}"))
        .await
        .unwrap()
}

async fn open_round_session(
    int: &mut ShareholderInternalClient<Channel>,
    round_id: u64,
    depth: u32,
    phase: SharePhase,
    session_id: &str,
) {
    int.open_round(OpenRoundRequest {
        round_id,
        depth,
        phase: phase as i32,
        session_id: session_id.to_string(),
    })
    .await
    .unwrap();
}

/// Build a stats-share submission for client `c{seed}`, seeded so each call
/// produces a distinct commitment.
fn stats_submission(seed: u64, session_id: &str) -> StatsShareSubmission {
    let mut client = Client::new(format!("c{seed}"), vec![1.0, 2.0], 1.0, 3, 2, Some(seed));
    let shares = client.compute_stat_shares().unwrap();
    let share = shares.into_iter().find(|s| s.share.x == 1).unwrap();
    StatsShareSubmission {
        commitment: commitment_to_bytes(&share.commitment),
        share: Some(share_to_proto(&share.share)),
        session_id: session_id.to_string(),
    }
}

/// Submit one client's stats share to `session_id` on the client-facing
/// service and return the count of stats commitments the internal service
/// now lists for that same session.
async fn submit_one_stats_and_count(
    svc: &mut ShareholderServiceClient<Channel>,
    int: &mut ShareholderInternalClient<Channel>,
    token: &str,
    seed: u64,
    session_id: &str,
) -> usize {
    let mut req = Request::new(stats_submission(seed, session_id));
    req.metadata_mut().insert("authorization", bearer(token));
    svc.submit_stats_shares(req).await.unwrap();
    int.list_commitments(ListCommitmentsRequest {
        phase: SharePhase::Stats as i32,
        depth: 0,
        session_id: session_id.to_string(),
    })
    .await
    .unwrap()
    .into_inner()
    .commitments
    .len()
}

/// Submit one stats share into `session_id`, seeded by `seed` so each call
/// produces a distinct commitment. Panics on rejection.
async fn submit_stats_for(
    svc: &mut ShareholderServiceClient<Channel>,
    session_id: &str,
    seed: u64,
) {
    try_submit_stats_for(svc, session_id, seed)
        .await
        .expect("submission should be accepted");
}

/// `submit_stats_for` that surfaces the error instead of panicking.
async fn try_submit_stats_for(
    svc: &mut ShareholderServiceClient<Channel>,
    session_id: &str,
    seed: u64,
) -> Result<(), tonic::Status> {
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();
    let mut req = Request::new(stats_submission(seed, session_id));
    req.metadata_mut().insert("authorization", bearer(&token));
    svc.submit_stats_shares(req).await.map(|_| ())
}

/// Commitments currently pooled for `session_id`'s stats phase.
async fn list_commitments_for(
    int: &mut ShareholderInternalClient<Channel>,
    session_id: &str,
) -> Vec<Vec<u8>> {
    int.list_commitments(ListCommitmentsRequest {
        phase: SharePhase::Stats as i32,
        depth: 0,
        session_id: session_id.to_string(),
    })
    .await
    .unwrap()
    .into_inner()
    .commitments
}

#[tokio::test]
async fn submission_rejected_when_no_round_open() {
    let RunningShareholder {
        client_addr: addr,
        handle: h,
        ..
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares = client.compute_stat_shares().unwrap();
    let share = shares.into_iter().find(|s| s.share.x == 1).unwrap();

    let mut req = Request::new(gradient_batch(
        5,
        0,
        &share.commitment,
        vec![(0, &share.share)],
        "",
    ));
    req.metadata_mut().insert("authorization", bearer(&token));
    let res = svc.submit_gradient_batch(req).await;
    // No session was ever opened (not even the default ""), so the pool
    // lookup itself rejects this before the round gate is ever consulted.
    assert_eq!(res.unwrap_err().code(), tonic::Code::FailedPrecondition);
    h.shutdown();
}

#[tokio::test]
async fn submission_rejected_for_wrong_round() {
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    open_round(&mut int, 5, 0, SharePhase::Gradient).await;

    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares = client.compute_stat_shares().unwrap();
    let share = shares.into_iter().find(|s| s.share.x == 1).unwrap();

    // Round is open for round_id=5, but this submission claims round_id=6.
    let mut req = Request::new(gradient_batch(
        6,
        0,
        &share.commitment,
        vec![(0, &share.share)],
        "",
    ));
    req.metadata_mut().insert("authorization", bearer(&token));
    let res = svc.submit_gradient_batch(req).await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::InvalidArgument);
    h.shutdown();
}

#[tokio::test]
async fn close_round_freezes_pool() {
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    open_round(&mut int, 5, 0, SharePhase::Gradient).await;

    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares = client.compute_stat_shares().unwrap();
    let share = shares.into_iter().find(|s| s.share.x == 1).unwrap();

    let submission = gradient_batch(5, 0, &share.commitment, vec![(0, &share.share)], "");

    let mut req = Request::new(submission.clone());
    req.metadata_mut().insert("authorization", bearer(&token));
    svc.submit_gradient_batch(req).await.unwrap();

    close_round(&mut int, 5, 0, SharePhase::Gradient).await;

    let mut req2 = Request::new(submission);
    req2.metadata_mut().insert("authorization", bearer(&token));
    let res = svc.submit_gradient_batch(req2).await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::InvalidArgument);
    h.shutdown();
}

#[tokio::test]
async fn duplicate_open_round_preserves_length_gate() {
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    open_round(&mut int, 5, 0, SharePhase::Gradient).await;

    // First submission: 2 features -> stat-share vector length 6. Fixes
    // expected_len at 6.
    let mut client_a = Client::new("a".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares_a = client_a.compute_stat_shares().unwrap();
    let share_a = shares_a.into_iter().find(|s| s.share.x == 1).unwrap();

    let mut req = Request::new(gradient_batch(
        5,
        0,
        &share_a.commitment,
        vec![(0, &share_a.share)],
        "",
    ));
    req.metadata_mut().insert("authorization", bearer(&token));
    svc.submit_gradient_batch(req).await.unwrap();

    // A duplicate/retried OpenRound for the same (round_id, depth, phase)
    // must not reopen the length gate.
    open_round(&mut int, 5, 0, SharePhase::Gradient).await;

    // Second submission: 3 features -> stat-share vector length 8. Must
    // still be rejected because the gate survived the duplicate open.
    let mut client_b = Client::new("b".into(), vec![1.0, 2.0, 3.0], 1.0, 3, 2, Some(2));
    let shares_b = client_b.compute_stat_shares().unwrap();
    let share_b = shares_b.into_iter().find(|s| s.share.x == 1).unwrap();

    let mut req2 = Request::new(gradient_batch(
        5,
        0,
        &share_b.commitment,
        vec![(0, &share_b.share)],
        "",
    ));
    req2.metadata_mut().insert("authorization", bearer(&token));
    let res = svc.submit_gradient_batch(req2).await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::InvalidArgument);
    h.shutdown();
}

#[tokio::test]
async fn close_round_mismatch_returns_failed_precondition() {
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    // Open round A (round_id=5, depth=0).
    open_round(&mut int, 5, 0, SharePhase::Gradient).await;

    // Close a different round B (round_id=6, depth=0), a stale/racing
    // close. It must be rejected, not silently Acked, so the aggregator's
    // confirmed-mask cannot be fooled into treating this shareholder as
    // frozen for a round it never held.
    let res = int
        .close_round(CloseRoundRequest {
            round_id: 6,
            depth: 0,
            phase: SharePhase::Gradient as i32,
            session_id: String::new(),
        })
        .await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::FailedPrecondition);

    // Round A must not have been frozen by the mismatched close: a
    // submission to round A is still accepted.
    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares = client.compute_stat_shares().unwrap();
    let share = shares.into_iter().find(|s| s.share.x == 1).unwrap();

    let mut req = Request::new(gradient_batch(
        5,
        0,
        &share.commitment,
        vec![(0, &share.share)],
        "",
    ));
    req.metadata_mut().insert("authorization", bearer(&token));
    svc.submit_gradient_batch(req).await.unwrap();
    h.shutdown();
}

#[tokio::test]
async fn new_session_open_resets_pool_same_session_open_preserves_it() {
    // The stats round always carries round_id=1, depth=u32::MAX, phase=Stats,
    // so a restarted aggregator's first OpenRound is byte-identical to the
    // dead session's except for session_id. The daemon must reset its pool on
    // the session_id change, but a genuine same-session retry (identical
    // session_id) must leave the pool intact.
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    // Session A opens the stats round and one client submits.
    open_round_session(
        &mut int,
        1,
        STATS_DEPTH_SENTINEL,
        SharePhase::Stats,
        "session-A",
    )
    .await;
    assert_eq!(
        submit_one_stats_and_count(&mut svc, &mut int, &token, 1, "session-A").await,
        1
    );

    // A duplicate OpenRound for the same session must not reset the pool: the
    // first commitment survives and a second submission accumulates to 2.
    open_round_session(
        &mut int,
        1,
        STATS_DEPTH_SENTINEL,
        SharePhase::Stats,
        "session-A",
    )
    .await;
    assert_eq!(
        submit_one_stats_and_count(&mut svc, &mut int, &token, 2, "session-A").await,
        2,
        "same-session duplicate OpenRound must preserve the pool"
    );

    // A restart: a NEW session opens the identical (round_id, depth, phase)
    // stats round. The daemon must drop the dead session's commitments.
    open_round_session(
        &mut int,
        1,
        STATS_DEPTH_SENTINEL,
        SharePhase::Stats,
        "session-B",
    )
    .await;
    let after_restart = int
        .list_commitments(ListCommitmentsRequest {
            phase: SharePhase::Stats as i32,
            depth: 0,
            session_id: "session-B".to_string(),
        })
        .await
        .unwrap()
        .into_inner()
        .commitments
        .len();
    assert_eq!(
        after_restart, 0,
        "a new session_id's OpenRound must reset the pool so stale commitments cannot pollute it"
    );

    h.shutdown();
}

/// Two sessions open the same round id (every session's stats round is id 1)
/// on one daemon. Each must accumulate its own pool: before per-session pools
/// the second OpenRound reset the first session's shares to zero.
#[tokio::test]
async fn two_sessions_keep_independent_stats_pools() {
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();

    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-a").await;
    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-b").await;

    // Two clients into session A, one into session B.
    submit_stats_for(&mut svc, "session-a", 1).await;
    submit_stats_for(&mut svc, "session-a", 2).await;
    submit_stats_for(&mut svc, "session-b", 3).await;

    let a = list_commitments_for(&mut int, "session-a").await;
    let b = list_commitments_for(&mut int, "session-b").await;
    assert_eq!(a.len(), 2, "session A kept both of its own submissions");
    assert_eq!(b.len(), 1, "session B kept exactly its own submission");
    assert!(
        a.iter().all(|c| !b.contains(c)),
        "no commitment may appear in both sessions' pools"
    );

    h.shutdown();
}

/// Opening a NEW round in session B must not disturb session A's pool. Before
/// per-session pools, any OpenRound whose session_id differed from the last
/// accepted round reset the single shared ShareHolder.
#[tokio::test]
async fn a_new_round_in_one_session_does_not_reset_another() {
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();

    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-a").await;
    submit_stats_for(&mut svc, "session-a", 1).await;

    // Session B advances through several of its own rounds.
    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-b").await;
    submit_stats_for(&mut svc, "session-b", 2).await;
    open_round_session(&mut int, 1u64 << 32, 0, SharePhase::Gradient, "session-b").await;

    let a = list_commitments_for(&mut int, "session-a").await;
    assert_eq!(
        a.len(),
        1,
        "session A's stats pool survived session B's rounds"
    );

    h.shutdown();
}

/// A submission naming a session with no open round is rejected, exactly as a
/// submission with no round open was before. An unknown session must not
/// silently create a pool, that would let any client allocate unbounded
/// server memory.
#[tokio::test]
async fn submission_for_unknown_session_is_rejected() {
    let RunningShareholder {
        client_addr: addr,
        internal_addr: internal,
        handle: h,
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();

    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-a").await;

    let err = try_submit_stats_for(&mut svc, "session-unknown", 9)
        .await
        .expect_err("a submission for a session with no open round must be rejected");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);

    h.shutdown();
}

/// EndSession frees the session's pool. A later submission naming it is
/// rejected rather than silently reviving a finished session's pool.
#[tokio::test]
async fn end_session_frees_the_pool() {
    let RunningShareholder { client_addr, internal_addr, handle } =
        serve(cfg(1)).await.unwrap();
    let mut svc = client(client_addr).await;
    let mut int = internal(internal_addr).await;

    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-a").await;
    submit_stats_for(&mut svc, "session-a", 1).await;
    assert_eq!(list_commitments_for(&mut int, "session-a").await.len(), 1);

    int.end_session(EndSessionRequest { session_id: "session-a".to_string() })
        .await
        .expect("end_session acks");

    let err = try_submit_stats_for(&mut svc, "session-a", 2)
        .await
        .expect_err("the pool is gone, so a later submission must be rejected");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);

    handle.shutdown();
}

/// EndSession for a session that was never opened is a no-op ack, so an
/// aggregator retrying its end-of-session broadcast cannot fail.
#[tokio::test]
async fn end_session_is_idempotent() {
    let RunningShareholder { internal_addr, handle, .. } =
        serve(cfg(1)).await.unwrap();
    let mut int = internal(internal_addr).await;

    for _ in 0..2 {
        int.end_session(EndSessionRequest { session_id: "never-opened".to_string() })
            .await
            .expect("end_session is idempotent");
    }

    handle.shutdown();
}

/// EndSession must free only the named session.
#[tokio::test]
async fn end_session_leaves_other_sessions_intact() {
    let RunningShareholder { client_addr, internal_addr, handle } =
        serve(cfg(1)).await.unwrap();
    let mut svc = client(client_addr).await;
    let mut int = internal(internal_addr).await;

    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-a").await;
    open_round_session(&mut int, 1, u32::MAX, SharePhase::Stats, "session-b").await;
    submit_stats_for(&mut svc, "session-a", 1).await;
    submit_stats_for(&mut svc, "session-b", 2).await;

    int.end_session(EndSessionRequest { session_id: "session-a".to_string() })
        .await
        .unwrap();

    assert_eq!(
        list_commitments_for(&mut int, "session-b").await.len(),
        1,
        "session B's pool must survive session A ending"
    );

    handle.shutdown();
}
