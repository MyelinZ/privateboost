//! Client-facing submission validation and storage: what a submission must
//! carry to be accepted (auth, matching x, one vector length, an atomic
//! batch), and that accepted shares list and sum correctly through the
//! internal service.

use crate::{
    AUD, ISS, KID, PRIV, STATS_DEPTH_SENTINEL, bearer, cfg, close_round, gradient_batch, open_round,
};
use pbr_client::jwt::mint;
use pbr_proto::convert::{commitment_to_bytes, share_from_proto, share_to_proto};
use pbr_proto::v1::shareholder_internal_client::ShareholderInternalClient;
use pbr_proto::v1::shareholder_service_client::ShareholderServiceClient;
use pbr_proto::v1::{
    GetSumsRequest, GradientBatchSubmission, ListCommitmentsRequest, SharePhase,
    StatsShareSubmission,
};
use pbr_server::shareholder::{RunningShareholder, serve};
use pbr_core::{Client, decode_all, reconstruct};
use tonic::Request;

#[tokio::test]
async fn distributed_stats_sum_equals_plaintext() {
    // Two shareholder daemons (x=1, x=2), threshold 2-of-3 clients.
    let RunningShareholder {
        client_addr: addr1,
        internal_addr: internal1,
        handle: h1,
    } = serve(cfg(1)).await.unwrap();
    let RunningShareholder {
        client_addr: addr2,
        internal_addr: internal2,
        handle: h2,
    } = serve(cfg(2)).await.unwrap();

    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    let mut svc1 = ShareholderServiceClient::connect(format!("http://{addr1}"))
        .await
        .unwrap();
    let mut svc2 = ShareholderServiceClient::connect(format!("http://{addr2}"))
        .await
        .unwrap();
    let mut int1 = ShareholderInternalClient::connect(format!("http://{internal1}"))
        .await
        .unwrap();
    let mut int2 = ShareholderInternalClient::connect(format!("http://{internal2}"))
        .await
        .unwrap();

    open_round(&mut int1, 0, STATS_DEPTH_SENTINEL, SharePhase::Stats).await;
    open_round(&mut int2, 0, STATS_DEPTH_SENTINEL, SharePhase::Stats).await;

    // 6 clients with 2 features each; keep plaintext sums for comparison.
    let feature_rows: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, 10.0 + i as f64]).collect();

    for (i, feats) in feature_rows.iter().enumerate() {
        let mut client = Client::new(format!("c{i}"), feats.clone(), 1.0, 3, 2, Some(i as u64));
        let shares = client.compute_stat_shares().unwrap();
        // Only two daemons exist (x=1, x=2); route by evaluation point and
        // drop x=3 entirely. Reconstruction from 2 of 3 shares is the
        // protocol's own threshold claim, so this doubles as that check.
        for share in shares {
            let svc = match share.share.x {
                1 => &mut svc1,
                2 => &mut svc2,
                3 => continue,
                other => panic!("unexpected evaluation point {other}"),
            };
            let mut req = Request::new(StatsShareSubmission {
                commitment: commitment_to_bytes(&share.commitment),
                share: Some(share_to_proto(&share.share)),
                session_id: String::new(),
            });
            req.metadata_mut().insert("authorization", bearer(&token));
            svc.submit_stats_shares(req).await.unwrap();
        }
    }

    close_round(&mut int1, 0, STATS_DEPTH_SENTINEL, SharePhase::Stats).await;
    close_round(&mut int2, 0, STATS_DEPTH_SENTINEL, SharePhase::Stats).await;

    // Aggregate through the internal service, reconstruct, compare.
    let list = int1
        .list_commitments(ListCommitmentsRequest {
            phase: SharePhase::Stats as i32,
            depth: 0,
            session_id: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(list.commitments.len(), 6);

    let req = GetSumsRequest {
        phase: SharePhase::Stats as i32,
        depth: 0,
        commitments: list.commitments.clone(),
        node_id: 0,
        session_id: String::new(),
    };
    let s1 = share_from_proto(
        int1.get_sums(req.clone())
            .await
            .unwrap()
            .into_inner()
            .share
            .unwrap(),
    )
    .unwrap();
    let s2 = share_from_proto(
        int2.get_sums(req)
            .await
            .unwrap()
            .into_inner()
            .share
            .unwrap(),
    )
    .unwrap();
    let totals = decode_all(&reconstruct(&[s1, s2], 2).unwrap());

    // `pbr-core`'s stats vector layout: x, x^2 per feature, then y, y^2.
    let expect_f0: f64 = feature_rows.iter().map(|r| r[0]).sum();
    let expect_f1: f64 = feature_rows.iter().map(|r| r[1]).sum();
    assert!((totals[0] - expect_f0).abs() < 1e-4);
    assert!((totals[2] - expect_f1).abs() < 1e-4);

    h1.shutdown();
    h2.shutdown();
}

#[tokio::test]
async fn unauthenticated_submission_rejected() {
    let RunningShareholder {
        client_addr: addr,
        handle: h,
        ..
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let res = svc
        .submit_stats_shares(StatsShareSubmission::default())
        .await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::Unauthenticated);
    h.shutdown();
}

#[tokio::test]
async fn min_clients_enforced_on_sums() {
    let RunningShareholder {
        internal_addr: internal,
        handle: h,
        ..
    } = serve(cfg(1)).await.unwrap();
    let mut int = ShareholderInternalClient::connect(format!("http://{internal}"))
        .await
        .unwrap();
    let res = int
        .get_sums(GetSumsRequest {
            phase: SharePhase::Stats as i32,
            depth: 0,
            // two identical commitments dedupe to one distinct client, below min_clients=5
            commitments: vec![vec![0u8; 32]; 2],
            node_id: 0,
            session_id: String::new(),
        })
        .await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::FailedPrecondition);
    h.shutdown();
}

#[tokio::test]
async fn mismatched_share_x_rejected() {
    // Daemon at x=1; send it a share evaluated at x=2 from a 3-party client.
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

    open_round(&mut int, 0, STATS_DEPTH_SENTINEL, SharePhase::Stats).await;

    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares = client.compute_stat_shares().unwrap();
    let wrong_share = shares.into_iter().find(|s| s.share.x == 2).unwrap();

    let mut req = Request::new(StatsShareSubmission {
        commitment: commitment_to_bytes(&wrong_share.commitment),
        share: Some(share_to_proto(&wrong_share.share)),
        session_id: String::new(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    let res = svc.submit_stats_shares(req).await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::InvalidArgument);
    h.shutdown();
}

#[tokio::test]
async fn duplicate_commitments_do_not_satisfy_min_clients() {
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

    open_round(&mut int, 0, STATS_DEPTH_SENTINEL, SharePhase::Stats).await;

    // Submit shares from fewer clients than min_clients (5).
    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares = client.compute_stat_shares().unwrap();
    let share = shares.into_iter().find(|s| s.share.x == 1).unwrap();

    let mut req = Request::new(StatsShareSubmission {
        commitment: commitment_to_bytes(&share.commitment),
        share: Some(share_to_proto(&share.share)),
        session_id: String::new(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    svc.submit_stats_shares(req).await.unwrap();

    close_round(&mut int, 0, STATS_DEPTH_SENTINEL, SharePhase::Stats).await;

    // Repeat the single commitment min_clients times; dedupe must collapse
    // it back down to one distinct client, so the floor is not satisfied.
    let commitment_bytes = commitment_to_bytes(&share.commitment);
    let res = int
        .get_sums(GetSumsRequest {
            phase: SharePhase::Stats as i32,
            depth: 0,
            commitments: vec![commitment_bytes; 5],
            node_id: 0,
            session_id: String::new(),
        })
        .await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::FailedPrecondition);
    h.shutdown();
}

#[tokio::test]
async fn out_of_range_round_id_rejected() {
    let RunningShareholder {
        client_addr: addr,
        handle: h,
        ..
    } = serve(cfg(1)).await.unwrap();
    let mut svc = ShareholderServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    let mut req = Request::new(GradientBatchSubmission {
        round_id: u64::MAX,
        ..Default::default()
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    let res = svc.submit_gradient_batch(req).await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::InvalidArgument);
    h.shutdown();
}

#[tokio::test]
async fn mismatched_vector_length_rejected() {
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

    // First submission: 2 features -> stat-share vector length 6.
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

    // Second submission: 3 features -> stat-share vector length 8. Fixes
    // expected_len at 6 above, so this must be rejected.
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
async fn matching_submission_accepted_and_summable() {
    // Same two-daemon threshold shape as distributed_stats_sum_equals_plaintext,
    // but routed through the gradient round-lifecycle (OpenRound/CloseRound
    // gated on round_id+depth, not just phase).
    let RunningShareholder {
        client_addr: addr1,
        internal_addr: internal1,
        handle: h1,
    } = serve(cfg(1)).await.unwrap();
    let RunningShareholder {
        client_addr: addr2,
        internal_addr: internal2,
        handle: h2,
    } = serve(cfg(2)).await.unwrap();
    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    let mut svc1 = ShareholderServiceClient::connect(format!("http://{addr1}"))
        .await
        .unwrap();
    let mut svc2 = ShareholderServiceClient::connect(format!("http://{addr2}"))
        .await
        .unwrap();
    let mut int1 = ShareholderInternalClient::connect(format!("http://{internal1}"))
        .await
        .unwrap();
    let mut int2 = ShareholderInternalClient::connect(format!("http://{internal2}"))
        .await
        .unwrap();

    let round_id = 7;
    let depth = 0;
    open_round(&mut int1, round_id, depth, SharePhase::Gradient).await;
    open_round(&mut int2, round_id, depth, SharePhase::Gradient).await;

    let feature_rows: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, 10.0 + i as f64]).collect();
    for (i, feats) in feature_rows.iter().enumerate() {
        let mut client = Client::new(format!("c{i}"), feats.clone(), 1.0, 3, 2, Some(i as u64));
        let shares = client.compute_stat_shares().unwrap();
        for share in shares {
            let svc = match share.share.x {
                1 => &mut svc1,
                2 => &mut svc2,
                3 => continue,
                other => panic!("unexpected evaluation point {other}"),
            };
            let mut req = Request::new(gradient_batch(
                round_id,
                depth,
                &share.commitment,
                vec![(0, &share.share)],
                "",
            ));
            req.metadata_mut().insert("authorization", bearer(&token));
            svc.submit_gradient_batch(req).await.unwrap();
        }
    }

    close_round(&mut int1, round_id, depth, SharePhase::Gradient).await;
    close_round(&mut int2, round_id, depth, SharePhase::Gradient).await;

    let list = int1
        .list_commitments(ListCommitmentsRequest {
            phase: SharePhase::Gradient as i32,
            depth,
            session_id: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(list.commitments.len(), 6);

    let req = GetSumsRequest {
        phase: SharePhase::Gradient as i32,
        depth,
        commitments: list.commitments.clone(),
        node_id: 0,
        session_id: String::new(),
    };
    let s1 = share_from_proto(
        int1.get_sums(req.clone())
            .await
            .unwrap()
            .into_inner()
            .share
            .unwrap(),
    )
    .unwrap();
    let s2 = share_from_proto(
        int2.get_sums(req)
            .await
            .unwrap()
            .into_inner()
            .share
            .unwrap(),
    )
    .unwrap();
    let totals = decode_all(&reconstruct(&[s1, s2], 2).unwrap());

    let expect_f0: f64 = feature_rows.iter().map(|r| r[0]).sum();
    assert!((totals[0] - expect_f0).abs() < 1e-4);

    h1.shutdown();
    h2.shutdown();
}

#[tokio::test]
async fn gradient_batch_atomic_on_length_mismatch() {
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

    // One batch, one commitment, two entries of different vector lengths
    // (2 features -> length 6, 3 features -> length 8).
    let mut client_a = Client::new("a".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares_a = client_a.compute_stat_shares().unwrap();
    let share_a = shares_a.into_iter().find(|s| s.share.x == 1).unwrap();
    let mut client_b = Client::new("b".into(), vec![1.0, 2.0, 3.0], 1.0, 3, 2, Some(2));
    let shares_b = client_b.compute_stat_shares().unwrap();
    let share_b = shares_b.into_iter().find(|s| s.share.x == 1).unwrap();

    let mut req = Request::new(gradient_batch(
        5,
        0,
        &share_a.commitment,
        vec![(0, &share_a.share), (1, &share_b.share)],
        "",
    ));
    req.metadata_mut().insert("authorization", bearer(&token));
    let res = svc.submit_gradient_batch(req).await;
    assert_eq!(res.unwrap_err().code(), tonic::Code::InvalidArgument);

    // Atomicity: nothing from the bad batch was stored, no commitment, no
    // node-id, not even the first (well-formed) entry.
    let list = int
        .list_commitments(ListCommitmentsRequest {
            phase: SharePhase::Gradient as i32,
            depth: 0,
            session_id: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(list.commitments.is_empty(), "partial store detected");
    assert!(list.node_ids.is_empty(), "partial store detected");

    // The failed batch must not have fixed the round's expected_len either
    // (validation happens before the gate is touched): a consistent
    // batch of the OTHER length is still accepted as the round's first.
    let mut req2 = Request::new(gradient_batch(
        5,
        0,
        &share_b.commitment,
        vec![(0, &share_b.share)],
        "",
    ));
    req2.metadata_mut().insert("authorization", bearer(&token));
    svc.submit_gradient_batch(req2).await.unwrap();
    h.shutdown();
}

#[tokio::test]
async fn gradient_batch_all_entries_stored() {
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

    // A valid 2-node batch (path-hiding shape): one commitment, entries for
    // nodes 1 and 2 with equal vector lengths.
    let mut client_a = Client::new("a".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(1));
    let shares_a = client_a.compute_stat_shares().unwrap();
    let share_a = shares_a.into_iter().find(|s| s.share.x == 1).unwrap();
    let mut client_b = Client::new("b".into(), vec![3.0, 4.0], 1.0, 3, 2, Some(2));
    let shares_b = client_b.compute_stat_shares().unwrap();
    let share_b = shares_b.into_iter().find(|s| s.share.x == 1).unwrap();

    let mut req = Request::new(gradient_batch(
        5,
        0,
        &share_a.commitment,
        vec![(1, &share_a.share), (2, &share_b.share)],
        "",
    ));
    req.metadata_mut().insert("authorization", bearer(&token));
    svc.submit_gradient_batch(req).await.unwrap();

    // Both node-ids are present under the single commitment.
    let list = int
        .list_commitments(ListCommitmentsRequest {
            phase: SharePhase::Gradient as i32,
            depth: 0,
            session_id: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(list.commitments.len(), 1);
    assert_eq!(list.node_ids, vec![1, 2]);
    h.shutdown();
}
