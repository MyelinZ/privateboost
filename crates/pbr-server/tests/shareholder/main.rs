//! Integration tests for the shareholder daemon, one theme per module:
//! `startup` (invalid config rejected at serve time), `rounds` (the
//! open/close round state machine and pool semantics), and `submission`
//! (payload validation, storage, and sum correctness). Helpers shared
//! across the modules live here.

use pbr_proto::convert::{commitment_to_bytes, share_to_proto};
use pbr_proto::v1::shareholder_internal_client::ShareholderInternalClient;
use pbr_proto::v1::{
    CloseRoundRequest, GradientBatchSubmission, GradientEntry, OpenRoundRequest, SharePhase,
};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
use pbr_core::{Commitment, Share};
use tonic::transport::Channel;

#[path = "../common/mod.rs"]
mod common;
use common::bearer;

mod rounds;
mod startup;
mod submission;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("../fixtures/test_key.pem");

// u32::MAX depth sentinel for stats-phase OpenRound/CloseRound calls, the
// same convention the aggregator's published stats context uses. Stats
// gating matches on phase only, so the exact value carried here is not
// load-bearing for the daemon, it only makes a test's open/close pair
// unambiguous.
const STATS_DEPTH_SENTINEL: u32 = u32::MAX;

fn cfg(x: u64) -> ShareholderConfig {
    ShareholderConfig {
        x_coord: x,
        min_clients: 5,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: AuthConfig {
            issuer: ISS.into(),
            audience: AUD.into(),
            static_keys: vec![StaticKey {
                kid: KID.into(),
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

/// One client's per-round contribution to one shareholder: a single
/// commitment plus `(node_id, share)` entries.
fn gradient_batch(
    round_id: u64,
    depth: u32,
    commitment: &Commitment,
    entries: Vec<(u32, &Share)>,
    session_id: &str,
) -> GradientBatchSubmission {
    GradientBatchSubmission {
        round_id,
        depth,
        commitment: commitment_to_bytes(commitment),
        entries: entries
            .into_iter()
            .map(|(node_id, share)| GradientEntry {
                node_id,
                share: Some(share_to_proto(share)),
            })
            .collect(),
        session_id: session_id.to_string(),
    }
}

async fn open_round(
    int: &mut ShareholderInternalClient<Channel>,
    round_id: u64,
    depth: u32,
    phase: SharePhase,
) {
    int.open_round(OpenRoundRequest {
        round_id,
        depth,
        phase: phase as i32,
        session_id: String::new(),
    })
    .await
    .unwrap();
}

async fn close_round(
    int: &mut ShareholderInternalClient<Channel>,
    round_id: u64,
    depth: u32,
    phase: SharePhase,
) {
    int.close_round(CloseRoundRequest {
        round_id,
        depth,
        phase: phase as i32,
        session_id: String::new(),
    })
    .await
    .unwrap();
}
