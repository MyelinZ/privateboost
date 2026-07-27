use pbr_client::jwt::mint;
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_proto::v1::shareholder_service_client::ShareholderServiceClient;
use pbr_proto::v1::{EnrollRequest, ListSessionsRequest, StatsShareSubmission};
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{DatasetTable, RunningAggregator, serve as serve_aggregator};
use pbr_server::auth::{Verifier, VerifierKey};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
use pbr_server::shareholder::{RunningShareholder, serve as serve_shareholder};
use std::sync::Arc;
use tonic::Request;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("fixtures/test_key.pem");
const PUB: &[u8] = include_bytes!("fixtures/test_key.pub.pem");

const KID_B: &str = "test-2";
const PRIV_B: &[u8] = include_bytes!("fixtures/jwks_key.pem");
const PUB_B: &[u8] = include_bytes!("fixtures/jwks_key.pub.pem");

const JWKS_JSON: &[u8] = include_bytes!("fixtures/jwks.json");
const JWKS_KID: &str = "jwks-test-1";

fn verifier() -> Arc<Verifier> {
    Arc::new(
        Verifier::from_static(
            ISS.into(),
            AUD.into(),
            vec![VerifierKey {
                kid: KID.into(),
                pem: PUB.to_vec(),
            }],
        )
        .unwrap(),
    )
}

#[test]
fn valid_token_yields_identity() {
    let token = mint(ISS, AUD, KID, "device-42", 300, PRIV).unwrap();
    let id = verifier().verify(&token).unwrap();
    assert_eq!(id.issuer, ISS);
    assert_eq!(id.subject, "device-42");
}

#[test]
fn wrong_audience_rejected() {
    let token = mint(ISS, "someone-else", KID, "device-42", 300, PRIV).unwrap();
    assert!(verifier().verify(&token).is_err());
}

#[test]
fn unknown_kid_rejected() {
    let token = mint(ISS, AUD, "other-kid", "device-42", 300, PRIV).unwrap();
    assert!(verifier().verify(&token).is_err());
}

#[test]
fn expired_token_rejected() {
    // leeway is 60 s; an exp 10 minutes in the past must fail
    let token = mint_with_exp_offset(-600);
    assert!(verifier().verify(&token).is_err());
}

fn mint_with_exp_offset(offset_secs: i64) -> String {
    // mint() sets exp to now + ttl, so it cannot produce an already-expired
    // token; encode the claims directly to place exp at now + offset_secs.
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    #[derive(serde::Serialize)]
    struct Claims {
        iss: String,
        aud: String,
        sub: String,
        exp: i64,
        iat: i64,
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(KID.into());
    encode(
        &header,
        &Claims {
            iss: ISS.into(),
            aud: AUD.into(),
            sub: "device-42".into(),
            exp: now + offset_secs,
            iat: now - 1,
        },
        &EncodingKey::from_rsa_pem(PRIV).unwrap(),
    )
    .unwrap()
}

#[test]
fn interceptor_rejects_missing_authorization_header() {
    use tonic::Request;
    let req = Request::new(());
    let f = pbr_server::auth::interceptor(verifier());
    let result = f(req);
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::Unauthenticated);
}

#[test]
fn interceptor_rejects_non_bearer_scheme() {
    use tonic::Request;
    let mut req = Request::new(());
    req.metadata_mut()
        .insert("authorization", "Basic abc123".parse().unwrap());
    let f = pbr_server::auth::interceptor(verifier());
    let result = f(req);
    assert!(result.is_err());
    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::Unauthenticated);
}

#[test]
fn interceptor_accepts_valid_bearer_and_inserts_identity() {
    use tonic::Request;
    let token = mint(ISS, AUD, KID, "device-42", 300, PRIV).unwrap();
    let mut req = Request::new(());
    req.metadata_mut().insert(
        "authorization",
        format!("Bearer {}", token).parse().unwrap(),
    );
    let f = pbr_server::auth::interceptor(verifier());
    let result = f(req);
    assert!(result.is_ok());
    let req = result.unwrap();
    let identity = req.extensions().get::<pbr_server::auth::Identity>();
    assert!(identity.is_some());
    let identity = identity.unwrap();
    assert_eq!(identity.subject, "device-42");
}

#[test]
fn key_rotation_via_update_keys() {
    let v = verifier();

    // Key A (the fixture the Verifier was constructed with) verifies.
    let token_a = mint(ISS, AUD, KID, "device-42", 300, PRIV).unwrap();
    assert!(v.verify(&token_a).is_ok());

    // Rotate to only key B.
    v.update_keys(vec![VerifierKey {
        kid: KID_B.into(),
        pem: PUB_B.to_vec(),
    }])
    .unwrap();

    // The old A-signed token now fails (unknown kid) ...
    let token_a2 = mint(ISS, AUD, KID, "device-42", 300, PRIV).unwrap();
    assert!(v.verify(&token_a2).is_err());

    // ... and a B-signed token verifies.
    let token_b = mint(ISS, AUD, KID_B, "device-99", 300, PRIV_B).unwrap();
    let id = v.verify(&token_b).unwrap();
    assert_eq!(id.subject, "device-99");
}

#[test]
fn update_keys_rejects_empty_and_keeps_previous() {
    let v = verifier();
    let token = mint(ISS, AUD, KID, "device-42", 300, PRIV).unwrap();
    assert!(v.verify(&token).is_ok());

    // A refresh that parsed to zero usable keys must be refused, not stored.
    assert!(
        v.update_keys(Vec::new()).is_err(),
        "an empty key set must be refused"
    );

    // The previously-loaded key still verifies: no silent lockout on a bad
    // refresh.
    let token2 = mint(ISS, AUD, KID, "device-42", 300, PRIV).unwrap();
    assert!(
        v.verify(&token2).is_ok(),
        "an empty refresh must keep the previous keys"
    );
}

/// Google publishes the Firebase `securetoken` signing keys as a standard
/// JWKS; `Verifier::update_keys_from_jwks` is the parse+swap for that shape.
/// End-to-end proof: the fixture JWKS (whose single key is `jwks_key.pem`'s
/// public half) loads, and a token signed by the matching private key
/// verifies under the JWKS `kid`.
#[test]
fn update_keys_from_jwks_loads_google_shaped_keys() {
    let v = verifier();
    v.update_keys_from_jwks(JWKS_JSON).unwrap();

    let token = mint(ISS, AUD, JWKS_KID, "device-13", 300, PRIV_B).unwrap();
    let id = v.verify(&token).unwrap();
    assert_eq!(id.subject, "device-13");
}

/// One bad JWKS entry must not sink the whole batch: the good key still
/// loads and the entry whose modulus fails to decode is skipped.
#[test]
fn update_keys_from_jwks_skips_bad_entries() {
    let jwks: serde_json::Value = serde_json::from_slice(JWKS_JSON).unwrap();
    let good_key = jwks["keys"][0].clone();
    let mixed = serde_json::json!({
        "keys": [
            good_key,
            { "kty": "RSA", "kid": "bad-kid", "use": "sig", "alg": "RS256",
              "n": "!!!not-base64url!!!", "e": "AQAB" },
        ]
    });

    let v = verifier();
    v.update_keys_from_jwks(&serde_json::to_vec(&mixed).unwrap())
        .unwrap();

    // The good key was swapped in; the bad entry was skipped, not stored.
    let good = mint(ISS, AUD, JWKS_KID, "device-13", 300, PRIV_B).unwrap();
    assert!(v.verify(&good).is_ok());
    let bad = mint(ISS, AUD, "bad-kid", "device-13", 300, PRIV_B).unwrap();
    assert!(
        v.verify(&bad).is_err(),
        "a skipped JWKS entry must not verify anything"
    );
}

/// Hits the real Google endpoint; network-dependent, so `#[ignore]`d by
/// default. Run manually with:
/// `cargo test -p pbr-server --test auth live_google_jwks_loads -- --ignored`
#[tokio::test]
#[ignore]
async fn live_google_jwks_loads() {
    let url =
        "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com";
    verifier()
        .refresh_from_jwks_url(url)
        .await
        .expect("live Firebase JWKS must fetch and yield at least one usable key");
}

/// Encode a token with the given header and signing key but otherwise-valid
/// claims (iss/aud match the verifier; exp 5 min out). The rejection tests
/// below vary only the header (alg, kid) or the signing key; the expiry case
/// is served by `mint_with_exp_offset`, which needs a past `exp`.
fn encode_with_header(header: jsonwebtoken::Header, key: &jsonwebtoken::EncodingKey) -> String {
    use jsonwebtoken::encode;
    #[derive(serde::Serialize)]
    struct Claims {
        iss: String,
        aud: String,
        sub: String,
        exp: i64,
        iat: i64,
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    encode(
        &header,
        &Claims {
            iss: ISS.into(),
            aud: AUD.into(),
            sub: "device-42".into(),
            exp: now + 300,
            iat: now - 1,
        },
        key,
    )
    .unwrap()
}

/// RS256, correctly signed, but the header carries no `kid` at all.
fn tok_missing_kid() -> String {
    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
    encode_with_header(
        header,
        &jsonwebtoken::EncodingKey::from_rsa_pem(PRIV).unwrap(),
    )
}

/// HS256 (symmetric) rather than RS256, with an otherwise-known kid, the
/// alg-confusion shape `Verifier::verify` rejects before any key lookup.
fn tok_hs256() -> String {
    let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256);
    header.kid = Some(KID.into());
    encode_with_header(
        header,
        &jsonwebtoken::EncodingKey::from_secret(b"not-the-rsa-key"),
    )
}

fn server_auth_cfg() -> AuthConfig {
    AuthConfig {
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
    }
}

/// Minimal aggregator config: only the JWT-gated client-facing listener is
/// exercised, so the shareholder endpoints never need to be real (the round
/// loop fails in the background, off the RPC path).
fn minimal_agg_cfg() -> AggregatorConfig {
    AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: vec!["http://127.0.0.1:1".into()],
        client_shareholder_endpoints: vec!["http://127.0.0.1:2".into()],
        threshold: 1,
        auth: server_auth_cfg(),
        fcm: None,
        tls: None,
        datasets: DatasetTable::default(),
        admin_token: None,
        state_path: ":memory:".into(),
        eval: None,
    }
}

fn minimal_sh_cfg() -> ShareholderConfig {
    ShareholderConfig {
        x_coord: 1,
        min_clients: 1,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: server_auth_cfg(),
        tls: None,
    }
}

fn bearer_request<T>(msg: T, token: &str) -> Request<T> {
    let mut req = Request::new(msg);
    req.metadata_mut()
        .insert("authorization", format!("Bearer {token}").parse().unwrap());
    req
}

async fn assert_aggregator_rejects(agg_url: &str, token: &str, case: &str) {
    let mut agg = AggregatorServiceClient::connect(agg_url.to_string())
        .await
        .unwrap();
    let status = agg
        .enroll_session(bearer_request(
            EnrollRequest {
                session_id: String::new(),
            },
            token,
        ))
        .await
        .err()
        .unwrap_or_else(|| panic!("aggregator accepted a {case} token"));
    assert_eq!(
        status.code(),
        tonic::Code::Unauthenticated,
        "aggregator must reject a {case} token as Unauthenticated (got {status:?})"
    );

    let status = agg
        .list_sessions(bearer_request(ListSessionsRequest {}, token))
        .await
        .err()
        .unwrap_or_else(|| panic!("aggregator accepted a {case} token via ListSessions"));
    assert_eq!(
        status.code(),
        tonic::Code::Unauthenticated,
        "aggregator must reject a {case} token as Unauthenticated via ListSessions (got {status:?})"
    );
}

async fn assert_shareholder_rejects(sh_url: &str, token: &str, case: &str) {
    let mut sh = ShareholderServiceClient::connect(sh_url.to_string())
        .await
        .unwrap();
    let status = sh
        .submit_stats_shares(bearer_request(StatsShareSubmission::default(), token))
        .await
        .err()
        .unwrap_or_else(|| panic!("shareholder accepted a {case} token"));
    assert_eq!(
        status.code(),
        tonic::Code::Unauthenticated,
        "shareholder must reject a {case} token as Unauthenticated (got {status:?})"
    );
}

/// The full JWT rejection matrix, exercised end-to-end through the interceptor
/// on both client-facing services (the aggregator's `EnrollSession` and the
/// shareholder's `SubmitStatsShares`). Each bad token must be refused with
/// `Unauthenticated` before the handler runs; a correctly-signed token must
/// get past auth on both. This pins the interceptor + `Verifier::verify`
/// signature/issuer/audience/alg/kid checks against a regression that
/// weakened any single one of them.
#[tokio::test]
async fn client_facing_services_enforce_jwt_rejection_matrix() {
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(minimal_agg_cfg()).await.unwrap();
    let RunningShareholder {
        client_addr: sh_addr,
        handle: sh_handle,
        ..
    } = serve_shareholder(minimal_sh_cfg()).await.unwrap();
    let agg_url = format!("http://{agg_addr}");
    let sh_url = format!("http://{sh_addr}");

    let bad_tokens = [
        // Known kid=test-1, but signed with the OTHER keypair (jwks_key.pem):
        // the signature cannot verify against test-1's public key.
        (
            "bad-signature",
            mint(ISS, AUD, KID, "device-42", 300, PRIV_B).unwrap(),
        ),
        (
            "wrong-issuer",
            mint(
                "https://other-issuer.local",
                AUD,
                KID,
                "device-42",
                300,
                PRIV,
            )
            .unwrap(),
        ),
        (
            "wrong-audience",
            mint(ISS, "someone-else", KID, "device-42", 300, PRIV).unwrap(),
        ),
        ("expired", mint_with_exp_offset(-600)),
        ("missing-kid", tok_missing_kid()),
        ("non-rs256-alg", tok_hs256()),
    ];
    for (case, token) in &bad_tokens {
        assert_aggregator_rejects(&agg_url, token, case).await;
        assert_shareholder_rejects(&sh_url, token, case).await;
    }

    // Positive control: a correctly-signed token must not be refused by auth,
    // so the matrix above pins genuine rejection rather than a blanket deny.
    // This cluster hosts no session, so an empty-selector enroll is refused on
    // its own merits (FailedPrecondition), but never as Unauthenticated,
    // which is the property this control fixes. `ListSessions` needs no
    // session and so passes outright.
    let good = mint(ISS, AUD, KID, "device-42", 300, PRIV).unwrap();
    let mut agg = AggregatorServiceClient::connect(agg_url.clone())
        .await
        .unwrap();
    let status = agg
        .enroll_session(bearer_request(
            EnrollRequest {
                session_id: String::new(),
            },
            &good,
        ))
        .await
        .expect_err("this cluster hosts no session, so an empty-selector enroll is refused");
    assert_ne!(
        status.code(),
        tonic::Code::Unauthenticated,
        "a correctly-signed token must pass aggregator auth (got {status:?})"
    );
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert!(
        status.message().contains("create one on the admin plane"),
        "the zero-session error must tell the operator to create one, not report a \
         misleading session count (got {status:?})"
    );
    agg.list_sessions(bearer_request(ListSessionsRequest {}, &good))
        .await
        .expect("aggregator must accept a correctly-signed token via ListSessions");

    // The shareholder rejects the empty submission on its own merits, but only
    // AFTER auth: a valid token must never surface as Unauthenticated.
    let mut sh = ShareholderServiceClient::connect(sh_url.clone())
        .await
        .unwrap();
    let status = sh
        .submit_stats_shares(bearer_request(StatsShareSubmission::default(), &good))
        .await
        .expect_err("an empty stats submission is rejected on its own merits");
    assert_ne!(
        status.code(),
        tonic::Code::Unauthenticated,
        "a correctly-signed token must pass shareholder auth (got {status:?})"
    );

    agg_handle.shutdown();
    sh_handle.shutdown();
}
