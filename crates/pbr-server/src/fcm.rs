//! FCM HTTP v1 sender.
//!
//! Mints a Google OAuth token via `gcloud-sdk`'s `GoogleAuthTokenGenerator`,
//! which auto-discovers Application Default Credentials (the authorized_user
//! file this repo's demo runs under,
//! `~/.config/gcloud/application_default_credentials.json`, or the GCE/Cloud
//! Run metadata server) or, when a service-account key file is configured,
//! loads and signs with that file directly. Either way
//! `GoogleAuthTokenGenerator` caches its token internally until it is near
//! expiry, so `FcmSender` does not need its own cache.
//!
//! POSTs FCM HTTP v1 data-only messages, stamped with a TTL and collapse key
//! so an undelivered round-open push dies once the next eligible push would
//! supersede it anyway; a non-2xx response (e.g. a dead device token)
//! becomes an `Err` that callers log and skip, never fatal to the round loop.

use gcloud_sdk::{GoogleAuthTokenGenerator, TokenSourceType};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// OAuth scope required to send FCM messages.
const FCM_SCOPE: &str = "https://www.googleapis.com/auth/firebase.messaging";
/// Whole-request timeout for an FCM send. Bounds a hung fcm.googleapis.com
/// connection so a spawned notify task (and its socket) cannot park
/// indefinitely; push is best-effort and off the round loop's critical path.
const FCM_TIMEOUT: Duration = Duration::from_secs(30);

/// Collapse key shared by every round-open push: FCM/APNs keep at most one
/// undelivered message per device under this key, replacing older with
/// newer, so a device offline through several rounds wakes to one push.
const COLLAPSE_KEY: &str = "round_open";

/// Bearer-token mint for the FCM endpoint. `Google` is production (ADC or a
/// service-account key via `gcloud-sdk`); `Static` is the test seam, a
/// fixed token for a stub endpoint, never valid against fcm.googleapis.com.
enum TokenProvider {
    Google(GoogleAuthTokenGenerator),
    #[cfg(test)]
    Static(String),
}

impl TokenProvider {
    async fn header_value(&self) -> anyhow::Result<String> {
        match self {
            TokenProvider::Google(g) => Ok(g.create_token().await?.header_value()),
            #[cfg(test)]
            TokenProvider::Static(t) => Ok(format!("Bearer {t}")),
        }
    }
}

pub struct FcmSender {
    project_id: String,
    endpoint_base: String,
    /// Message lifetime: equal to the notify floor, so an undelivered push
    /// dies once the next eligible push would supersede it anyway.
    ttl: Duration,
    provider: TokenProvider,
    client: reqwest::Client,
}

impl FcmSender {
    /// `service_account_path = None` uses Application Default Credentials
    /// (`TokenSourceType::Default`); `Some(path)` loads and signs with that
    /// service-account JSON key file directly (`TokenSourceType::File`).
    /// Either token source works with the same `send_data` call.
    pub async fn from_config(
        project_id: String,
        service_account_path: Option<String>,
        ttl: Duration,
    ) -> anyhow::Result<Self> {
        let source_type = match service_account_path {
            Some(path) => TokenSourceType::File(path.into()),
            None => TokenSourceType::Default,
        };
        let provider =
            GoogleAuthTokenGenerator::new(source_type, vec![FCM_SCOPE.to_string()]).await?;
        Ok(Self {
            project_id,
            endpoint_base: "https://fcm.googleapis.com".to_string(),
            ttl,
            provider: TokenProvider::Google(provider),
            client: reqwest::Client::builder().timeout(FCM_TIMEOUT).build()?,
        })
    }

    /// Test constructor: fixed bearer token against a stub endpoint.
    #[cfg(test)]
    pub(crate) fn for_tests(
        project_id: &str,
        endpoint_base: &str,
        token: &str,
        ttl: Duration,
    ) -> Self {
        Self {
            project_id: project_id.to_string(),
            endpoint_base: endpoint_base.to_string(),
            ttl,
            provider: TokenProvider::Static(token.to_string()),
            client: reqwest::Client::builder()
                .timeout(FCM_TIMEOUT)
                .build()
                .expect("reqwest client"),
        }
    }

    /// POST a data-only message to `fcm_token`. A dead/invalid token (or any
    /// other non-2xx FCM response) yields `Err`; the caller treats a single
    /// failure as non-fatal (log + skip), never aborting the round loop.
    pub async fn send_data(
        &self,
        fcm_token: &str,
        data: HashMap<String, String>,
    ) -> anyhow::Result<()> {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let resp = self
            .post(&build_body(
                fcm_token,
                data,
                self.ttl.as_secs(),
                now_secs,
                false,
            ))
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("FCM send failed ({status}): {body}");
        }
        Ok(())
    }

    /// Test-only: send with `validate_only: true` (an FCM dry-run, never
    /// delivers) and return the raw status, so the live auth-check test can
    /// assert 400 INVALID_ARGUMENT for a dummy token without a real device.
    #[cfg(test)]
    async fn send_validate_only(&self, fcm_token: &str) -> anyhow::Result<reqwest::StatusCode> {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let resp = self
            .post(&build_body(
                fcm_token,
                HashMap::new(),
                self.ttl.as_secs(),
                now_secs,
                true,
            ))
            .await?;
        Ok(resp.status())
    }

    async fn post<B: serde::Serialize>(&self, body: &B) -> anyhow::Result<reqwest::Response> {
        let token = self.provider.header_value().await?;
        let url = format!(
            "{}/v1/projects/{}/messages:send",
            self.endpoint_base, self.project_id
        );
        Ok(self
            .client
            .post(url)
            .header(reqwest::header::AUTHORIZATION, token)
            .json(body)
            .send()
            .await?)
    }
}

/// The FCM HTTP v1 `messages:send` request body. Typed rather than an ad-hoc
/// JSON blob so the compiler enforces the wire shape: a mistyped or mis-nested
/// key (e.g. `content_available` for `content-available`) would still serialize
/// but silently stop waking iOS, and that failure is invisible until a device
/// does not wake.
#[derive(serde::Serialize)]
struct FcmRequest {
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    validate_only: bool,
    message: FcmMessage,
}

/// A data-only message that wakes a backgrounded app on both platforms: Android
/// via high-priority delivery, iOS via the APNs `background` push type with
/// `content-available` (without which the OS drops a data-only message to a
/// backgrounded app instead of waking it).
#[derive(serde::Serialize)]
struct FcmMessage {
    token: String,
    android: Android,
    apns: Apns,
    data: HashMap<String, String>,
}

#[derive(serde::Serialize)]
struct Android {
    priority: &'static str,
    collapse_key: &'static str,
    /// Seconds with an "s" suffix, e.g. "900s" (the FCM v1 duration format).
    ttl: String,
}

#[derive(serde::Serialize)]
struct Apns {
    headers: ApnsHeaders,
    payload: ApnsPayload,
}

#[derive(serde::Serialize)]
struct ApnsHeaders {
    #[serde(rename = "apns-push-type")]
    push_type: &'static str,
    #[serde(rename = "apns-priority")]
    priority: &'static str,
    #[serde(rename = "apns-collapse-id")]
    collapse_id: &'static str,
    /// Epoch seconds at which APNs discards the undelivered push.
    #[serde(rename = "apns-expiration")]
    expiration: String,
}

#[derive(serde::Serialize)]
struct ApnsPayload {
    aps: Aps,
}

#[derive(serde::Serialize)]
struct Aps {
    #[serde(rename = "content-available")]
    content_available: u8,
}

/// Build the FCM request body for a `round_open` wake. Pure (no I/O) so the
/// serialized shape can be unit-tested without a live send.
fn build_body(
    fcm_token: &str,
    data: HashMap<String, String>,
    ttl_secs: u64,
    now_epoch_secs: u64,
    validate_only: bool,
) -> FcmRequest {
    FcmRequest {
        validate_only,
        message: FcmMessage {
            token: fcm_token.to_string(),
            android: Android {
                priority: "high",
                collapse_key: COLLAPSE_KEY,
                ttl: format!("{ttl_secs}s"),
            },
            apns: Apns {
                headers: ApnsHeaders {
                    push_type: "background",
                    priority: "5",
                    collapse_id: COLLAPSE_KEY,
                    expiration: (now_epoch_secs + ttl_secs).to_string(),
                },
                payload: ApnsPayload {
                    aps: Aps {
                        content_available: 1,
                    },
                },
            },
            data,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcm_message_body_is_valid() {
        let mut data = HashMap::new();
        data.insert("kind".to_string(), "round_open".to_string());
        data.insert("sentAt".to_string(), "1700000000123".to_string());

        let body = serde_json::to_value(build_body(
            "device-token-abc",
            data,
            900,
            1_700_000_000,
            false,
        ))
        .unwrap();
        assert_eq!(body["message"]["token"], "device-token-abc");
        assert_eq!(body["message"]["android"]["priority"], "high");
        // One queued round-open per device, and it expires once superseded.
        assert_eq!(body["message"]["android"]["collapse_key"], "round_open");
        assert_eq!(body["message"]["android"]["ttl"], "900s");
        // iOS silent-push wake: background push type + content-available.
        assert_eq!(
            body["message"]["apns"]["headers"]["apns-push-type"],
            "background"
        );
        assert_eq!(
            body["message"]["apns"]["headers"]["apns-collapse-id"],
            "round_open"
        );
        assert_eq!(
            body["message"]["apns"]["headers"]["apns-expiration"],
            "1700000900"
        );
        assert_eq!(
            body["message"]["apns"]["payload"]["aps"]["content-available"],
            1
        );
        assert_eq!(body["message"]["data"]["kind"], "round_open");
        assert!(
            body.get("validate_only").is_none(),
            "a real send must not carry validate_only"
        );
    }

    #[test]
    fn validate_only_body_sets_the_flag() {
        let body = serde_json::to_value(build_body(
            "device-token-abc",
            HashMap::new(),
            900,
            1_700_000_000,
            true,
        ))
        .unwrap();
        assert_eq!(body["validate_only"], true);
        assert_eq!(body["message"]["token"], "device-token-abc");
    }

    /// Live auth check: using ADC (this machine's gcloud
    /// `application_default_credentials.json`, authorized_user, quota
    /// project pboost-test-12345), mint a real token and POST a
    /// `validate_only` message with a dummy device token. FCM rejects the
    /// token (400 INVALID_ARGUMENT) but only AFTER accepting the bearer
    /// token, so a 400 here proves the ADC token source works end to end.
    /// `#[ignore]`d: needs live network + this machine's ADC file; run
    /// manually with `cargo test -p pbr-server --lib fcm::tests::live_fcm_validate_only -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn live_fcm_validate_only() {
        let sender = FcmSender::from_config(
            "pboost-test-12345".to_string(),
            None,
            Duration::from_secs(900),
        )
        .await
        .expect("ADC token source must be available on this machine");
        let status = sender
            .send_validate_only("DUMMY")
            .await
            .expect("request must complete (FCM rejects the token, not the connection)");
        assert_eq!(
            status,
            reqwest::StatusCode::BAD_REQUEST,
            "expected 400 INVALID_ARGUMENT for a dummy token with validate_only, proving auth succeeded"
        );
    }
}
