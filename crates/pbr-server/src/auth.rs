use arc_swap::ArcSwap;
use jsonwebtoken::jwk::JwkSet;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tonic::{Request, Status};

/// How often the background task re-fetches the Firebase key set. Google
/// rotates `securetoken` keys roughly daily, so an hour stays well ahead of a
/// rotation without hammering the endpoint.
pub const GOOGLE_JWKS_REFRESH_INTERVAL: Duration = Duration::from_secs(3600);

#[derive(Clone, Debug)]
pub struct Identity {
    pub issuer: String,
    pub subject: String,
}

#[derive(Deserialize)]
struct Claims {
    iss: String,
    sub: String,
}

/// A signing key the [`Verifier`] trusts: the JWT `kid` and its RSA public
/// key as SPKI PEM (never a certificate or a private key).
pub struct VerifierKey {
    pub kid: String,
    pub pem: Vec<u8>,
}

pub struct Verifier {
    issuer: String,
    audience: String,
    keys: ArcSwap<HashMap<String, DecodingKey>>,
}

impl Verifier {
    /// Build a `Verifier` from a fixed `(kid, public-key PEM)` set, for issuers
    /// whose keys never rotate. A rotating issuer swaps keys in at runtime via
    /// `update_keys` or `refresh_from_jwks_url`; `verify` is identical.
    pub fn from_static(
        issuer: String,
        audience: String,
        keys: Vec<VerifierKey>,
    ) -> anyhow::Result<Self> {
        let map = build_key_map(keys)?;
        Ok(Self {
            issuer,
            audience,
            keys: ArcSwap::from_pointee(map),
        })
    }

    /// Atomically replace the kid to key map. Readers in `verify` never block
    /// and never see a partial update.
    ///
    /// A refresh yielding zero usable keys is refused, keeping the previous set
    /// rather than storing an empty map that would reject every token. So a
    /// background refresh returning nothing is a logged non-event, while a
    /// startup fetch that parses to nothing fails `serve()` outright.
    pub fn update_keys(&self, keys: Vec<VerifierKey>) -> anyhow::Result<()> {
        self.store(build_key_map(keys)?)
    }

    /// Parse a JWKS document (RFC 7517) and swap the keys in, with the same
    /// swap-or-refuse semantics as [`update_keys`]. Google publishes the
    /// Firebase `securetoken` keys as one at
    /// <https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com>.
    /// An unusable entry (no `kid`, or a key `DecodingKey::from_jwk` rejects)
    /// is logged and skipped rather than aborting the batch, so one bad entry
    /// never disables verification for the keys that did parse.
    ///
    /// [`update_keys`]: Self::update_keys
    pub fn update_keys_from_jwks(&self, jwks_json: &[u8]) -> anyhow::Result<()> {
        let set: JwkSet = serde_json::from_slice(jwks_json)?;
        let map = set
            .keys
            .into_iter()
            .filter_map(|jwk| {
                let Some(kid) = jwk.common.key_id.clone() else {
                    tracing::warn!("skipping JWKS entry without a kid");
                    return None;
                };
                match DecodingKey::from_jwk(&jwk) {
                    Ok(key) => Some((kid, key)),
                    Err(e) => {
                        tracing::warn!(kid = %kid, error = %e, "skipping unusable JWKS entry");
                        None
                    }
                }
            })
            .collect();
        self.store(map)
    }

    /// Fetch a JWKS URL (Firebase's `securetoken` keys, for this deployment)
    /// and swap the keys in via [`update_keys_from_jwks`].
    ///
    /// [`update_keys_from_jwks`]: Self::update_keys_from_jwks
    pub async fn refresh_from_jwks_url(&self, url: &str) -> anyhow::Result<()> {
        let body = reqwest::get(url).await?.error_for_status()?.bytes().await?;
        self.update_keys_from_jwks(&body)
    }

    fn store(&self, map: HashMap<String, DecodingKey>) -> anyhow::Result<()> {
        anyhow::ensure!(
            !map.is_empty(),
            "key refresh yielded zero usable keys; keeping the previous key set"
        );
        self.keys.store(Arc::new(map));
        Ok(())
    }

    #[allow(clippy::result_large_err)]
    pub fn verify(&self, token: &str) -> Result<Identity, Status> {
        let unauthenticated = |m: &str| Status::unauthenticated(m.to_string());
        let header = decode_header(token).map_err(|_| unauthenticated("bad header"))?;
        if header.alg != Algorithm::RS256 {
            return Err(unauthenticated("alg must be RS256"));
        }
        let kid = header.kid.ok_or_else(|| unauthenticated("missing kid"))?;
        let keys = self.keys.load();
        let key = keys
            .get(&kid)
            .ok_or_else(|| unauthenticated("unknown kid"))?;

        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_audience(&[&self.audience]);
        validation.set_issuer(&[&self.issuer]);
        validation.set_required_spec_claims(&["exp", "aud", "iss", "sub"]);
        validation.leeway = 60;

        let data = decode::<Claims>(token, key, &validation)
            .map_err(|_| unauthenticated("invalid token"))?;
        Ok(Identity {
            issuer: data.claims.iss,
            subject: data.claims.sub,
        })
    }
}

fn build_key_map(keys: Vec<VerifierKey>) -> anyhow::Result<HashMap<String, DecodingKey>> {
    let mut map = HashMap::new();
    for VerifierKey { kid, pem } in keys {
        map.insert(kid, DecodingKey::from_rsa_pem(&pem)?);
    }
    Ok(map)
}

/// Tonic interceptor: requires `authorization: Bearer <jwt>`, inserts the
/// verified Identity into request extensions. Fail-closed.
#[allow(clippy::result_large_err)]
pub fn interceptor(
    verifier: Arc<Verifier>,
) -> impl Fn(Request<()>) -> Result<Request<()>, Status> + Clone {
    move |mut req: Request<()>| {
        let header = req
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("missing authorization header"))?;
        let token = header
            .strip_prefix("Bearer ")
            .ok_or_else(|| Status::unauthenticated("expected Bearer token"))?;
        let identity = verifier.verify(token)?;
        req.extensions_mut().insert(identity);
        Ok(req)
    }
}
