use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
struct Claims {
    iss: String,
    aud: String,
    sub: String,
    exp: i64,
    iat: i64,
}

/// Mint an RS256 JWT, for CLI drivers, tests and self-hosted issuers. In
/// production the token comes from Firebase Auth.
pub fn mint(
    issuer: &str,
    audience: &str,
    kid: &str,
    sub: &str,
    ttl_secs: u64,
    rsa_private_pem: &[u8],
) -> anyhow::Result<String> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.to_string());
    let claims = Claims {
        iss: issuer.to_string(),
        aud: audience.to_string(),
        sub: sub.to_string(),
        exp: now + ttl_secs as i64,
        iat: now,
    };
    Ok(encode(
        &header,
        &claims,
        &EncodingKey::from_rsa_pem(rsa_private_pem)?,
    )?)
}
