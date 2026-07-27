//! Admin-plane authentication.
//!
//! The admin plane is guarded by a single static bearer token from the
//! aggregator's config, separate from the device identity provider: device
//! tokens must never reach it, and an operator must be able to schedule a
//! session without holding a device account. The token is a
//! secret: it is never logged, and a rejection never echoes the presented
//! value.

use tonic::{Request, Status};

/// Constant-time byte comparison: content is compared in constant time, so a
/// rejection's timing does not leak how much of a same-length token was
/// correct. The length check below is a fast path and is not constant-time,
/// a rejection can leak whether the presented token's length matches the
/// configured one. Token length is treated as non-secret; this matches
/// standard practice for bearer-token comparison.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Rejects every request whose `authorization` header is not exactly
/// `Bearer <configured token>`.
#[allow(clippy::result_large_err)]
pub fn interceptor(token: String) -> impl Fn(Request<()>) -> Result<Request<()>, Status> + Clone {
    move |req: Request<()>| {
        let presented = req
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| Status::unauthenticated("admin bearer token required"))?;
        if ct_eq(presented.as_bytes(), token.as_bytes()) {
            Ok(req)
        } else {
            Err(Status::unauthenticated("invalid admin token"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_matches_equal_byte_strings() {
        assert!(ct_eq(b"secret-token", b"secret-token"));
    }

    #[test]
    fn ct_eq_rejects_same_length_difference() {
        assert!(!ct_eq(b"secret-token", b"secret-tokeN"));
    }

    #[test]
    fn ct_eq_rejects_different_length() {
        assert!(!ct_eq(b"secret-token", b"secret-token-longer"));
        assert!(!ct_eq(b"short", b""));
    }
}
