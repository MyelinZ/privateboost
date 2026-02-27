use sha2::{Digest, Sha256};

pub type Commitment = [u8; 32];

/// SHA256(round_id_be_8bytes || client_id_utf8 || nonce)
///
/// IMPORTANT: round_id is serialized as 8-byte big-endian i64 to match
/// the Python implementation which uses `round_id.to_bytes(8, "big")`.
pub fn compute_commitment(round_id: i32, client_id: &str, nonce: &[u8]) -> Commitment {
    let mut hasher = Sha256::new();
    hasher.update((round_id as i64).to_be_bytes());
    hasher.update(client_id.as_bytes());
    hasher.update(nonce);
    hasher.finalize().into()
}

/// Generate 32 random bytes for use as a nonce.
pub fn generate_nonce() -> [u8; 32] {
    rand::random()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commitment_consistency() {
        let nonce = [0u8; 32];
        let c1 = compute_commitment(0, "client_1", &nonce);
        let c2 = compute_commitment(0, "client_1", &nonce);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commitment_round_separation() {
        let nonce = [0u8; 32];
        let c1 = compute_commitment(0, "client_1", &nonce);
        let c2 = compute_commitment(1, "client_1", &nonce);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_nonce_uniqueness() {
        let n1 = generate_nonce();
        let n2 = generate_nonce();
        assert_ne!(n1, n2);
    }
}
