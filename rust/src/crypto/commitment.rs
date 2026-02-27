use sha2::{Digest, Sha256};

pub type Commitment = [u8; 32];

const DOMAIN_TAG: &[u8] = b"privateboost-commitment-v1\0";

/// Compute a commitment that binds to the round, client identity, nonce, AND share data.
///
/// commitment = SHA256(domain_tag || u64_be(round_id) || u32_be(len(client_id)) || client_id || nonce || share_hash)
///
/// The share_hash binds the commitment to the actual share values, preventing
/// share swapping attacks.
pub fn compute_commitment(
    round_id: i32,
    client_id: &str,
    nonce: &[u8],
    share_data: &[u8],
) -> Commitment {
    // Hash share data separately
    let share_hash = {
        let mut h = Sha256::new();
        h.update(share_data);
        h.finalize()
    };

    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update((round_id as i64).to_be_bytes());
    hasher.update((client_id.len() as u32).to_be_bytes());
    hasher.update(client_id.as_bytes());
    hasher.update(nonce);
    hasher.update(share_hash);
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
        let share_data = b"test_share";
        let c1 = compute_commitment(0, "client_1", &nonce, share_data);
        let c2 = compute_commitment(0, "client_1", &nonce, share_data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commitment_round_separation() {
        let nonce = [0u8; 32];
        let share_data = b"test_share";
        let c1 = compute_commitment(0, "client_1", &nonce, share_data);
        let c2 = compute_commitment(1, "client_1", &nonce, share_data);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_commitment_share_binding() {
        let nonce = [0u8; 32];
        let c1 = compute_commitment(0, "client_1", &nonce, b"share_a");
        let c2 = compute_commitment(0, "client_1", &nonce, b"share_b");
        assert_ne!(c1, c2, "different share data must produce different commitments");
    }

    #[test]
    fn test_commitment_domain_separation() {
        let nonce = [0u8; 32];
        let share_data = b"share";
        let c1 = compute_commitment(0, "client_1", &nonce, share_data);
        let c2 = compute_commitment(0, "client_", &nonce, share_data);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_nonce_uniqueness() {
        let n1 = generate_nonce();
        let n2 = generate_nonce();
        assert_ne!(n1, n2);
    }
}
