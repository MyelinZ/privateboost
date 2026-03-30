use rand::{TryRngCore, rngs::OsRng};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Commitment(pub [u8; 32]);

impl std::fmt::Display for Commitment {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for b in &self.0[..8] {
            write!(f, "{:02x}", b)?;
        }
        write!(f, "...")
    }
}

pub fn commit(round_id: u64, client_id: &str, nonce: &[u8; 32]) -> Commitment {
    let mut hasher = Sha256::new();
    hasher.update(round_id.to_be_bytes());
    hasher.update(client_id.as_bytes());
    hasher.update(nonce);
    let result = hasher.finalize();
    Commitment(result.into())
}

pub fn generate_nonce() -> [u8; 32] {
    let mut nonce = [0u8; 32];
    OsRng.try_fill_bytes(&mut nonce).expect("OsRng failed");
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commitment_consistency() {
        let nonce = [b'x'; 32];
        let c1 = commit(0, "client_1", &nonce);
        let c2 = commit(0, "client_1", &nonce);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commitment_different_nonce() {
        let c1 = commit(0, "client_1", &[b'x'; 32]);
        let c2 = commit(0, "client_1", &[b'y'; 32]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_commitment_round_separation() {
        let nonce = [b'z'; 32];
        let c0 = commit(0, "client_1", &nonce);
        let c1 = commit(1, "client_1", &nonce);
        assert_ne!(c0, c1);
    }

    #[test]
    fn test_commitment_different_client() {
        let nonce = [b'a'; 32];
        let c1 = commit(0, "client_1", &nonce);
        let c2 = commit(0, "client_2", &nonce);
        assert_ne!(c1, c2);
    }
}
