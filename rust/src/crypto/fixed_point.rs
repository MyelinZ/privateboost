use curve25519_dalek::scalar::Scalar;

/// Convert an f64 value to a Scalar using fixed-point arithmetic.
///
/// Scales the value, rounds to the nearest integer, and represents it
/// in the ristretto255 scalar field. Negative values use field negation
/// (Scalar::ZERO - positive).
pub fn encode(value: f64, scale: f64) -> Scalar {
    let scaled = (value * scale).round();

    if scaled == 0.0 {
        return Scalar::ZERO;
    }

    let abs_scaled = scaled.abs() as u128;
    let pos = scalar_from_u128(abs_scaled);

    if scaled < 0.0 {
        Scalar::ZERO - pos
    } else {
        pos
    }
}

/// Convert a Scalar back to f64 using fixed-point arithmetic.
///
/// Determines sign by checking if the scalar is in the upper half of the
/// field (> ℓ/2). The field order ℓ ≈ 2^252, so ℓ/2 ≈ 2^251.
/// Bit 251 corresponds to bit 3 of byte 31 in LE representation.
pub fn decode(scalar: Scalar, scale: f64) -> f64 {
    let bytes = scalar.to_bytes();

    // Check if the value is "negative" (in upper half of field).
    // ℓ ≈ 2^252, so ℓ/2 ≈ 2^251. Bit 251 = bit 3 of byte 31.
    let is_negative = bytes[31] >= 0x08;

    if is_negative {
        let negated = Scalar::ZERO - scalar;
        let magnitude = scalar_to_f64(&negated);
        -magnitude / scale
    } else {
        scalar_to_f64(&scalar) / scale
    }
}

/// Encode a slice of f64 values to Scalars.
pub fn encode_vec(values: &[f64], scale: f64) -> Vec<Scalar> {
    values.iter().map(|&v| encode(v, scale)).collect()
}

/// Decode a slice of Scalars back to f64 values.
pub fn decode_vec(scalars: &[Scalar], scale: f64) -> Vec<f64> {
    scalars.iter().map(|&s| decode(s, scale)).collect()
}

/// Convert a u128 to a Scalar via a 32-byte LE array.
fn scalar_from_u128(val: u128) -> Scalar {
    let mut bytes = [0u8; 32];
    let val_bytes = val.to_le_bytes();
    bytes[..16].copy_from_slice(&val_bytes);
    let ct = Scalar::from_canonical_bytes(bytes);
    let opt: Option<Scalar> = ct.into();
    opt.expect("u128 always fits in scalar field (2^128 < ℓ ≈ 2^252)")
}

/// Convert a Scalar to f64 by interpreting its LE bytes.
///
/// Accumulates from MSB to LSB (Horner's method) for better numerical
/// precision — adding large terms first minimizes rounding error.
/// Lossy for values > 2^53, which is acceptable since we divide
/// by scale after conversion.
fn scalar_to_f64(s: &Scalar) -> f64 {
    let bytes = s.to_bytes();
    let mut result: f64 = 0.0;
    for &b in bytes.iter().rev() {
        result = result * 256.0 + b as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_positive() {
        let scale = 1e12;
        let value = 356.789;
        let encoded = encode(value, scale);
        let decoded = decode(encoded, scale);
        assert!((decoded - value).abs() < 1e-9, "got {decoded}");
    }

    #[test]
    fn test_roundtrip_negative() {
        let scale = 1e12;
        let value = -42.5;
        let encoded = encode(value, scale);
        let decoded = decode(encoded, scale);
        assert!((decoded - value).abs() < 1e-9, "got {decoded}");
    }

    #[test]
    fn test_roundtrip_zero() {
        let scale = 1e12;
        let encoded = encode(0.0, scale);
        assert_eq!(encoded, Scalar::ZERO);
        assert_eq!(decode(encoded, scale), 0.0);
    }

    #[test]
    fn test_sum_preserving() {
        let scale = 1e12;
        let values = vec![10.5, -3.2, 7.1, 100.0];
        let encoded: Vec<Scalar> = values.iter().map(|&v| encode(v, scale)).collect();
        let field_sum: Scalar = encoded.iter().copied().reduce(|a, b| a + b).unwrap();
        let decoded_sum = decode(field_sum, scale);
        let expected_sum: f64 = values.iter().sum();
        assert!((decoded_sum - expected_sum).abs() < 1e-9);
    }

    #[test]
    fn test_large_value() {
        let scale = 1e12;
        let value = 360000.0;
        let encoded = encode(value, scale);
        let decoded = decode(encoded, scale);
        assert!((decoded - value).abs() < 1e-6, "got {decoded}");
    }

    #[test]
    fn test_shamir_with_fixed_point() {
        use crate::crypto::shamir;
        let scale = 1e12;
        let values = vec![42.5, -17.3, 600.0];
        let encoded = encode_vec(&values, scale);
        let shares = shamir::share(&encoded, 3, 2).unwrap();
        let result = shamir::reconstruct(&shares[..2], 2).unwrap();
        let decoded = decode_vec(&result, scale);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-9, "expected {a}, got {b}");
        }
    }

    /// Simulate multiple clients sharing values, shareholders summing,
    /// and aggregator reconstructing the sum.
    #[test]
    fn test_multi_client_sum_pipeline() {
        use crate::crypto::shamir;

        let scale = 1e12;
        let n_shareholders = 3;
        let threshold = 2;

        // Each client has a different value
        let client_values: Vec<Vec<f64>> = vec![
            vec![10.0, -5.0],
            vec![20.0, 3.0],
            vec![-15.0, 8.0],
            vec![100.0, -50.0],
            vec![0.5, 1.5],
        ];

        // Each client encodes and creates shares
        let mut sh_sums: Vec<Vec<Scalar>> = vec![Vec::new(); n_shareholders];

        for values in &client_values {
            let encoded = encode_vec(values, scale);
            let shares = shamir::share(&encoded, n_shareholders, threshold).unwrap();
            for (i, share) in shares.iter().enumerate() {
                if sh_sums[i].is_empty() {
                    sh_sums[i] = share.y.clone();
                } else {
                    for (sum, val) in sh_sums[i].iter_mut().zip(share.y.iter()) {
                        *sum += val;
                    }
                }
            }
        }

        // Aggregator collects sum-shares and reconstructs
        let sum_shares: Vec<shamir::Share> = sh_sums
            .into_iter()
            .enumerate()
            .map(|(i, y)| shamir::Share { x: (i + 1) as i32, y })
            .collect();
        let reconstructed = shamir::reconstruct(&sum_shares[..threshold], threshold).unwrap();
        let decoded = decode_vec(&reconstructed, scale);

        // Expected: element-wise sum of all client values
        let expected: Vec<f64> = (0..2)
            .map(|j| client_values.iter().map(|v| v[j]).sum::<f64>())
            .collect();

        for (a, b) in expected.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
    }
}
