use anyhow::{bail, Result};
use curve25519_dalek::scalar::Scalar;
use rand::RngCore;

#[derive(Debug, Clone)]
pub struct Share {
    pub x: i32,
    pub y: Vec<Scalar>,
}

/// Split values into n Shamir shares with given threshold.
///
/// Uses finite-field arithmetic over the ristretto255 scalar field
/// for information-theoretic security (no floating-point precision loss).
///
/// Note: We implement vector Shamir directly rather than using the `vsss-rs`
/// crate because vsss-rs operates on single scalars. Our protocol needs to
/// share vectors of scalars efficiently with a single polynomial per element.
/// The arithmetic is straightforward (polynomial evaluation + Lagrange
/// interpolation) and uses the audited `curve25519-dalek` Scalar type.
pub fn share(values: &[Scalar], n_parties: usize, threshold: usize) -> Result<Vec<Share>> {
    if threshold > n_parties {
        bail!("threshold {threshold} cannot exceed n_parties {n_parties}");
    }
    let n_values = values.len();
    let mut rng = rand::rng();

    // coeffs: threshold rows x n_values cols
    // Row 0 = the actual values (constant term of polynomial)
    // Rows 1..threshold-1 = uniformly random Scalars
    let mut coeffs = vec![vec![Scalar::ZERO; n_values]; threshold];
    for (i, val) in values.iter().enumerate() {
        coeffs[0][i] = *val;
    }
    for row in coeffs.iter_mut().skip(1) {
        for col in row.iter_mut() {
            let mut wide = [0u8; 64];
            rng.fill_bytes(&mut wide);
            *col = Scalar::from_bytes_mod_order_wide(&wide);
        }
    }

    // Evaluate polynomial at x = 1, 2, ..., n_parties
    let mut shares = Vec::with_capacity(n_parties);
    for i in 0..n_parties {
        let x = (i + 1) as i32;
        let x_scalar = Scalar::from(x as u64);
        let mut y = vec![Scalar::ZERO; n_values];
        for (val_idx, y_val) in y.iter_mut().enumerate() {
            let mut sum = Scalar::ZERO;
            let mut x_power = Scalar::ONE;
            for coeff_row in coeffs.iter().take(threshold) {
                sum += x_power * coeff_row[val_idx];
                x_power *= x_scalar;
            }
            *y_val = sum;
        }
        shares.push(Share { x, y });
    }

    Ok(shares)
}

/// Reconstruct values from threshold shares using Lagrange interpolation at x=0.
pub fn reconstruct(shares: &[Share], threshold: usize) -> Result<Vec<Scalar>> {
    if shares.len() < threshold {
        bail!(
            "need at least {threshold} shares, got {}",
            shares.len()
        );
    }
    let shares = &shares[..threshold];
    for s in shares {
        if s.x <= 0 {
            bail!("share x-coordinate must be positive, got {}", s.x);
        }
    }
    let n_values = shares[0].y.len();

    // Lagrange coefficients at x=0:
    // L_j(0) = prod_{i!=j} (-x_i / (x_j - x_i))
    let x_scalars: Vec<Scalar> = shares
        .iter()
        .map(|s| Scalar::from(s.x as u64))
        .collect();
    let n = x_scalars.len();

    let mut lagrange_coeffs = vec![Scalar::ONE; n];
    for j in 0..n {
        for i in 0..n {
            if i != j {
                let neg_xi = Scalar::ZERO - x_scalars[i];
                let denom = x_scalars[j] - x_scalars[i];
                lagrange_coeffs[j] *= neg_xi * denom.invert();
            }
        }
    }

    // result = sum_j lagrange_coeffs[j] * shares[j].y
    let mut result = vec![Scalar::ZERO; n_values];
    for (j, share) in shares.iter().enumerate() {
        for (v, r) in result.iter_mut().enumerate() {
            *r += lagrange_coeffs[j] * share.y[v];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_from_i64(val: i64) -> Scalar {
        if val >= 0 {
            Scalar::from(val as u64)
        } else {
            Scalar::ZERO - Scalar::from((-val) as u64)
        }
    }

    #[test]
    fn test_shamir_reconstruction() {
        let values = vec![
            Scalar::from(42u64),
            scalar_from_i64(-17),
            Scalar::from(100u64),
        ];
        let shares = share(&values, 3, 2).unwrap();
        assert_eq!(shares.len(), 3);
        let result = reconstruct(&shares[..2], 2).unwrap();
        assert_eq!(result[0], values[0]);
        assert_eq!(result[1], values[1]);
        assert_eq!(result[2], values[2]);
    }

    #[test]
    fn test_shamir_linearity() {
        let v1 = vec![Scalar::from(10u64), Scalar::from(20u64)];
        let v2 = vec![Scalar::from(30u64), Scalar::from(40u64)];
        let s1 = share(&v1, 3, 2).unwrap();
        let s2 = share(&v2, 3, 2).unwrap();
        // Sum shares element-wise per party
        let summed: Vec<Share> = s1
            .iter()
            .zip(s2.iter())
            .map(|(a, b)| Share {
                x: a.x,
                y: a.y.iter().zip(b.y.iter()).map(|(x, y)| x + y).collect(),
            })
            .collect();
        let result = reconstruct(&summed[..2], 2).unwrap();
        assert_eq!(result[0], Scalar::from(40u64));
        assert_eq!(result[1], Scalar::from(60u64));
    }

    #[test]
    fn test_3_of_5_threshold() {
        let values = vec![Scalar::from(99u64)];
        let shares = share(&values, 5, 3).unwrap();
        assert_eq!(shares.len(), 5);
        // Exact equality — no precision loss with finite-field arithmetic
        let result = reconstruct(&shares[1..4], 3).unwrap();
        assert_eq!(result[0], Scalar::from(99u64));
    }

    #[test]
    fn test_negative_values() {
        let neg_five = Scalar::ZERO - Scalar::from(5u64);
        let neg_1000 = Scalar::ZERO - Scalar::from(1000u64);
        let values = vec![neg_five, neg_1000];
        let shares = share(&values, 3, 2).unwrap();
        let result = reconstruct(&shares[..2], 2).unwrap();
        assert_eq!(result[0], neg_five);
        assert_eq!(result[1], neg_1000);
    }

    #[test]
    fn test_threshold_exceeds_parties() {
        let values = vec![Scalar::from(1u64)];
        let result = share(&values, 3, 5);
        assert!(result.is_err());
    }
}
