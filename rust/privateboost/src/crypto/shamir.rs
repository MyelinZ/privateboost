use super::field::{F, MersenneField, ZERO};
use crate::{Error, Result};
use rand::Rng;
use rand::distr::StandardUniform;

#[derive(Clone, Debug)]
pub struct Share {
    pub x: u64,
    pub values: Vec<F>,
}

impl Share {
    /// Element-wise addition of share values (homomorphic property).
    /// Both shares must have the same x coordinate (same evaluation point).
    /// Used by shareholders to sum shares from different clients.
    pub fn add(&self, other: &Share) -> Share {
        assert_eq!(
            self.x, other.x,
            "Cannot add shares with different x coordinates"
        );
        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Share { x: self.x, values }
    }
}

pub fn share(
    values: &[F],
    n_parties: usize,
    threshold: usize,
    rng: &mut impl Rng,
) -> Result<Vec<Share>> {
    if threshold > n_parties {
        return Err(Error::ThresholdExceedsParties {
            threshold,
            n_parties,
        });
    }

    let n = values.len();

    // Generate random coefficients for degree 1..threshold-1
    let mut coeffs: Vec<Vec<F>> = Vec::with_capacity(threshold - 1);
    for _ in 0..threshold - 1 {
        let coeff: Vec<F> = (0..n).map(|_| rng.sample(StandardUniform)).collect();
        coeffs.push(coeff);
    }

    // Evaluate polynomial at x = 1, 2, ..., n_parties
    let mut shares = Vec::with_capacity(n_parties);
    for x in 1..=n_parties {
        let x_field = MersenneField::from_u64(x as u64);
        let mut y = values.to_vec();

        let mut x_power = MersenneField::from_u64(x as u64);
        for coeff_row in &coeffs {
            for (yi, ci) in y.iter_mut().zip(coeff_row.iter()) {
                *yi += *ci * x_power;
            }
            x_power *= x_field;
        }

        shares.push(Share {
            x: x as u64,
            values: y,
        });
    }

    Ok(shares)
}

pub fn reconstruct(shares: &[Share], threshold: usize) -> Result<Vec<F>> {
    if shares.len() < threshold {
        return Err(Error::InsufficientShares {
            needed: threshold,
            got: shares.len(),
        });
    }

    let shares = &shares[..threshold];
    let n = shares[0].values.len();
    let x_values: Vec<F> = shares
        .iter()
        .map(|s| MersenneField::from_u64(s.x))
        .collect();

    // Lagrange coefficients at x = 0
    let mut lagrange_coeffs = Vec::with_capacity(threshold);
    for j in 0..threshold {
        let mut num = MersenneField::from_u64(1);
        let mut den = MersenneField::from_u64(1);
        for i in 0..threshold {
            if i != j {
                num *= -x_values[i];
                den *= x_values[j] - x_values[i];
            }
        }
        lagrange_coeffs.push(num * den.inverse().ok_or(Error::FieldInverse)?);
    }

    // Interpolate
    let mut result = vec![ZERO; n];
    for (share, coeff) in shares.iter().zip(lagrange_coeffs.iter()) {
        for (rj, sj) in result.iter_mut().zip(share.values.iter()) {
            *rj += *sj * *coeff;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::encoding::{decode_all, encode_all};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_shamir_reconstruction_2_of_3() {
        let mut rng = StdRng::seed_from_u64(42);
        let values = encode_all(&[42.0, 100.0, -7.5]);
        let shares = share(&values, 3, 2, &mut rng).unwrap();

        let r01 = decode_all(&reconstruct(&[shares[0].clone(), shares[1].clone()], 2).unwrap());
        let r02 = decode_all(&reconstruct(&[shares[0].clone(), shares[2].clone()], 2).unwrap());
        let r12 = decode_all(&reconstruct(&[shares[1].clone(), shares[2].clone()], 2).unwrap());

        let expected = [42.0, 100.0, -7.5];
        for (i, exp) in expected.iter().enumerate() {
            assert!((r01[i] - exp).abs() < 1e-5, "r01[{i}]: {} != {exp}", r01[i]);
            assert!((r02[i] - exp).abs() < 1e-5, "r02[{i}]: {} != {exp}", r02[i]);
            assert!((r12[i] - exp).abs() < 1e-5, "r12[{i}]: {} != {exp}", r12[i]);
        }
    }

    #[test]
    fn test_shamir_linearity() {
        let mut rng = StdRng::seed_from_u64(42);
        let values_a = encode_all(&[10.0, 20.0, 30.0]);
        let values_b = encode_all(&[1.0, 2.0, 3.0]);

        let shares_a = share(&values_a, 3, 2, &mut rng).unwrap();
        let shares_b = share(&values_b, 3, 2, &mut rng).unwrap();

        let summed: Vec<Share> = shares_a
            .iter()
            .zip(shares_b.iter())
            .map(|(a, b)| a.add(b))
            .collect();

        let reconstructed =
            decode_all(&reconstruct(&[summed[0].clone(), summed[1].clone()], 2).unwrap());
        let expected = [11.0, 22.0, 33.0];
        for (i, exp) in expected.iter().enumerate() {
            assert!((reconstructed[i] - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_3_of_5_threshold() {
        let mut rng = StdRng::seed_from_u64(42);
        let values = encode_all(&[42.0, 100.0]);
        let shares = share(&values, 5, 3, &mut rng).unwrap();

        let r1 = decode_all(
            &reconstruct(
                &[shares[0].clone(), shares[2].clone(), shares[4].clone()],
                3,
            )
            .unwrap(),
        );
        let r2 = decode_all(
            &reconstruct(
                &[shares[1].clone(), shares[3].clone(), shares[4].clone()],
                3,
            )
            .unwrap(),
        );

        for (i, exp) in [42.0, 100.0].iter().enumerate() {
            assert!((r1[i] - exp).abs() < 1e-5);
            assert!((r2[i] - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_threshold_exceeds_parties() {
        let mut rng = StdRng::seed_from_u64(42);
        let values = encode_all(&[1.0]);
        assert!(share(&values, 2, 3, &mut rng).is_err());
    }

    #[test]
    fn test_insufficient_shares() {
        let mut rng = StdRng::seed_from_u64(42);
        let values = encode_all(&[1.0]);
        let shares = share(&values, 3, 2, &mut rng).unwrap();
        assert!(reconstruct(&[shares[0].clone()], 2).is_err());
    }
}
