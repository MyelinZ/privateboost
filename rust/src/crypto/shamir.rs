use rand::Rng;

#[derive(Debug, Clone)]
pub struct Share {
    pub x: i32,
    pub y: Vec<f64>,
}

/// Split values into n Shamir shares with given threshold.
///
/// Algorithm:
/// 1. Build coefficient matrix: row 0 = values, rows 1..threshold-1 = random
/// 2. Build Vandermonde matrix: vander[party][k] = (party+1)^k
/// 3. y_matrix = vander @ coeffs
pub fn share(values: &[f64], n_parties: usize, threshold: usize) -> Vec<Share> {
    assert!(threshold <= n_parties, "threshold cannot exceed n_parties");
    let n_values = values.len();
    let mut rng = rand::rng();

    // coeffs: threshold rows x n_values cols
    // Row 0 = the actual values (constant term of polynomial)
    // Rows 1..threshold-1 = random coefficients
    let mut coeffs = vec![vec![0.0f64; n_values]; threshold];
    coeffs[0] = values.to_vec();
    for row in coeffs.iter_mut().skip(1) {
        for col in row.iter_mut() {
            *col = rng.random_range(-1e10..1e10);
        }
    }

    // Evaluate polynomial at x = 1, 2, ..., n_parties
    let mut shares = Vec::with_capacity(n_parties);
    for i in 0..n_parties {
        let x = (i + 1) as i32;
        let xf = x as f64;
        let mut y = vec![0.0f64; n_values];
        for (val_idx, y_val) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            let mut x_power = 1.0;
            for coeff_row in coeffs.iter().take(threshold) {
                sum += x_power * coeff_row[val_idx];
                x_power *= xf;
            }
            *y_val = sum;
        }
        shares.push(Share { x, y });
    }

    shares
}

/// Reconstruct values from threshold shares using Lagrange interpolation at x=0.
pub fn reconstruct(shares: &[Share], threshold: usize) -> Vec<f64> {
    assert!(
        shares.len() >= threshold,
        "need at least {threshold} shares, got {}",
        shares.len()
    );
    let shares = &shares[..threshold];
    let n_values = shares[0].y.len();

    // Lagrange coefficients at x=0
    let x_values: Vec<f64> = shares.iter().map(|s| s.x as f64).collect();
    let n = x_values.len();
    let mut lagrange_coeffs = vec![1.0f64; n];
    for j in 0..n {
        for i in 0..n {
            if i != j {
                lagrange_coeffs[j] *= -x_values[i] / (x_values[j] - x_values[i]);
            }
        }
    }

    // result = lagrange_coeffs @ y_matrix
    let mut result = vec![0.0f64; n_values];
    for (j, share) in shares.iter().enumerate() {
        for (v, r) in result.iter_mut().enumerate() {
            *r += lagrange_coeffs[j] * share.y[v];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shamir_reconstruction() {
        let values = vec![42.0, -17.5, 100.0];
        let shares = share(&values, 3, 2);
        assert_eq!(shares.len(), 3);
        let result = reconstruct(&shares[..2], 2);
        for (a, b) in values.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
    }

    #[test]
    fn test_shamir_linearity() {
        let v1 = vec![10.0, 20.0];
        let v2 = vec![30.0, 40.0];
        let s1 = share(&v1, 3, 2);
        let s2 = share(&v2, 3, 2);
        // Sum shares element-wise per party
        let summed: Vec<Share> = s1
            .iter()
            .zip(s2.iter())
            .map(|(a, b)| Share {
                x: a.x,
                y: a.y.iter().zip(b.y.iter()).map(|(x, y)| x + y).collect(),
            })
            .collect();
        let result = reconstruct(&summed[..2], 2);
        assert!((result[0] - 40.0).abs() < 1e-6);
        assert!((result[1] - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_3_of_5_threshold() {
        let values = vec![99.0];
        let shares = share(&values, 5, 3);
        assert_eq!(shares.len(), 5);
        // Any 3 shares should reconstruct
        let result = reconstruct(&shares[1..4], 3);
        assert!((result[0] - 99.0).abs() < 1e-6);
    }
}
