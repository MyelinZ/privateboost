/// Inverse normal CDF (Beasley-Springer-Moro algorithm).
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 4] = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    const B: [f64; 4] = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    const C: [f64; 9] = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];

    let y = p - 0.5;
    if y.abs() < 0.42 {
        let r = y * y;
        y * (((A[3] * r + A[2]) * r + A[1]) * r + A[0])
            / ((((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0)
    } else {
        let r = if y < 0.0 { p } else { 1.0 - p };
        let s = (-2.0 * r.ln()).sqrt();
        let t = C[0]
            + s * (C[1]
                + s * (C[2]
                    + s * (C[3]
                        + s * (C[4]
                            + s * (C[5]
                                + s * (C[6]
                                    + s * (C[7] + s * C[8])))))));
        if y < 0.0 { -t } else { t }
    }
}

/// Derive the optimal range (in standard deviations) for equal-width binning.
/// With n_bins inner bins + 2 overflow bins, each should hold 1/(n_bins+2) of
/// a Gaussian. The inner range covers the central n_bins/(n_bins+2) fraction.
pub fn bin_range_stds(n_bins: usize) -> f64 {
    norm_ppf(1.0 - 1.0 / (n_bins as f64 + 2.0))
}

use crate::protocol::{BinConfiguration, BinMethod};

#[derive(Clone, Debug)]
pub struct FeatureStats {
    pub mean: f64,
    pub std: f64,
}

impl FeatureStats {
    pub fn from_totals(total_x: f64, total_x2: f64, n: usize) -> Self {
        let mean = total_x / n as f64;
        let variance = (total_x2 / n as f64) - (mean * mean);
        let std = variance.max(0.0).sqrt();
        Self { mean, std }
    }

    pub fn to_bins(&self, feature_idx: usize, n_bins: usize, bin_method: &BinMethod) -> BinConfiguration {
        let n_inner = n_bins + 1;
        let mut inner_edges = Vec::with_capacity(n_inner);

        match bin_method {
            BinMethod::Uniform => {
                let k = bin_range_stds(n_bins);
                let range_min = self.mean - k * self.std;
                let range_max = self.mean + k * self.std;
                for i in 0..n_inner {
                    inner_edges.push(
                        range_min + (range_max - range_min) * i as f64 / (n_inner - 1) as f64,
                    );
                }
            }
            BinMethod::Gaussian => {
                for i in 0..n_inner {
                    let p = (i + 1) as f64 / (n_inner + 1) as f64;
                    inner_edges.push(self.mean + self.std * norm_ppf(p));
                }
            }
        }

        let mut edges = Vec::with_capacity(n_inner + 2);
        edges.push(f64::NEG_INFINITY);
        edges.extend_from_slice(&inner_edges);
        edges.push(f64::INFINITY);

        BinConfiguration { feature_idx, edges }
    }
}
