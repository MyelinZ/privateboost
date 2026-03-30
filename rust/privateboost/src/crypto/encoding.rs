use super::field::{F, MersenneField};

const PRECISION: i64 = 1 << 24;

pub fn encode(value: f64) -> MersenneField {
    let scaled = (value * PRECISION as f64).round() as i64;
    MersenneField::from_i64(scaled)
}

pub fn decode(element: MersenneField) -> f64 {
    element.to_i64() as f64 / PRECISION as f64
}

pub fn encode_all(values: &[f64]) -> Vec<F> {
    values.iter().map(|&v| encode(v)).collect()
}

pub fn decode_all(elements: &[F]) -> Vec<f64> {
    elements.iter().map(|&e| decode(e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_positive() {
        let value = 42.0;
        let encoded = encode(value);
        let decoded = decode(encoded);
        assert!((decoded - value).abs() < 1e-5, "got {decoded}");
    }

    #[test]
    fn test_encode_decode_negative() {
        let value = -7.5;
        let encoded = encode(value);
        let decoded = decode(encoded);
        assert!((decoded - value).abs() < 1e-5, "got {decoded}");
    }

    #[test]
    fn test_encode_decode_zero() {
        let decoded = decode(encode(0.0));
        assert!((decoded).abs() < 1e-7);
    }

    #[test]
    fn test_encode_decode_batch() {
        let values = vec![42.0, 100.0, -7.5, 0.0, 12.345];
        let encoded = encode_all(&values);
        let decoded = decode_all(&encoded);
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 1e-5, "{orig} != {dec}");
        }
    }
}
