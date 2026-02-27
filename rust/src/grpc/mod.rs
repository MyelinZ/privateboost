pub mod aggregator_service;
pub mod remote_shareholder;
pub mod shareholder_service;

use curve25519_dalek::scalar::Scalar;

use crate::proto;

pub(crate) fn vec_to_ndarray(values: &[f64]) -> proto::NdArray {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    proto::NdArray {
        dtype: proto::DType::Float64 as i32,
        shape: vec![values.len() as i64],
        data: bytes,
    }
}

pub(crate) fn proto_share_to_scalars(share: &proto::Share) -> Result<(i32, Vec<Scalar>), String> {
    let count = share.count as usize;
    if share.scalars.len() != count * 32 {
        return Err("share scalars length mismatch".into());
    }
    let mut scalars = Vec::with_capacity(count);
    for chunk in share.scalars.chunks_exact(32) {
        let bytes: [u8; 32] = chunk.try_into().unwrap();
        let scalar: Option<Scalar> = Scalar::from_canonical_bytes(bytes).into();
        scalars.push(scalar.ok_or("non-canonical scalar in share")?);
    }
    Ok((share.x, scalars))
}

pub(crate) fn scalars_to_bytes(scalars: &[Scalar]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(scalars.len() * 32);
    for s in scalars {
        bytes.extend_from_slice(&s.to_bytes());
    }
    bytes
}

pub(crate) fn bytes_to_scalars(data: &[u8]) -> Result<Vec<Scalar>, String> {
    if data.len() % 32 != 0 {
        return Err("scalar data length not multiple of 32".into());
    }
    let mut scalars = Vec::with_capacity(data.len() / 32);
    for chunk in data.chunks_exact(32) {
        let bytes: [u8; 32] = chunk.try_into().unwrap();
        let scalar: Option<Scalar> = Scalar::from_canonical_bytes(bytes).into();
        scalars.push(scalar.ok_or("non-canonical scalar")?);
    }
    Ok(scalars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalars_bytes_roundtrip() {
        let scalars = vec![
            Scalar::from(42u64),
            Scalar::ZERO - Scalar::from(17u64),
            Scalar::from(999999u64),
        ];
        let bytes = scalars_to_bytes(&scalars);
        let recovered = bytes_to_scalars(&bytes).unwrap();
        assert_eq!(scalars, recovered);
    }

    #[test]
    fn test_proto_share_roundtrip() {
        let scalars = vec![Scalar::from(100u64), Scalar::from(200u64)];
        let mut data = Vec::with_capacity(scalars.len() * 32);
        for s in &scalars {
            data.extend_from_slice(&s.to_bytes());
        }
        let proto_share = proto::Share {
            x: 3,
            scalars: data,
            count: scalars.len() as i32,
        };
        let (x, recovered) = proto_share_to_scalars(&proto_share).unwrap();
        assert_eq!(x, 3);
        assert_eq!(scalars, recovered);
    }

    #[test]
    fn test_bytes_to_scalars_invalid_length() {
        let result = bytes_to_scalars(&[0u8; 33]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bytes_to_scalars_empty() {
        let result = bytes_to_scalars(&[]).unwrap();
        assert!(result.is_empty());
    }
}
