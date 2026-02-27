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

#[allow(dead_code)]
pub(crate) fn ndarray_to_vec(arr: &proto::NdArray) -> Vec<f64> {
    arr.data
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
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
