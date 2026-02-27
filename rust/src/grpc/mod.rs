pub mod aggregator_service;
pub mod remote_shareholder;
pub mod shareholder_service;

use crate::proto;

pub(crate) fn vec_to_ndarray(values: &[f64]) -> proto::NdArray {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    proto::NdArray {
        dtype: proto::DType::Float64 as i32,
        shape: vec![values.len() as i64],
        data: bytes,
    }
}

pub(crate) fn ndarray_to_vec(arr: &proto::NdArray) -> Vec<f64> {
    arr.data
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
