//! Shared helper for `pbr-server`'s integration test binaries. Cargo compiles
//! every file under `tests/` (and every `tests/*/main.rs`) as its own crate,
//! so this module is not a dependency but is pulled in verbatim via
//! `mod common;` (or, from a subdirectory binary like `tests/shareholder`,
//! `#[path = "../common/mod.rs"] mod common;`).

use tonic::metadata::MetadataValue;

pub(crate) fn bearer(token: &str) -> MetadataValue<tonic::metadata::Ascii> {
    format!("Bearer {token}").parse().unwrap()
}
