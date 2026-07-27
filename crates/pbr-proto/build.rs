fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tonic_prost_build 0.14 emits no rerun-if-changed itself, and the proto
    // lives outside this package, so without this line a proto-only edit
    // never regenerates the bindings (cargo's fallback only watches package
    // sources).
    println!("cargo:rerun-if-changed=../../proto");
    tonic_prost_build::configure().compile_protos(
        &["../../proto/privateboost/v1/privateboost.proto"],
        &["../../proto"],
    )?;
    Ok(())
}
