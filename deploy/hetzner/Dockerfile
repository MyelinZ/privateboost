FROM rust:1-bookworm AS build
# tonic-build/prost-build shell out to protoc at build time; the devenv PROTOC
# env is not present in the container, so tonic-build finds this one on PATH.
# libprotobuf-dev supplies the well-known types (google/protobuf/timestamp.proto
# and friends) the proto imports; protoc-only would fail to resolve them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/privateboost-revision
# The `pbr-core` protocol crate lives at crates/pbr-core, so the
# whole workspace is inside this build context — no external context needed.
COPY . .
RUN cargo build --release --bin pbr-server

FROM gcr.io/distroless/cc-debian12 AS runtime
COPY --from=build /build/privateboost-revision/target/release/pbr-server /usr/local/bin/pbr-server
# Held-out eval split for [eval.datasets]: `pbr-core`'s heart_disease copy,
# whose first 80% is exactly the app's bundled train split, so the last 20%
# never overlaps training rows. data/heart_disease.csv has the same rows in a
# different order and must not be used here.
COPY --from=build /build/privateboost-revision/crates/pbr-core/tests/data/heart_disease.csv /data/heart_disease.csv
# breast_cancer, pima_diabetes, and cdc_diabetes have no reordered copy to
# trip over: each app's bundled train split is byte-identical to its file's
# first 80% (455/569, 614/768, 56553/70692 rows respectively), so the last
# 20% is a clean held-out split.
COPY --from=build /build/privateboost-revision/data/breast_cancer.csv /data/breast_cancer.csv
COPY --from=build /build/privateboost-revision/data/pima_diabetes.csv /data/pima_diabetes.csv
COPY --from=build /build/privateboost-revision/data/cdc_diabetes.csv /data/cdc_diabetes.csv
ENTRYPOINT ["/usr/local/bin/pbr-server"]
