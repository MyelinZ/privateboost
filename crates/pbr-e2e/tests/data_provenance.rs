//! Pins `app/assets/heart_disease_train.csv` to the train split of
//! `pbr-core`'s dataset. Flutter assets must live under `app/`, so the app
//! carries its own copy of the first 80% of
//! `crates/pbr-core/tests/data/heart_disease.csv` rather than reading that
//! file directly. If the pbr-core CSV ever changes row order, the app's copy
//! goes stale silently, this test turns that into a build failure instead
//! of a quietly-wrong demo row or a train/held-out split that no longer
//! matches the e2e test's.

use std::path::Path;

#[test]
fn app_train_asset_matches_pbr_core_split() {
    let core_csv = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../pbr-core/tests/data/heart_disease.csv"),
    )
    .expect("read crates/pbr-core/tests/data/heart_disease.csv");
    let app_asset = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/assets/heart_disease_train.csv"),
    )
    .expect("read app/assets/heart_disease_train.csv");

    let core_lines: Vec<&str> = core_csv.lines().collect();
    let app_lines: Vec<&str> = app_asset.lines().collect();

    // Header + first 237 data rows: 80% of the pbr-core file's 297 rows.
    assert_eq!(
        app_lines.len(),
        238,
        "app/assets/heart_disease_train.csv row count no longer matches the 80% split"
    );
    // Without this bound a shrunken pbr-core file would silently truncate the
    // zip below and the comparison would pass on the surviving prefix.
    assert!(
        core_lines.len() >= app_lines.len(),
        "crates/pbr-core/tests/data/heart_disease.csv shrank below the app's train split"
    );
    for (i, (app_line, core_line)) in app_lines.iter().zip(core_lines.iter()).enumerate() {
        assert_eq!(
            app_line, core_line,
            "app/assets/heart_disease_train.csv line {i} has desynchronized from \
             crates/pbr-core/tests/data/heart_disease.csv"
        );
    }
}
