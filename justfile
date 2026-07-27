# Recipes that work in the published artifact, which carries no manuscript/.
# The private justfile imports this file; scripts/export_public.sh ships it as
# the artifact's justfile. Keep manuscript-only recipes (paper, paper-zip,
# paper-figures, paper-data) out of here: they would break in a tree that has
# no manuscript/ to read.

build:
    cargo build --workspace

# Fast tier: every workspace test, process-per-test, globally scheduled
# (timeouts + test groups live in .config/nextest.toml).
test:
    cargo nextest run --workspace

lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Runs every suite `just test` misses: the standalone app/rust workspace and the Flutter app.
test-all: test
    cd app/rust && cargo test
    cd app && flutter test

# The 4-process end-to-end (AUC gate on heart_disease). Slow; the harness
# builds the release server/client binaries itself. Excludes wire_grid_sweep
# (the ~30-40 min wire-cost sweep is `just e2e-wire-cost`, not a correctness gate).
e2e:
    cargo nextest run -p pbr-e2e --release --run-ignored all -E 'not test(=wire_grid_sweep)'

# Measure per-client wire bytes over TLS for the full 56-config grid ->
# results/wire_measured.csv (slow: ~30-40 min, fresh cluster per config).
# The -E filter runs only wire_grid_sweep, not the other ignored tests in
# the same test binary.
e2e-wire-cost:
    cargo nextest run -p pbr-e2e --test wire_grid --release --run-ignored ignored-only -E 'test(=wire_grid_sweep)'

# Test coverage over the fast tier; HTML drill-down (red lines = untested)
# plus a per-file terminal summary.
coverage:
    cargo llvm-cov nextest --workspace --no-fail-fast --no-report
    cargo llvm-cov report --html
    cargo llvm-cov report --summary-only
    @echo "HTML report: target/llvm-cov/html/index.html"

# Coverage including the release e2e (slow). Adds the pbr-client driver-lib
# paths the e2e exercises in-process; the spawned pbr-server processes are
# SIGKILLed by the harness and never flush their profiles, so server code
# still counts only through the in-process tests.
coverage-full:
    cargo llvm-cov clean --workspace
    cargo llvm-cov nextest --workspace --release --no-fail-fast --no-report
    cargo llvm-cov nextest -p pbr-e2e --release --no-fail-fast --no-report --run-ignored all -E 'not test(=wire_grid_sweep)'
    cargo llvm-cov report --release --html
    cargo llvm-cov report --release --summary-only
    @echo "HTML report: target/llvm-cov/html/index.html"

# Schedule a demo session on a running deploy/local or deploy/emulator
# cluster (both listen on 127.0.0.1:42800 and commit the same dev
# admin_token) — start the cluster first, then `just demo-session`. Defaults
# (min/target 10) match deploy/local/shareholder-N.toml's committed
# anonymity floor; deploy/emulator's floor is 1, so its demo overrides both
# (see app/README.md). The 2-minute default window gives an interactive
# operator ~20 minutes of give-up budget (window × 10); tests wanting the old
# 5s window can pass it explicitly. Note: runs `cargo run -p pbr-admin`
# (debug mode), so the first invocation after a clean checkout pauses to
# compile — run `cargo build -p pbr-admin` beforehand to avoid that.
demo-session dataset="heart_disease" min_clients="10" target_clients="10" aggregator="http://127.0.0.1:42800" window_ms="120000":
    PBR_ADMIN_TOKEN=dev-admin-token cargo run -p pbr-admin -- create-session \
        --aggregator {{aggregator}} \
        --dataset {{dataset}} \
        --min-clients {{min_clients}} \
        --target-clients {{target_clients}} \
        --window-ms {{window_ms}}

# Build the pilot APK (arm64, TLS-pinned to the fleet VM), install it on the
# connected device, and launch it. `serial` picks a device when adb sees
# several; `ip` overrides the terraform-provisioned VM address; `batch`/`of`
# pick this device's train-split slice (args are positional: app-pilot SERIAL IP
# BATCH OF POLL); `poll` overrides the foreground poll interval to that many
# seconds (the in-app switch, off by default, is what turns the poller on; 0
# leaves the interval at its built-in default). The rustup stable toolchain
# must lead PATH: the nix cargo carries no Android std.
app-pilot serial="" ip="" batch="0" of="1" poll="0":
    #!/usr/bin/env bash
    set -euo pipefail
    ip="{{ip}}"; [ -n "$ip" ] || ip=$(terraform -chdir=deploy/hetzner/infra output -raw server_ip)
    serial_flag=(); [ -n "{{serial}}" ] && serial_flag=(-s "{{serial}}")
    export PATH="$(dirname "$(rustup which --toolchain stable cargo)"):$HOME/.cargo/bin:$PATH"
    cp deploy/hetzner/secrets/ca.crt app/assets/pilot-ca.pem
    (cd app/rust && cargo ndk -t arm64-v8a --platform 24 -o ../android/app/src/main/jniLibs build --release --lib)
    (cd app && flutter build apk --release \
      --dart-define=PBR_AGG_ENDPOINT="https://$ip:42800" \
      --dart-define=PBR_BATCH_ID={{batch}} \
      --dart-define=PBR_BATCH_COUNT={{of}} \
      --dart-define=PBR_FG_POLL_SECONDS={{poll}})
    adb "${serial_flag[@]}" install -r app/build/app/outputs/flutter-apk/app-release.apk
    adb "${serial_flag[@]}" shell monkey -p dev.pboost.pboost_app -c android.intent.category.LAUNCHER 1

# Regenerate all results/ from data/ (hours; args restrict to named datasets)
reproduce-experiments *ARGS:
    scripts/run_experiments.sh {{ARGS}}

# Regenerate the named datasets into a scratch tree and diff every committed CSV
# against it, so the byte-for-byte reproducibility claim is checked rather than
# asserted. Defaults to heart_disease, the fastest dataset; pass more to widen
# the check (`just reproduce-verify heart_disease breast_cancer`).
#
# timing.csv is skipped because it records wall-clock. Nothing else regenerable
# carries a timing column; results/fleet_*.csv do, but they are field
# measurements this recipe never regenerates. Only paths git tracks are
# compared, which keeps git-ignored intermediates out of the comparison.
reproduce-verify *DATASETS:
    #!/usr/bin/env bash
    set -euo pipefail
    datasets="{{DATASETS}}"; [ -n "$datasets" ] || datasets=heart_disease
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    RESULTS_ROOT="$scratch" scripts/run_experiments.sh $datasets
    compared=0
    failed=0
    while IFS= read -r committed; do
        case "$committed" in */timing.csv) continue;; esac
        regenerated="$scratch/${committed#results/}"
        [ -f "$regenerated" ] || continue
        compared=$((compared + 1))
        if ! cmp -s "$committed" "$regenerated"; then
            echo "DIFFERS: $committed"
            failed=$((failed + 1))
        fi
    done < <(git ls-files 'results/*.csv')
    if [ "$compared" -eq 0 ]; then
        echo "reproduce-verify: compared nothing, so this proved nothing" >&2
        exit 1
    fi
    if [ "$failed" -ne 0 ]; then
        echo "reproduce-verify: $failed of $compared CSVs did not reproduce" >&2
        exit 1
    fi
    echo "reproduce-verify: all $compared compared CSVs reproduced exactly"

# Regenerate the plotted figures and print the binning and fleet tables.
figures-plots:
    uv run python analysis/generate_figures.py
    uv run python analysis/plot_binning_analysis.py
    uv run python analysis/binning_tables.py
    uv run python analysis/fleet_table.py

# Build the three TikZ figures (architecture, fl-comparison, path-hiding) from
# their .tex sources. latexmk works in place, so its .aux/.fls/.log output lands
# beside the sources and is git-ignored. The sources sit under manuscript/ in
# this tree and at the top level once exported.
figures-tikz:
    #!/usr/bin/env bash
    set -euo pipefail
    figdir=manuscript/figures
    [ -d "$figdir" ] || figdir=figures
    cd "$figdir" && latexmk -pdf -interaction=nonstopmode \
        architecture.tex fl-comparison.tex path-hiding.tex

# Everything the artifact can regenerate from data/, end to end: protocol runs,
# XGBoost baselines, then figures and tables. Hours, dominated by the CDC BRFSS
# sweeps. The field measurements in results/fleet_*.csv are not regenerable and
# are left alone.
reproduce: reproduce-experiments figures

# Every figure and table from the committed results/, without re-running any
# experiment. Minutes. This is the useful entry point for a reader.
figures: figures-plots figures-tikz
