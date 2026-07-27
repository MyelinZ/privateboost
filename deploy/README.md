# Local 4-process cluster

Three shareholder daemons plus one aggregator, all on loopback, run by hand.
`just e2e` drives the same configs headless on freshly reserved ports (prefer
the recipe over `cargo test -p pbr-e2e --release -- --ignored`, which also
picks up the 30-40 minute `wire_grid_sweep`).

Config paths are **relative to the repository root**, so start every process
from there. Auth is the workspace's dev issuer (`https://test-issuer.local`,
audience `pbr`, kid `test-1`) against the committed keypair
`crates/pbr-server/tests/fixtures/test_key{,.pub}.pem`: a dev/test identity
only.

## Topology

| Process       | Config                            | Client-facing      | Internal (localhost-only) |
|---------------|-----------------------------------|--------------------|---------------------------|
| shareholder 1 | `deploy/local/shareholder-1.toml` | `127.0.0.1:42801`  | `127.0.0.1:42811`         |
| shareholder 2 | `deploy/local/shareholder-2.toml` | `127.0.0.1:42802`  | `127.0.0.1:42812`         |
| shareholder 3 | `deploy/local/shareholder-3.toml` | `127.0.0.1:42803`  | `127.0.0.1:42813`         |
| aggregator    | `deploy/local/aggregator.toml`    | `127.0.0.1:42800`  | (none)                    |

Shareholder order is load-bearing: endpoint *i* serves Shamir evaluation point
x = *i*, so the aggregator's `internal_shareholder_endpoints` and any client's
`--shareholders` list must both be in x = 1, 2, 3 order.

## Running it

Any start order works; the aggregator retries its internal connections while
the shareholders come up.

```bash
# pbr-client's binary is behind its `cli` feature, off by default so the
# CLI-only deps stay out of the mobile build.
cargo build --release --bin pbr-server --bin pbr-client --features pbr-client/cli

RUST_LOG=info target/release/pbr-server --role shareholder --config deploy/local/shareholder-1.toml &
RUST_LOG=info target/release/pbr-server --role shareholder --config deploy/local/shareholder-2.toml &
RUST_LOG=info target/release/pbr-server --role shareholder --config deploy/local/shareholder-3.toml &
RUST_LOG=info target/release/pbr-server --role aggregator  --config deploy/local/aggregator.toml &
```

The aggregator boots hosting no session, so schedule one (see "Admin plane"),
then run ten clients against it:

```bash
just demo-session

for i in $(seq 10); do
  target/release/pbr-client \
    --aggregator http://127.0.0.1:42800 \
    --features "$((40 + i)).0,0.0,2.0,112.0,160.0,0.0,0.0,138.0,0.0,0.0,2.0,0.0,3.0" \
    --label "$((i % 2))" \
    --mint --mint-key crates/pbr-server/tests/fixtures/test_key.pem &
done
wait
```

Ten, and concurrent, both matter. `just demo-session`'s defaults
(`min_clients = target_clients = 10`) match each shareholder's committed
anonymity floor (`min_clients = 10`), and a round closes only once that many
distinct clients have submitted. One invocation drives the whole session, every
round waiting for 10 distinct submitters inside its own window, so ten
sequential runs deadlock: the first waits for nine peers that never arrive and
fails once its give-up budget is spent.

Each client mints a dev token from the committed key (the `--mint-*` defaults
match the configs), learns the shareholder list and threshold from
`EnrollSession`, computes its shares with `pbr-core`, fans them out (2-of-3
best effort), and exits at `COMPLETED`. For the full heart_disease shape
(`target_clients = 237`, what `just e2e` trains):
`just demo-session heart_disease 10 237`.

Stop the cluster with `kill %1 %2 %3 %4`, or `pkill -x pbr-server`.

## Admin plane: scheduling sessions

Every session, the first included, is created over the admin plane, and
`deploy/local/aggregator.toml` commits the three things that needs:

- **`admin_token`** gates `AdminService`. The dev value `dev-admin-token` is no
  new secret class on a loopback cluster already using committed dev keys, but
  a cluster reachable from outside must keep its token out of git the way
  `deploy/hetzner/secrets/` does (git-ignored via its own `.gitignore`).
- **`[datasets]`** lists the accepted dataset ids and their feature widths
  (`heart_disease = 13`). `CreateSession` refuses anything unlisted.
- **`state_path`** is a mandatory SQLite file holding the session list. A
  restart re-serves it under the original ids with anything still in flight
  demoted to `Failed`, so a polling client sees a clean failure and re-enrolls
  instead of waiting on a session that cannot resume.

`just demo-session` wraps `pbr-admin` with the dev token and demo-sized
defaults; call `pbr-admin` directly for full control. Its defaults match the
heart_disease reference configuration for
`--trees`/`--depth`/`--bins`/`--lr`/`--lambda`, but the two differ on
`--window-ms`: `pbr-admin` defaults to 5000, the recipe to 120000 (2 minutes,
~20 minutes of give-up budget at window × 10). Pass the recipe's positionally
for the shorter window:
`just demo-session heart_disease 10 10 http://127.0.0.1:42800 5000`. The admin
token comes from `PBR_ADMIN_TOKEN` only, never a flag, so it cannot land in
shell history or `ps` output.

```bash
export PBR_ADMIN_TOKEN=dev-admin-token
cargo run -p pbr-admin -- create-session --aggregator http://127.0.0.1:42800 \
  --dataset heart_disease --min-clients 10 --target-clients 237
cargo run -p pbr-admin -- list-sessions  --aggregator http://127.0.0.1:42800
cargo run -p pbr-admin -- delete-session --aggregator http://127.0.0.1:42800 \
  --session-id <id>
```

Nothing is replaced implicitly: a failed or wrong-sized run stays listed, and a
second `demo-session` adds a second live session. That breaks the CLI, which
has no `--session-id` flag and so always sends the empty selector: the
aggregator refuses it while more than one session is live (`session_id
required: this process hosts 2 live sessions`). Delete the unwanted id, which
aborts its round loop and closes it out at the shareholders, or wait, since a
completed or failed session stops counting as live. At most 16 may be live at
once; past that `create-session` fails.

**`--min-clients` is the round-close target, not the privacy floor.** At least
this many distinct clients must contribute before a round closes, so raising it
raises the anonymity set a round realises. The floor itself is each shareholder
daemon's own `min_clients` in `deploy/*/shareholder-N.toml`, checked when the
aggregator tries to reconstruct a sum; no session parameter or admin caller can
lower it. Setting `--min-clients` below it is a misconfiguration rather than a
weakened guarantee: the round closes normally, then a shareholder refuses to
release its sum, reconstruction comes up short, and the session moves to
`Failed`. The CLI has no default; match it to how the shareholders were
deployed.
