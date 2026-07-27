# Hetzner deployment

Single-VM deployment of the four PrivateBoost services: one aggregator on
`:42800` and three shareholders on `:42801`-`:42803`. All four present the same
server certificate over TLS, and its SAN pins the VM's IP address.

## Fleet APK version lockstep

**Rebuild and reinstall every device's APK together whenever the protocol
crate's share layout changes** (for example, the stats share vector growing
from `2f+2` to `2f+3` elements to carry a trailing record count). Shareholder
share-summing zip-truncates on a length mismatch and raises no error, so one
stale device submitting the old layout alongside updated devices in the same
session silently corrupts the fleet-wide sums — wrong means, variances, and
bin edges, with no error signal anywhere to catch it after the fact.

Verifying every device in the fleet is running the current build is therefore
a precondition of any capture run, not an optional check.

## Generate TLS material

`gen-certs.sh` produces a self-signed CA and one server certificate for the VM.
Run it once per VM, passing the VM's public IP:

```bash
deploy/hetzner/gen-certs.sh <VM_IP>
```

It writes four files to `deploy/hetzner/secrets/`:

| File         | Purpose                                                              |
|--------------|---------------------------------------------------------------------|
| `ca.crt`     | CA certificate; committed and bundled into the app so clients trust the server |
| `ca.key`     | CA private key; local only, never committed                         |
| `server.crt` | Server certificate with an `IP:<VM_IP>` SAN, shared by all four services |
| `server.key` | Server private key; local only, never committed                     |

Certificates are valid for 825 days. Re-running the script overwrites them in
place. Only `ca.crt` is committed; the keys and `server.crt` under `secrets/`
are git-ignored.

## FCM service account

The aggregator sends round-open notifications through Firebase Cloud Messaging.
A production deployment authenticates those sends with a service-account key
instead of developer credentials. Create one once, in the Firebase console for
the project named by `GOOGLE_CLOUD_PROJECT`:

1. Open the project's service-account settings.
2. Create a service account and grant it the **Firebase Cloud Messaging API
   Admin** role.
3. Generate a JSON key for the account and download the file.
4. Save it as `deploy/hetzner/secrets/fcm-service-account.json`.

The file stays local (git-ignored). Point the aggregator's `[fcm]` config at
this path to use it.

## Admin token

The aggregator's admin plane (`CreateSession`) is what an operator uses to open
a training session for a specific dataset — the session the phones then join.
It is guarded by a bearer token: `admin_token` in `configs/aggregator.toml`
carries an `<ADMIN_TOKEN>` placeholder that `ship.sh` substitutes from a secret
file, exactly as it does `<VM_IP>`. Generate the token once per VM:

```bash
openssl rand -hex 32 > deploy/hetzner/secrets/admin-token
```

The file stays local (git-ignored); the token never enters git. Without it
`ship.sh` refuses to deploy (the phones would have no dataset session to join),
and the aggregator would log the admin plane as disabled. The same token value
goes in `PBR_ADMIN_TOKEN` when you run `pbr-admin` (see "Fleet capture run").

## Build the deploy image

`docker-compose.yml` runs the image tagged `pbr-server:deploy`. The image builds
only `pbr-server`. The `pbr-core` crate lives at
`crates/pbr-core`, so the whole workspace is inside the build context:

```bash
docker build -t pbr-server:deploy -f deploy/hetzner/Dockerfile .
```

Run it from the repo root (the build context is the repo root, trimmed by the
root `.dockerignore`). Stage 1 is `rust:1-bookworm` with
`protobuf-compiler`/`libprotobuf-dev`; stage 2 is a distroless runtime carrying
just the binary.

## Run the stack

`docker-compose.yml` starts the four services on the host network. The aggregator
(`:42800`) and shareholders (`:42801`-`:42803`) bind `0.0.0.0` so phones can reach
them; the internal gather plane stays on `127.0.0.1:42811`-`:42813`. Each service
mounts its config read-only at `/config.toml` and the shared `server.crt` /
`server.key` under `/secrets/`; the aggregator also mounts
`fcm-service-account.json` and `firestore-sa.json`.

Generate the TLS material and FCM key (above), replace `<VM_IP>` in
`configs/aggregator.toml`'s `client_shareholder_endpoints` with the VM's public
IP, then:

```bash
docker compose -f deploy/hetzner/docker-compose.yml up -d
```

## Deploy runbook

`ship.sh` and `smoke.sh` drive the whole deploy against a real VM; the sections
above document what each piece touches. The full sequence, from an empty project
to a running stack, is:

1. **Provision the VM.** From the repo root, with the Hetzner token in the
   environment:

   ```bash
   HCLOUD_TOKEN=<token> terraform -chdir=deploy/hetzner/infra apply
   ```

   Note the `server_ip` output (see `infra/README.md`).

2. **Generate the TLS material** for that IP (see "Generate TLS material"):

   ```bash
   deploy/hetzner/gen-certs.sh "$(terraform -chdir=deploy/hetzner/infra output -raw server_ip)"
   ```

3. **Place the FCM key** at `deploy/hetzner/secrets/fcm-service-account.json`
   (see "FCM service account"), and **generate the admin token**:

   ```bash
   openssl rand -hex 32 > deploy/hetzner/secrets/admin-token
   ```

   (see "Admin token").

   Also place the datastore-scoped Firestore key at
   `deploy/hetzner/secrets/firestore-sa.json` — `ship.sh` refuses to ship
   without it, since the aggregator's per-tree metrics need it at startup.
   See "Fleet capture run" for the key's required role.

4. **Build, ship, and start the stack:**

   ```bash
   deploy/hetzner/ship.sh
   ```

   It builds `pbr-server:deploy`, loads it onto the VM, renders the configs
   (substituting the IP into every `<VM_IP>` in a temp copy, never the tracked
   files), ships the compose file, rendered configs, and secrets to `/opt/pbr`,
   then runs `docker compose up -d`. It reads the IP from terraform; pass
   `ship.sh <VM_IP>` to override.

5. **Smoke-test the deployment:**

   ```bash
   deploy/hetzner/smoke.sh
   ```

   It verifies the TLS handshake on `:42800`-`:42803` against `secrets/ca.crt`,
   confirms the aggregator logged `aggregator up`, and prints its recent log so
   you can see the session state.

6. **Point the app at the VM.** Build the app against the VM's IP with the
   committed `ca.crt` bundled as its trust root (that build command is covered
   by the app build docs). The certificate's SAN pins the IP, so the app must
   use the same address `gen-certs.sh` signed.

### Session operations

The aggregator boots hosting no session at all — nothing is running until an
operator opens one on the admin plane (see "Fleet capture run"). Restarting it
is therefore safe at any time and mints nothing:

```bash
ssh root@<VM_IP> 'cd /opt/pbr && docker compose restart aggregator'
```

With `state_path` set (as in `configs/aggregator.toml`), the session list
persists across the restart, so an in-progress session's history survives; a
restart mid-round still loses in-flight round state (the in-memory device
registry and any not-yet-closed round are gone), so a session that was live
when the aggregator went down will not resume on its own. Open the next
session explicitly with `pbr-admin create-session`.

### Deleting sessions

The aggregator boots hosting nothing, but every completed or failed run
stays listed, so the hosted list still accumulates dead entries that show up
on every phone. Remove them by id — do NOT wipe `state/sessions.sqlite` on the
VM, which nukes all history and needs a restart:

```bash
export PBR_ADMIN_TOKEN="$(cat deploy/hetzner/secrets/admin-token)"
cargo run -p pbr-admin --release -- list-sessions \
  --aggregator https://<VM_IP>:42800 \
  --ca-cert deploy/hetzner/secrets/ca.crt
cargo run -p pbr-admin --release -- delete-session \
  --aggregator https://<VM_IP>:42800 \
  --ca-cert deploy/hetzner/secrets/ca.crt \
  --session-id <SESSION_ID>
```

`list-sessions` prints one line per session: id, phase, dataset (blank for a
dataset-less session), creation time. Deleting a LIVE session aborts its round
loop and frees its shareholder pools (best-effort; the idle sweep is the
backstop) — it is also the kill switch for a stuck run. Removals are
checkpointed, so deleted sessions stay gone across restarts. A delete against
unreachable shareholders may report a timeout even though the removal already
succeeded and persisted (the checkpoint precedes the best-effort shareholder
cleanup); re-run `list-sessions` to confirm.

### Fleet capture run

The three phones each run ONE batch client (their `PBR_BATCH_ID` slice of the
train split, submitted as a single batch). They join a `heart_disease` session
that the operator creates on the admin plane. Order for a capture:

1. **Open a fresh session.** From the repo root, with the admin token in the
   environment (the same value shipped in `secrets/admin-token`):

   ```bash
   export PBR_ADMIN_TOKEN="$(cat deploy/hetzner/secrets/admin-token)"
   cargo run -p pbr-admin --release -- create-session \
     --aggregator https://<VM_IP>:42800 \
     --ca-cert deploy/hetzner/secrets/ca.crt \
     --dataset heart_disease \
     --min-clients 3 --target-clients 3 --window-ms 120000
   ```

   (`--aggregator`, `--ca-cert`, and the rest are all arguments of the
   `create-session` subcommand, so they follow it.)

   It prints the new session id. `--min-clients 3` is the 3-device fleet size:
   a functional demo, not a privacy-grade anonymity set (the shareholders'
   `min_clients = 3` floor matches). `--window-ms 120000` sets the per-round
   window to 2 minutes (give-up budget window x 10 = 20 min); WITHOUT it the
   session defaults to a 5-second window and a 50-second give-up, which leaves
   no slack for a phone that is slow to wake — pass it explicitly. The
   aggregator config carries no window of its own; every session's window
   comes from the `create-session` call that opens it.

   Per-tree quality metrics need no separate process: the aggregator scores
   every finished tree of the `heart_disease` session against the held-out
   split baked into its image (`/data/heart_disease.csv`, the last 20% of
   `pbr-core`'s CSV whose first 80% is the app's train split) and writes each row
   to Firestore `paperSimTreeMetrics` itself. This requires the
   datastore-scoped service-account key at
   `deploy/hetzner/secrets/firestore-sa.json` (git-ignored) before shipping:
   the key needs Firestore write (`roles/datastore.user`, e.g. the project's
   Firebase Admin SDK key) — the FCM-only key will get a 403. With `[eval]`
   configured, a missing or unreadable key stops the aggregator at startup.

2. **Install and sign in** on the three phones (see the app build docs for the
   fleet APKs). Each device signs into its own account; the app's session list
   shows the `heart_disease` session, and Contribute joins it.

3. **Watch it train.** Follow `docker compose logs -f aggregator` for the
   tree-finished lines and the `wrote per-tree metric` lines carrying each
   tree's AUC.

**All three devices must participate in every round.** With `min_clients` and
`target_clients` both 3 and the shareholder floor at 3, a round closes only
when all three commitments overlap — there is no slack, because closing on two
would leave the gather below the shareholders' floor and wedge on
`InsufficientShares`. If one device's screen sleeps and its wake does not fire,
the round stalls until the give-up budget (window x 10) elapses and the session
fails. Keep all three screens on and charging for the duration of the capture.

A re-ship that changes only configs keeps the same image id, so
`docker compose up -d` does NOT recreate the containers; restart the
affected services for the new configs to take effect.

Follow a session live:

```bash
ssh root@<VM_IP> 'cd /opt/pbr && docker compose logs -f aggregator'
```

Tear the VM down when finished:

```bash
HCLOUD_TOKEN=<token> terraform -chdir=deploy/hetzner/infra destroy
```

Destroying releases the public IP. The next `apply` gets a different address,
so re-run `gen-certs.sh` with the new IP and re-ship: the server certificate's
SAN pins the old one, and clients would otherwise reject the handshake.

## Local smoke

`docker-compose.smoke.yml` runs the same four services against a single local CLI
client: the dev JWT issuer instead of Firebase, no FCM, the shareholders'
`min_clients = 1` floor, and TLS with a `127.0.0.1` certificate. It checks the
image and wiring end to end; it is not a privacy configuration. The smoke
configs live under `configs/smoke/`.

Generate loopback certs into a scratch directory, point `SMOKE_CERT_DIR` at it,
build the image (above), then bring the stack up:

```bash
export SMOKE_CERT_DIR=$(mktemp -d)
deploy/hetzner/gen-certs.sh 127.0.0.1 --out "$SMOKE_CERT_DIR"
docker compose -f deploy/hetzner/docker-compose.smoke.yml up -d
```

The aggregator boots hosting no session — open one on the admin plane with the
committed dev token (`configs/smoke/aggregator.toml`'s `admin_token`), sized to
match the single-client smoke run (`--min-clients 1 --target-clients 1`) and a
generous window so the manual steps below have budget:

```bash
PBR_ADMIN_TOKEN=dev-admin-token cargo run -p pbr-admin --release -- create-session \
  --aggregator https://127.0.0.1:42800 \
  --ca-cert "$SMOKE_CERT_DIR/ca.crt" \
  --dataset heart_disease \
  --min-clients 1 --target-clients 1 --window-ms 120000 \
  --trees 3 --depth 2 --bins 5
```

Nothing deletes a session: if this run fails or trains the wrong shape,
the session stays listed. Running `create-session` again does not replace
it — it adds a second session, and the empty-selector CLI flow (`pbr-client`
with no `--session-id`) then refuses with `session_id required: this
process hosts 2 live sessions`, because it can no longer guess which one you
mean. Restart the aggregator container to clear its session list (the smoke
config's `state_path` is `":memory:"`, an ephemeral SQLite store, so a
restart starts empty) before retrying, rather than creating another one:

```bash
docker compose -f deploy/hetzner/docker-compose.smoke.yml restart aggregator
```

Drive one training row through the session (it mints its own dev token and
pins the smoke CA):

```bash
cargo run -p pbr-client --features cli --release -- \
  --aggregator https://127.0.0.1:42800 \
  --shareholders https://127.0.0.1:42801,https://127.0.0.1:42802,https://127.0.0.1:42803 \
  --threshold 2 \
  --features 45.0,0.0,2.0,112.0,160.0,0.0,0.0,138.0,0.0,0.0,2.0,0.0,3.0 \
  --label 0 \
  --mint --mint-key crates/pbr-server/tests/fixtures/test_key.pem \
  --ca-cert "$SMOKE_CERT_DIR/ca.crt"
```

The aggregator logs `session completed` when the run finishes. Keep
`SMOKE_CERT_DIR` exported for the tear-down:

```bash
docker compose -f deploy/hetzner/docker-compose.smoke.yml down
```

## Regenerating the TLS test fixtures

The in-app TLS tests consume fixtures produced by this same script, so the test
and production certificates cannot drift. Regenerate them from the repo root:

```bash
deploy/hetzner/gen-certs.sh 127.0.0.1 --out crates/pbr-server/tests/fixtures/tls
```

Those fixtures carry the loopback address as their SAN and are committed.
