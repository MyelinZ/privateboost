# PrivateBoost demo app

A minimal Flutter app that participates in a PrivateBoost federated-XGBoost
training round from a real device, authenticated with Firebase.

It is a thin UI over the `app/rust/` flutter_rust_bridge crate, which wraps the
`pbr-client` protocol library. Dart owns sign-in and the "Contribute" button;
all crypto/protocol/networking runs in the Rust `.so`.

## What the end-to-end demo proves

Verified on the `Pixel_7` Android emulator (x86_64) against a host-run cluster:

1. The full Rust protocol stack (Shamir sharing, tonic gRPC, the federated
   client) cross-compiles for Android and loads as `librust_lib_privateboost_app.so`.
2. The app signs in with **real Firebase email/password** (`firebase_ui_auth`),
   minting a genuine Firebase ID token on-device.
3. Tapping **Contribute** drives the native client through enroll → secret-share
   submission to the shareholders.
4. The aggregator **verifies the real Firebase token** — it fetched Google's
   live `securetoken` x509 signing certificates at startup
   (`issuer = https://securetoken.google.com/<GOOGLE_CLOUD_PROJECT>`,
   `audience = <GOOGLE_CLOUD_PROJECT>`).
5. A full session trains to completion; the app shows `completed: true`.

Evidence: `demo/01-login.png`, `demo/02-signed-in.png`, `demo/03-completed.png`
(`completed: true, rounds submitted: 4`), and `demo/aggregator-session.log`
(`stats phase complete, bins defined n_features=13` -> `tree finished trees=1..3`
-> `session completed`).

## Firebase configuration

`android/app/google-services.json` is git-ignored, so supply one before
building: register the Android package id `dev.pboost.pboost_app` in a Firebase
project with email/password sign-in enabled, then drop that project's
`google-services.json` at `android/app/google-services.json`. The aggregator
must be pointed at the same project, since its `[auth]` `issuer`
(`https://securetoken.google.com/<project-id>`) and `audience` (`<project-id>`)
are what its token check compares against; `deploy/emulator/aggregator.toml`
shows the shape.

The package id carries the `dev.pboost` prefix because the maintainers register
it in an existing project of theirs rather than provisioning a new one. Nothing
depends on that prefix; a different project can register the same id.

## Running the demo

Prereqs: Flutter, `cargo-ndk`, an Android emulator (`Pixel_7`), the Android NDK.

```bash
# 1. Boot the emulator
emulator -avd Pixel_7 -no-window -no-audio &
adb wait-for-device

# 2. Map the emulator's localhost to the host cluster ports
for p in 42800 42801 42802 42803; do adb reverse tcp:$p tcp:$p; done

# 3. Start the cluster with the emulator configs (aggregator verifies real
#    Firebase tokens). Run from the repo root:
cargo build --release --bin pbr-server
for x in 1 2 3; do
  ./target/release/pbr-server --role shareholder --config deploy/emulator/shareholder-$x.toml &
done
./target/release/pbr-server --role aggregator --config deploy/emulator/aggregator.toml &

# 4. Build + install the app (x86_64 for the emulator)
cd app && ./build_android.sh
adb install -r build/app/outputs/flutter-apk/app-debug.apk

# 5. The aggregator boots hosting no session — schedule the single-client
#    demo one now, right before launching, so its submission window is
#    still open when you tap Contribute. `just demo-session` would inherit
#    pbr-admin's 15-tree/depth-3 reference defaults, too slow for a phone
#    demo, so this calls pbr-admin directly with the small emulator shape
#    (3 trees, depth 2, 5 bins) instead (the committed dev admin_token is
#    `dev-admin-token`, see deploy/README.md).
#    heart_disease is the app's bundled default dataset; 1/1 matches
#    deploy/emulator/shareholder-N.toml's committed anonymity floor.
#    Note: the session's give-up clock starts at creation; with the
#    2-minute submission window below, you have ~20 minutes of total budget.
#    Create the session before slow first-time app setup only if you'll be
#    signed in and ready to tap Contribute within that window, otherwise
#    create it after the app is installed and you're signed in.
PBR_ADMIN_TOKEN=dev-admin-token cargo run -p pbr-admin -- create-session \
  --aggregator http://127.0.0.1:42800 \
  --dataset heart_disease \
  --min-clients 1 --target-clients 1 --window-ms 120000 \
  --trees 3 --depth 2 --bins 5

# 6. Launch, register/sign in, tap Contribute promptly — the round loop is
#    already waiting for a submission.
adb shell am start -n dev.pboost.pboost_app/dev.pboost.privateboost_app.MainActivity
```

## Pilot build against a public VM (TLS)

The demo above runs debug builds against the loopback emulator cluster over
cleartext http. A pilot targets the public aggregator VM over TLS instead. Two
things differ from the emulator flow: the endpoint is passed at build time, and
the deployment CA is bundled as the app's trust root.

```bash
# 1. Copy the deployment CA (produced by deploy/hetzner/gen-certs.sh) over the
#    committed placeholder asset. Paths are relative to app/.
cp ../deploy/hetzner/secrets/ca.crt assets/pilot-ca.pem

# 2. Build a release APK pinned to the VM's IP (the same address gen-certs.sh
#    signed into the server certificate's SAN).
flutter build apk --release --dart-define=PBR_AGG_ENDPOINT=https://<VM_IP>:42800
```

The `https` scheme makes the app load `assets/pilot-ca.pem` and pin it as the
sole TLS trust root for the aggregator and shareholders. A release built without
copying the real `ca.crt` in fails at participation time (the placeholder is not
a certificate) rather than falling back to the system trust store. The default
`http` endpoint and cleartext traffic only work in debug/emulator builds, so no
CA copy is needed for the demo above.

## Scope / caveats

- **This is a functional vertical-slice demo, not a privacy demonstration.**
  The demo session is scheduled with `min_clients = target_clients = 1` (see
  step 5 above) so one device can drive a visible round; that provides no
  anonymity set. A real deployment schedules sessions with `min_clients` set
  accordingly (each shareholder's own committed floor is the hard enforcement
  point — see `deploy/README.md`'s "Admin plane" section).
- Cleartext HTTP (`usesCleartextTraffic`) over `adb reverse` is confined to the
  debug manifest -- fine for the localhost emulator demo. A network pilot uses a
  TLS `https` endpoint with a pinned CA instead (see "Pilot build against a
  public VM").
- Single client contributing one record trains a trivial model; the point is
  that the authenticated end-to-end path completes, not the model quality.

The "Running the demo" steps above drive the foreground path (the Contribute
button). Background participation via FCM push and WorkManager is documented
next.

## Background participation via FCM silent push

Verified end-to-end on the `Pixel_7` emulator (evidence in `demo/m3-*`):

- The device registers its FCM token on sign-in via the JWT-gated `RegisterDevice`
  RPC, and enrolls in a session — via the join tap, or any participation — via
  `EnrollSession`.
- A process-wide notify tick runs every 60 s on the aggregator. While a session
  has open work (stats or training phase), the tick sends a **real FCM silent
  (data) push** — authenticated with the machine's gcloud ADC (no
  service-account key), delivered through Google — to every device enrolled in
  that session whose per-account floor has elapsed (`[fcm] interval_minutes`;
  the emulator config sets 2 for a fast demo, production defaults to 15). The
  aggregator logs `round_open notify tick` with `notified=N` for the devices it
  sent to that tick.
- With the app **backgrounded** (`FLTFireMsgReceiver: broadcast received`), the
  `@pragma('vm:entry-point')` handler wakes a background isolate, re-inits
  RustLib + Firebase, gets a fresh ID token, and participates — training a full
  session with **no foreground and no tap**. App-side log:
  `flutter : privateboost fcm wake: completed=true rounds=4`, and the aggregator
  drives `stats phase complete` → `tree finished trees=1..3` → `session completed`.
- Each participation records mobile-side telemetry to a per-trigger shard under
  `<app_dir>/` (`telemetry.<trigger>.jsonl`, merged on read) — see
  `demo/m3-telemetry.jsonl` (`trigger:"push"` and `trigger:"foreground"`
  entries with per-round timings).
- A WorkManager periodic task (`pbr-contribute`, 15 min, network-constrained) is
  registered on sign-in as a fallback and runs the same background participation;
  it fires on Android's own schedule.

Run: enable `[fcm] project_id = "<GOOGLE_CLOUD_PROJECT>"` in the aggregator config
(uses ADC — `gcloud auth application-default login` as a project member), then
follow the "Running the demo" steps above. Push delivery to a backgrounded app
requires Google Play services (present on the Pixel_7 AVD).

### Scope note
Single-client `min_clients=1` demo (functional, not privacy). iOS is deferred to
a macOS build environment. The gcloud ADC stands in for a production service
account; both are supported by the `gcloud-sdk`-based sender.
