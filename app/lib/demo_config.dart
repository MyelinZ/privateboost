/// Shared demo constants: one real row from
/// `crates/privateboost/tests/data/heart_disease.csv` (13 features: age, sex, cp,
/// trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal),
/// baked in for the "Contribute" demo. Referenced only by
/// `demo_config_test.dart`; no participation path wires it in.
const List<double> demoRow = [
  52.0, 1.0, 4.0, 112.0, 230.0, 0.0, 0.0, 160.0, 0.0, 0.0, 1.0, 1.0, 3.0,
];
const double demoLabel = 1.0;

/// The aggregator endpoint. Defaults to the loopback demo cluster
/// (`deploy/emulator/aggregator.toml`, the only config that verifies the real
/// Firebase ID tokens the app sends; the `deploy/local` dev issuer would reject
/// them), reached from the emulator via an `adb reverse` mapping of this port
/// to the host cluster (see `app/README.md`).
///
/// A pilot build overrides it with the public VM address:
/// `--dart-define=PBR_AGG_ENDPOINT=https://<VM_IP>:42800`. The scheme selects
/// the transport: `https` pins the bundled CA (see `pilot_ca.dart`), `http`
/// stays cleartext. This must stay `const`: `String.fromEnvironment` only reads
/// the `--dart-define` value in a const context; a non-const use silently keeps
/// the default in release builds.
const String aggEndpoint = String.fromEnvironment(
  'PBR_AGG_ENDPOINT',
  defaultValue: 'http://127.0.0.1:42800',
);

/// This device's slice of the baked train split: it contributes part
/// [batchId] (0-indexed) of a fleet split into [batchCount] equal slices (see
/// `dataset.dart`). A fleet build sets `PBR_BATCH_COUNT` to the device count and
/// each device a distinct `PBR_BATCH_ID`; the default `count = 1` makes the
/// single slice the whole set, so the emulator/default build still works. Both
/// must stay `const`: `int.fromEnvironment` only reads the `--dart-define` value
/// in a const context (see `aggEndpoint`).
const int batchId = int.fromEnvironment('PBR_BATCH_ID', defaultValue: 0);
const int batchCount = int.fromEnvironment('PBR_BATCH_COUNT', defaultValue: 1);

/// Build-time override for the foreground poll interval, in seconds. This sets
/// only the *cadence*; whether the poller runs at all is the in-app
/// `foregroundPollingProvider` switch (`providers/settings.dart`), off by
/// default. 0 (the default, no `--dart-define`) means "no override": the
/// effective interval is then [kDefaultForegroundPollSeconds]. A pilot build
/// tightens it with e.g. `--dart-define=PBR_FG_POLL_SECONDS=20` for fast
/// session testing. Must stay `const`: `int.fromEnvironment` only reads the
/// `--dart-define` value in a const context.
const int fgPollSeconds = int.fromEnvironment('PBR_FG_POLL_SECONDS', defaultValue: 0);

/// Foreground poll cadence used when no `--dart-define` override is set
/// ([fgPollSeconds] == 0).
const int kDefaultForegroundPollSeconds = 120;

/// The cadence the foreground poller ticks at: the [fgPollSeconds] build
/// override when positive, else [kDefaultForegroundPollSeconds]. Pure so both
/// branches are unit-testable without a `--dart-define`.
int effectiveForegroundPollSeconds(int override) =>
    override > 0 ? override : kDefaultForegroundPollSeconds;

/// The active foreground poll cadence for this build.
int get foregroundPollSeconds => effectiveForegroundPollSeconds(fgPollSeconds);
