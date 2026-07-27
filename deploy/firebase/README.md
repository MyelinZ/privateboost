# Firestore rules

`firestore.rules` is the security ruleset for the Firebase project named by
`GOOGLE_CLOUD_PROJECT` (set it in `.env`; see `.env.example`). Deploy it from a
devenv shell, where that variable is exported:

```
firebase deploy --only firestore:rules --project "$GOOGLE_CLOUD_PROJECT"
```

The three collections:

- `roundMetrics` — per-round metrics keyed by uid. Own-uid create, own-uid
  read, append-only.
- `paperSimRoundMetrics` — this app's per-round systems telemetry, written under
  Firebase Auth. Same own-uid create / own-uid read / append-only shape.
- `paperSimTreeMetrics` — per-tree quality metrics. **Client-locked** (`allow
  read, write: if false`). Only the aggregator's admin service-account writes
  it, and only ADC reads it back for analysis; both bypass security rules, so
  no authenticated client ever needs access. Locking it out keeps a signed-in
  device from reading or forging quality numbers.

A default-deny `match /{document=**}` closes everything else.
