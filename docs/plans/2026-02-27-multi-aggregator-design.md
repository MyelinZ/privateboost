# Multi-Aggregator Design

## Goal

Extend privateboost so that no single aggregator is trusted. Multiple aggregators (2f+1) independently compute results, and shareholders run majority vote to reach consensus. This defends against both malicious aggregators and single points of failure.

## Threat Model

Today a single malicious aggregator can:

1. **Lie about splits** — announce a different split than what it computed
2. **Selective reconstruction** — choose commitment subsets that isolate individual client values
3. **Lie about bin edges** — send clients fake bins to concentrate a target client's data
4. **Manipulate round progression** — skip or reorder rounds to extract extra information

The multi-aggregator design prevents all of these by requiring majority agreement.

## Topology

All addresses are configured out-of-band at deploy time. No service provides addresses for any other — this prevents a compromised service from redirecting traffic to servers it controls.

Every participant is configured with:
- Coordinator address
- Shareholder addresses
- Aggregator addresses

```
                    Admin
                      │
                      ▼
               ┌─────────────┐
               │ Coordinator  │   run config + lifecycle only
               └─────────────┘
                 ▲    ▲    ▲
                 │    │    │
          SH(1..n)  AGG(1..2f+1)  Clients
```

Data flow:
```
Clients ──shares──▶ ShareHolders ◀──get_sums── Aggregators
                         │                          │
                         │◀──submit_result──────────┘
                         │
                         ├── majority vote
                         │
                    Clients (fetch consensus bins/splits)
```

Clients never talk to aggregators. Aggregators are pure stateless compute nodes.

## Roles

**Coordinator** — Lightweight run registry. Admin creates/cancels runs here. All other participants poll it for run config and lifecycle. Low-trust: the worst it can do is disrupt availability (refuse to create runs), not compromise privacy. Does not hold any addresses.

**ShareHolders** — Store shares (unchanged). New responsibilities: drive round progression, accept results from aggregators, run majority vote, serve consensus results to clients, poll coordinator for run lifecycle to clean up stale runs.

**Aggregators (2f+1)** — Stateless compute. Fetch commitment sets and share sums from shareholders, reconstruct via Lagrange interpolation, compute bins/splits, submit results back to shareholders. Can crash and restart — they catch up by checking which step shareholders are on. Untrusted: could be run by third parties.

**Clients** — Submit shares to shareholders, poll shareholders for current step, fetch consensus results (bins, splits, leaf values) from shareholders. Configured with shareholder addresses out-of-band.

## Shareholder Freeze Mechanism

For consensus to work, all aggregators must reconstruct from the same set of commitments. Since shares arrive asynchronously, two aggregators polling at different times would see different commitment sets.

Solution: shareholders freeze the commitment set once enough shares arrive.

Per step (stats round or gradient round/depth), each shareholder transitions:

```
ACCEPTING → FROZEN
```

- **ACCEPTING**: new shares are stored. Commitment set is growing.
- **FROZEN**: triggered when `len(commitments) >= target`. No new shares accepted for this step. All aggregators see the same fixed commitment set.

Late-arriving clients get an error and know they missed the window.

Since all shareholders freeze based on the same threshold, and the `select_shareholders` overlap logic is deterministic (same inputs → same overlap → same selected group), all aggregators compute from identical data.

## Consensus Voting

All aggregators run the same Rust binary, so identical inputs produce bit-identical outputs (same floating point evaluation order). Voting uses exact comparison.

1. Each aggregator computes a result for the current step (bin config or split decisions)
2. Each aggregator submits the result to shareholders: `submit_result(run_id, step_id, aggregator_id, result)`
3. Each shareholder collects results from all aggregators. Once 2f+1 are received, group by value. If >f agree, that's the consensus.
4. If no majority — the run is flagged as disputed. Coordinator is notified.

## Shareholder State Machine

Shareholders drive round progression. They advance to the next step once consensus is reached for the current step.

```
COLLECTING_STATS
  → FROZEN_STATS (target reached)
  → STATS_CONSENSUS (aggregators voted on bins)
  → COLLECTING_GRADIENTS(round=0, depth=0)
    → FROZEN_GRADIENTS
    → SPLITS_CONSENSUS (aggregators voted on splits)
    → COLLECTING_GRADIENTS(round=0, depth=1)
      → ...
    → ROUND_COMPLETE(round=0) (aggregators voted on leaf values)
  → COLLECTING_GRADIENTS(round=1, depth=0)
    → ...
  → TRAINING_COMPLETE
```

Clients poll shareholders for the current step. Before submitting gradients for the next step, they fetch the consensus result from the previous step to update their local state (node assignments or predictions).

## Protocol Flow

### Run creation
1. Admin calls `create_run(config)` on coordinator
2. Shareholders, aggregators, clients poll coordinator and discover the run

### Stats round
1. Clients submit stats shares to shareholders
2. Each shareholder freezes when it hits the target count
3. All aggregators query `get_stats_commitments` on all shareholders → get frozen sets
4. Each aggregator runs `select_shareholders` → `get_stats_sum` → Lagrange reconstruction → computes bin configs
5. Each aggregator submits bin config to shareholders via `submit_result`
6. Shareholders vote, store consensus bins, advance to gradient collection
7. Clients fetch consensus bins from shareholders

### Gradient rounds (per tree, per depth)
1. Clients fetch consensus splits from the previous step, update node assignments
2. Clients submit gradient shares to shareholders
3. Shareholders freeze when target reached
4. Aggregators reconstruct histograms, compute splits
5. Aggregators submit splits to shareholders
6. Shareholders vote, store consensus splits, advance

### Round completion
1. After max depth, aggregators compute leaf values and submit
2. Shareholders vote, store consensus tree
3. Clients fetch the tree, update predictions
4. Next round begins

### Run cleanup
Shareholders periodically poll the coordinator for active runs. Any local run not in the active set gets cleaned up. This replaces the `cancelled` session bookkeeping entirely.

## Session Cleanup (Original Problem)

The cancelled sessions HashSet is eliminated. Shareholders learn run lifecycle from the coordinator:
- Active runs → accept shares
- Cancelled/completed runs → drop local state
- Unknown run IDs in incoming requests → reject

No separate bookkeeping, no unbounded sets, no zombie sessions.

## Security Properties

All existing properties are preserved, plus:

- **No trusted aggregator**: any f of 2f+1 aggregators can be malicious without affecting correctness
- **No trusted coordinator**: coordinator cannot compromise privacy (doesn't hold addresses or shares), can only disrupt availability
- **Threshold security**: any m-1 colluding shareholders still learn nothing (unchanged)
- **Aggregate-only**: aggregators see only sums, never individual values (unchanged)
- **Deterministic verification**: same binary + same inputs = bit-identical results, so honest aggregators always agree

## What Changes from Today

| Component | Current | New |
|-----------|---------|-----|
| Aggregator count | 1 | 2f+1 |
| Client talks to | Aggregator + Shareholders | Shareholders only |
| Round progression | Aggregator drives | Shareholders drive |
| Bin/split source | Aggregator serves to clients | Shareholders serve consensus to clients |
| Session lifecycle | `cancelled` HashSet on shareholder | Poll coordinator |
| Aggregator state | Stateful (owns training loop) | Stateless (compute on demand) |
| Python servers | Aggregator + Shareholder | Removed (Rust only) |

## Future Considerations

- **Aggregator authentication**: aggregators should sign their submitted results so shareholders can verify identity and prevent spoofing
- **Shareholder-to-shareholder communication**: shareholders may need to sync freeze status for consistency
- **Client retry**: protocol for clients that miss the freeze window
- **Coordinator HA**: coordinator is a single point for availability (not security) — could be replicated
