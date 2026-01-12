# Shamir Secret Sharing with Commitments

> Design for privacy-preserving aggregation with 2-of-3 threshold and client-set verification

## Overview

Replace additive secret sharing with Shamir 2-of-3 threshold scheme. Add commitment protocol so shareholders can verify consistent client sets without revealing client identities to the aggregator.

## Design Decisions

| Component | Decision |
|-----------|----------|
| Secret sharing | Shamir 2-of-3 threshold, vectorized |
| Commitment | `SHA256(round_id \|\| client_id \|\| nonce)` |
| Shareholder selection | Aggregator picks 2 with largest overlap |
| Privacy enforcement | Shareholders reject requests for < N clients |

## Commitment Scheme

**Purpose:** Verify same clients sent to multiple shareholders without revealing identity.

```python
commitment = SHA256(round_id || client_id || nonce)
```

- `round_id`: 8-byte integer
- `client_id`: Client's internal identifier (never sent to aggregator)
- `nonce`: 32-byte random, fresh each round

**Properties:**
- Opaque to aggregator (can't reverse hash)
- Same client + round + nonce → same commitment
- Fresh nonce prevents cross-round correlation

## Shamir Secret Sharing

**Polynomial:** For value `v`, create `f(x) = v + a*x` where `a` is random.

**Shares:** Evaluate at x = 1, 2, 3:
```
ShareHolder 1: (1, v + a)
ShareHolder 2: (2, v + 2a)
ShareHolder 3: (3, v + 3a)
```

**Reconstruction:** Lagrange interpolation from any 2 points:
```python
def reconstruct(shares):
    (x1, y1), (x2, y2) = shares
    return y1 * (-x2) / (x1 - x2) + y2 * (-x1) / (x2 - x1)
```

**Vectorized:** Each value in a vector gets independent random coefficient.

**Linearity:** Sum of shares = share of sum. Shareholders sum before sending.

## Protocol Flow

```
PHASE 1: CLIENT SUBMISSION
  Client:
    commitment = SHA256(round_id || client_id || nonce)
    shares = shamir_share_vector(values, n_parties=3)
  Client → ShareHolder_i: (round_id, commitment, shares[i])

PHASE 2: COMMITMENT COLLECTION
  Aggregator → All ShareHolders: "List commitments for round R"
  ShareHolder → Aggregator: [commitment_1, commitment_2, ...]

PHASE 3: SHAREHOLDER SELECTION
  Aggregator:
    overlap_01 = SH0_commitments ∩ SH1_commitments
    overlap_02 = SH0_commitments ∩ SH2_commitments
    overlap_12 = SH1_commitments ∩ SH2_commitments
    Select pair with largest overlap

PHASE 4: SHARE REQUEST
  Aggregator → Selected ShareHolders: "Sum shares for [c1, c2, ...]"
  ShareHolder:
    IF len(requested) < N: REJECT
    ELSE: sum shares, send (x, summed_y)

PHASE 5: RECONSTRUCTION
  Aggregator: shamir_reconstruct(sum_from_SH_A, sum_from_SH_B)
```

## Minimum N Enforcement

Shareholders refuse requests for fewer than N commitments:

```python
def get_shares_for_commitments(self, round_id, requested_commitments):
    if len(requested_commitments) < self.min_clients:
        raise ValueError(f"Minimum {self.min_clients} required")
    # Sum and return
```

This prevents aggregator from isolating individual clients.

## Security Properties

| Attack | Mitigation |
|--------|------------|
| Request single client | Min N enforcement by shareholders |
| Reverse commitment | SHA256 preimage resistance |
| Cross-round tracking | Fresh nonce each round |
| 1 shareholder collusion | Need 2 shares for Shamir |

**Trust assumption:** Security breaks if aggregator colludes with 2 shareholders.

## Code Changes

```
src/privateboost/
├── crypto.py        # Add shamir_share_vector, shamir_reconstruct_vector, compute_commitment
├── messages.py      # Add ShamirShare, CommittedStatsShare, CommittedGradientShare
├── client.py        # Generate commitment, use Shamir
├── shareholder.py   # Store by commitment, enforce min N, get_commitments()
└── aggregator.py    # Select shareholders, request by commitment, reconstruct
```

## Round ID Scheme

```python
STATS_ROUND = 0
HISTOGRAM_ROUND = 1
GRADIENT_ROUND_BASE = 1000  # tree_idx * 100 + depth
```
