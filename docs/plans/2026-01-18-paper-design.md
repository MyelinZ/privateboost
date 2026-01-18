# Paper Design: Privacy-Preserving Federated XGBoost

**Date:** 2026-01-18
**Target:** ArXiv preprint
**Status:** Design approved, ready for implementation

## Overview

Academic paper presenting privateboost — a privacy-preserving federated XGBoost system for the extreme non-IID setting where each client holds exactly one data sample.

## Key Claims

1. **First federated XGBoost for one-sample-per-client cross-device setting**
2. **Raw data never leaves client** — shareholders see shares, aggregator sees sums
3. **No client-to-client coordination** — star topology through shareholders
4. **Suitable for medical applications** — patients participate directly without institutional intermediaries
5. **Empirically validated** — 98% gain retention, <2% accuracy loss on medical datasets

## Folder Structure

```
paper/
├── main.typ           # Single file (flat structure)
├── figures/           # Copied from docs/figures/
│   ├── architecture.png
│   ├── gain_retention.png
│   ├── learning_curves.png
│   └── dropout_resilience.png
├── references.bib
└── build/             # Generated PDFs (gitignored)
```

## Paper Structure

### Abstract (~150 words)

We present privateboost, a privacy-preserving federated XGBoost system for the extreme non-IID setting where each client holds exactly one data sample. This cross-device setting arises naturally in medical applications where each patient controls their own record. Our protocol uses m-of-n Shamir secret sharing with commitment-based anonymous aggregation: raw feature values never leave the client, shareholders see only secret shares, and the aggregator reconstructs only aggregate gradient sums. Unlike SecAgg-based approaches requiring client-to-client coordination for pairwise key agreement, our architecture uses a fixed set of stateless shareholders — clients communicate only with shareholders, never with each other. We evaluate on UCI medical datasets, demonstrating 98% split gain retention compared to centralized XGBoost with <2% accuracy degradation, while maintaining resilience to client dropout. We discuss extensions including differential privacy and k-anonymous tree structures.

### 1. Introduction (~1 page)

**Opening hook:**
- Federated learning enables ML on distributed data, but most work assumes clients hold multiple samples (cross-silo)
- In medical settings, "federated" typically means hospitals or institutions each manage patient cohorts — the institution is still a trusted aggregation point
- True cross-device medical FL — where individual patients control their own records without institutional intermediaries — remains underexplored
- This setting is inherently extreme non-IID: one patient = one sample
- Existing approaches have drawbacks: SecAgg requires client-to-client coordination which is problematic in cross-device settings where clients are intermittently available; HE is computationally expensive

**Problem statement:**
- Train XGBoost across n clients where each holds exactly one sample
- No institutional intermediary — patients participate directly
- Requirements: (1) raw data never leaves client, (2) no single point of trust, (3) resilient to client dropout

**Our contribution:**
1. Protocol using m-of-n Shamir sharing with commitment-based anonymity
2. Three-party architecture: clients → shareholders → aggregator
3. Only aggregator sees sums; shareholders see shares; clients see raw data
4. Empirical validation on medical datasets with dropout resilience analysis

**Paper outline:**
- §2 Related Work, §3 Protocol, §4 Security Analysis, §5 Experiments, §6 Future Work, §7 Conclusion

### 2. Related Work (~0.5-1 page)

**Secure Aggregation:**
- SecAgg (Bonawitz et al., 2017): Pairwise masking for federated averaging
- Requires client-to-client key agreement; challenging when clients are intermittently online
- Designed for neural network gradient aggregation, not tree-based methods

**Federated Gradient Boosting:**
- SecureBoost (Cheng et al., 2021): Vertical FL where features split across parties
- Federated Forest: Assumes institutional clients with local datasets
- Neither addresses one-sample-per-client cross-device setting

**Secret Sharing in ML:**
- Shamir sharing used in MPC frameworks (SPDZ, ABY)
- Typically for secure inference, less explored for federated training
- Our contribution: applying threshold sharing to gradient histogram aggregation

**Gap we fill:**
- Cross-device FL (one sample per client) + tree-based methods + no client coordination

**Positioning Table:**

| Prior Work | Coordination Model | Our Advantage |
|------------|-------------------|----------------|
| SecAgg | Client-to-client pairwise key agreement | No client-to-client communication; star topology through shareholders |
| SecureBoost | Vertical FL across institutions | Horizontal FL, one-sample-per-client, true cross-device |

### 3. Protocol (~2 pages)

**3.1 System Model:**
- n clients, each holding one sample (x_i, y_i)
- k shareholders (e.g., 3), m-of-k threshold (e.g., 2-of-3)
- 1 aggregator that coordinates training
- Trust model: shareholders don't collude, aggregator is honest-but-curious

**3.2 Building Blocks:**
- Shamir secret sharing: split value into k shares, any m reconstruct
- Additive homomorphism: sum of shares = share of sum
- Commitments: SHA256(round_id || client_id || nonce) — unlinkable across rounds

**3.3 Protocol Phases:**

| Phase | Clients do | Shareholders do | Aggregator does |
|-------|-----------|-----------------|-----------------|
| 1. Statistics | Share x, x² with commitments | Store by commitment | Reconstruct mean/var → define bins |
| 2. Gradients (per depth) | Compute g,h; share with commitments | Sum shares for requested commitments | Reconstruct sums → find best splits |
| 3. Finish tree | Update predictions | Clear round state | Add tree to ensemble |

**3.4 Figures:**
- `architecture.png` — overall system diagram
- Consider adding: protocol flow showing message sequence

### 4. Security Analysis (~0.5-1 page)

**4.1 Threat Model:**
- Honest-but-curious: all parties follow protocol but try to learn extra information
- Aggregator: sees commitment hashes and aggregate sums only
- Shareholders: see shares, never raw values; cannot link commitments to client IDs
- No single shareholder can reconstruct (requires m of k)

**4.2 Privacy Guarantees:**

| Party | What they see | What they learn |
|-------|---------------|-----------------|
| Client | Own raw data | Nothing new |
| Shareholder | Shares + commitments | Nothing (threshold not met alone) |
| Aggregator | Sums + commitment hashes | Aggregate statistics only |

**4.3 What is revealed (necessary for utility):**
- Split thresholds (required for prediction)
- Aggregate gradient sums per bin
- Number of clients per node (but not which clients)

**4.4 Limitations:**
- 2-of-3 threshold: any 2 colluding shareholders break privacy
- No formal DP guarantees yet
- Aggregator learns client counts per branch: While the aggregator cannot identify *which* clients are in a branch, they know *how many*. In small branches, this narrows the anonymity set. Combined with auxiliary knowledge (e.g., knowing total participant count), this could enable inference. This motivates k-anonymous tree structures (§6 Future Work).
- Tree structure may leak information about feature distributions

### 5. Experiments (~1-1.5 pages)

**5.1 Datasets:**

| Dataset | Samples | Features | Task |
|---------|---------|----------|------|
| Heart Disease (UCI) | 237 | 13 | Binary classification |
| Breast Cancer Wisconsin | 455 | 30 | Binary classification |

**5.2 Experimental Setup:**
- 2-of-3 Shamir threshold
- 10 bins per feature (equal-width)
- Max depth 3, 15 trees
- Baseline: XGBoost with matched hyperparameters

**5.3 Results:**

| Experiment | Figure | Key Finding |
|------------|--------|-------------|
| Split quality | `gain_retention.png` | 98% mean gain retention vs optimal splits |
| Accuracy | `learning_curves.png` | <2% accuracy degradation vs XGBoost |
| Robustness | `dropout_resilience.png` | Stable up to ~30% client dropout |

**5.4 Discussion:**
- Binning causes most accuracy loss (not the secret sharing itself)
- Skewed features (e.g., `oldpeak`) show 90-93% retention — room for improvement
- Dropout resilience comes "for free" from commitment-based aggregation

### 6. Future Work (~0.5 page)

**6.1 Differential Privacy:**
- Add calibrated Laplace noise to gradient sums before reconstruction
- Noise calibration can leverage the aggregated statistics (mean/variance) already computed in the binning phase — no additional privacy cost for calibration
- Per-tree privacy budget (ε = 1.0-2.0 suggested)
- Composition across trees using privacy accounting

**6.2 K-Anonymous Tree Structure:**
- Enforce minimum k clients per leaf/branch during split selection
- Reject splits that would create nodes with fewer than k clients
- Prevents membership inference via tree path analysis
- Connects to existing `min_clients` shareholder enforcement

### 7. Conclusion (~0.25 page)

- Presented privateboost: privacy-preserving federated XGBoost for extreme non-IID
- Key properties: raw data never leaves client, no client-to-client coordination, dropout resilient
- Validated on medical datasets: 98% gain retention, <2% accuracy loss
- Enables true cross-device medical FL where patients participate directly without institutional intermediaries

## References to Include

- Bonawitz et al., 2017 — SecAgg
- Cheng et al., 2021 — SecureBoost
- Shamir, 1979 — Secret sharing
- Chen & Guestrin, 2016 — XGBoost
- McMahan et al., 2017 — Federated learning

## Additional Experiments (deferred)

For potential venue submission later:
- One more dataset (larger, e.g., Adult Income)
- Communication overhead measurements
- Runtime comparison vs vanilla XGBoost
