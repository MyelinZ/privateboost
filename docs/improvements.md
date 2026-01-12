# privateboost Improvements Design Document

> Privacy-Preserving Federated XGBoost: Addressing Challenges and Alternatives

## Executive Summary

This document proposes improvements to privateboost for cross-device federated learning where each client holds one sample. The current additive secret sharing approach achieves ~98% gain retention but has limitations around collusion resistance, information leakage, and scalability. We propose a phased approach: (1) add differential privacy for formal guarantees, (2) improve binning for accuracy, (3) strengthen the shareholder model with threshold secret sharing, and (4) add verification for Byzantine robustness.

---

## 1. Current State Analysis

### 1.1 Architecture

```
Clients (n)        ShareHolders (3)      Aggregator (1)
    │                    │                    │
    ├──shares[1]────────>│                    │
    ├──shares[2]────────>│                    │
    ├──shares[3]────────>│                    │
    │                    ├────sums──────────>│
    │                    │                    ├──> Statistics
    │<───────────────────┼────────────────────┤    Splits
                         │                    │    Leaf values
```

### 1.2 Security Properties

| Property | Current Status |
|----------|----------------|
| Individual value privacy | ✓ Protected by secret sharing |
| Collusion resistance | ✗ Any 2 of 3 shareholders can reconstruct |
| Aggregate privacy | ✗ Exact sums revealed to aggregator |
| Byzantine robustness | ✗ No verification of shares |
| Communication efficiency | ~ O(clients × shareholders × features × bins) |

### 1.3 Known Limitations

1. **Equal-width binning** loses 9-15% gain on skewed features
2. **3-party secret sharing** vulnerable to 2-party collusion
3. **No differential privacy** enables composition attacks
4. **Honest-but-curious** assumption only
5. **Single aggregator** is trust bottleneck and DoS target

---

## 2. Threat Model

### 2.1 Adversary Capabilities

We consider three adversary types:

| Adversary | Controls | Goal |
|-----------|----------|------|
| Curious Server | Aggregator | Learn individual feature values |
| Colluding Servers | 2+ ShareHolders | Reconstruct client data |
| Malicious Client | Own shares | Corrupt model or extract others' data |

### 2.2 Security Goals

1. **Input Privacy**: No party learns individual client values
2. **Output Privacy**: Aggregates reveal minimal information (differential privacy)
3. **Integrity**: Malicious parties cannot corrupt training undetected
4. **Availability**: Protocol tolerates dropout and partial failures

---

## 3. Proposed Improvements

### 3.1 Phase 1: Differential Privacy for Aggregates

**Problem**: Exact aggregate statistics enable composition attacks. An adversary knowing n-1 samples can compute the nth exactly.

**Solution**: Add calibrated Laplace noise to all reconstructed sums before use.

#### 3.1.1 Mechanism

```python
class DPAggregator(Aggregator):
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget = PrivacyAccountant(epsilon, delta)

    def _add_noise(self, value: float, sensitivity: float, query_name: str) -> float:
        """Add Laplace noise calibrated to sensitivity and remaining budget."""
        eps_query = self.privacy_budget.allocate(query_name)
        scale = sensitivity / eps_query
        return value + np.random.laplace(0, scale)

    def _reconstruct_stats(self) -> dict:
        """Reconstruct statistics with DP noise."""
        raw_stats = super()._reconstruct_stats()

        # Sensitivity analysis:
        # - sum(x): bounded by feature range, say [-1e6, 1e6] → sensitivity = 2e6
        # - sum(x²): bounded by range² → sensitivity = 1e12
        # With normalization to [0,1], sensitivities become 1 and 1

        noisy_stats = {}
        for feature_idx, stats in raw_stats.items():
            noisy_stats[feature_idx] = StatsSums(
                sum_x=self._add_noise(stats.sum_x, sensitivity=1.0, query_name=f"sum_x_{feature_idx}"),
                sum_x2=self._add_noise(stats.sum_x2, sensitivity=1.0, query_name=f"sum_x2_{feature_idx}")
            )
        return noisy_stats
```

#### 3.1.2 Privacy Budget Allocation

| Phase | Queries | Suggested Budget |
|-------|---------|------------------|
| Statistics round | 2 per feature (sum_x, sum_x²) | 20% |
| Histogram round | 1 per feature | 10% |
| Gradient rounds | 2 per feature per depth level | 60% |
| Leaf computation | 2 per leaf | 10% |

For a tree of depth 3 with 13 features:
- Statistics: 26 queries
- Gradient rounds (3 levels): 78 queries
- Leaves (8 nodes): 16 queries
- Total: ~120 queries per tree

With ε=2.0 total budget and 15 trees: ε_per_query ≈ 0.001

#### 3.1.3 Trade-offs

| ε (epsilon) | Privacy | Accuracy Impact |
|-------------|---------|-----------------|
| 0.1 | Strong | Significant noise, may hurt convergence |
| 1.0 | Moderate | Small accuracy loss (~1-2%) |
| 10.0 | Weak | Negligible noise |

**Recommendation**: Start with ε=2.0 per tree, evaluate accuracy impact empirically.

---

### 3.2 Phase 2: Improved Histogram Binning

**Problem**: Equal-width bins from μ±3σ perform poorly on skewed distributions.

**Solution**: Adaptive binning strategies that better capture feature distributions.

#### 3.2.1 Option A: More Bins

Simple approach - increase from 10 to 32-64 bins:

```python
DEFAULT_N_BINS = 32  # was 10

# Communication cost increases linearly
# 32 bins → 3.2x more data per feature
# But gain retention improves to ~99.5%
```

**Trade-off**: Linear increase in communication for diminishing accuracy returns.

#### 3.2.2 Option B: Federated Quantile Sketches

Compute approximate quantiles privately using mergeable sketches:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class KLLSketch:
    """Space-efficient quantile sketch (Karnin-Lang-Liberty)."""
    k: int  # accuracy parameter
    compactors: list[list[float]]

    @classmethod
    def from_value(cls, value: float, k: int = 200) -> "KLLSketch":
        """Create sketch from single value."""
        return cls(k=k, compactors=[[value]])

    def merge(self, other: "KLLSketch") -> "KLLSketch":
        """Merge two sketches (associative, commutative)."""
        # Combine compactors level by level
        # Compact when level exceeds capacity
        ...

    def quantile(self, q: float) -> float:
        """Estimate q-th quantile."""
        ...


class QuantileAggregator(Aggregator):
    def _compute_quantile_bins(self, feature_idx: int) -> np.ndarray:
        """Compute bin edges from merged quantile sketches."""
        # Clients secret-share their KLL sketch contributions
        # Shareholders sum sketch components
        # Aggregator merges and extracts quantiles

        merged_sketch = self._reconstruct_sketch(feature_idx)
        quantiles = [i / self.n_bins for i in range(1, self.n_bins)]
        edges = [merged_sketch.quantile(q) for q in quantiles]
        return np.array([-np.inf] + edges + [np.inf])
```

**Challenge**: KLL sketches aren't naturally additively homomorphic. Options:
1. **Approximate**: Treat sketch as vector, sum element-wise (loses guarantees)
2. **Interactive**: Multi-round protocol to build sketch securely
3. **Hybrid**: Use DP + wide bins for first tree, refine based on split history

#### 3.2.3 Option C: Domain-Specific Bins

When feature semantics are known, use domain-appropriate ranges:

```python
DOMAIN_BINS = {
    "age": [0, 30, 40, 50, 60, 70, 100],      # Decade boundaries
    "blood_pressure": [0, 90, 120, 140, 180, 300],  # Clinical thresholds
    "binary": [0, 0.5, 1],                     # For 0/1 features
}

def get_bins_for_feature(feature_name: str, stats: StatsSums) -> np.ndarray:
    if feature_name in DOMAIN_BINS:
        return np.array(DOMAIN_BINS[feature_name])
    else:
        return compute_statistical_bins(stats)  # Fall back to μ±3σ
```

**Recommendation**: Combine options - use 32 bins by default, domain-specific when available, quantile sketches for critical skewed features.

---

### 3.3 Phase 3: Strengthen ShareHolder Model

**Problem**: Current 3-of-3 additive sharing means any 2 colluding shareholders break privacy.

**Why not Secure Aggregation?** Google's Secure Aggregation requires O(n²) pairwise key exchanges between clients. This is impractical for cross-device FL:
- Clients can't reach each other directly (NAT, firewalls, cellular networks)
- Coordination overhead is massive at scale (10K clients = 100M key exchanges)
- Dropout handling requires even more client-to-client communication

The shareholder model is actually well-suited for cross-device settings:
- Clients only communicate with a small fixed set of always-online servers
- No peer-to-peer networking required
- Shareholders handle aggregation, reducing client computation

**Solution**: Improve the shareholder model with more parties and threshold schemes.

#### 3.3.1 Increase ShareHolders (5 or 7)

More shareholders = more parties must collude:

```python
# Current: 3 shareholders, all 3 needed to reconstruct
# Any 2 colluding can reconstruct (they have 2/3 of every share)

# Proposed: 5 shareholders, all 5 needed
# Any 4 colluding still missing 1/5 of every share
# Collusion becomes harder (need to compromise more independent parties)

DEFAULT_N_SHAREHOLDERS = 5  # was 3
```

**Trade-off**: Linear increase in client→shareholder communication, but:
- ShareHolders can be geographically/jurisdictionally distributed
- Different cloud providers (AWS, GCP, Azure, on-prem, etc.)
- Legal/organizational separation makes collusion harder

#### 3.3.2 Threshold Secret Sharing (Shamir)

Move from additive (n-of-n) to threshold (t-of-n) sharing:

```python
from scipy.interpolate import lagrange
import numpy as np

def shamir_share(secret: float, threshold: int, n_parties: int) -> list[tuple[int, float]]:
    """
    Split secret into n shares where any t can reconstruct.

    Uses polynomial: f(x) = secret + a1*x + a2*x² + ... + a_{t-1}*x^{t-1}
    Share i = (i, f(i))
    """
    # Random polynomial coefficients (degree t-1)
    coeffs = [secret] + [np.random.uniform(-1e6, 1e6) for _ in range(threshold - 1)]

    def evaluate_poly(x: int) -> float:
        return sum(c * (x ** i) for i, c in enumerate(coeffs))

    # Each party gets (x, f(x)) for x = 1, 2, ..., n
    return [(i, evaluate_poly(i)) for i in range(1, n_parties + 1)]


def shamir_reconstruct(shares: list[tuple[int, float]]) -> float:
    """Reconstruct secret from t or more shares using Lagrange interpolation."""
    xs = [s[0] for s in shares]
    ys = [s[1] for s in shares]

    # Lagrange interpolation at x=0 gives the secret
    poly = lagrange(xs, ys)
    return float(poly(0))


class ShamirClient(Client):
    def __init__(self, *args, threshold: int = 3, n_shareholders: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.n_shareholders = n_shareholders

    def _share_value(self, value: float) -> list[tuple[int, float]]:
        """Create threshold shares of a value."""
        return shamir_share(value, self.threshold, self.n_shareholders)


class ShamirAggregator(Aggregator):
    def __init__(self, threshold: int = 3):
        self.threshold = threshold

    def _reconstruct_sum(self, shareholder_sums: list[tuple[int, float]]) -> float:
        """
        Reconstruct sum from shareholder contributions.

        Key insight: Shamir sharing is linear!
        If share_i(x) = f_x(i) and share_i(y) = f_y(i)
        Then share_i(x+y) = f_x(i) + f_y(i)

        So shareholders can sum their shares, and we reconstruct the sum.
        """
        if len(shareholder_sums) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shareholders")

        return shamir_reconstruct(shareholder_sums[:self.threshold])
```

#### 3.3.3 Benefits of Threshold Sharing

| Property | Additive (n-of-n) | Threshold (t-of-n) |
|----------|-------------------|-------------------|
| Reconstruction | All n required | Any t sufficient |
| Collusion to break | 2 parties | t parties |
| Dropout tolerance | None | n-t dropouts OK |
| Typical config | 3-of-3 | 3-of-5 or 4-of-7 |

**Example configurations**:
- **3-of-5**: Tolerates 2 dropouts, requires 3 colluding to break
- **4-of-7**: Tolerates 3 dropouts, requires 4 colluding to break
- **5-of-9**: High availability and security for critical deployments

#### 3.3.4 ShareHolder Topology

Distribute shareholders across trust boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│                      Clients (mobile devices)                │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │   SH1   │          │   SH2   │          │   SH3   │
   │  (AWS)  │          │  (GCP)  │          │ (Azure) │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐          ┌─────────┐
   │   SH4   │          │   SH5   │
   │(On-prem)│          │  (EU)   │
   └────┬────┘          └────┬────┘
        │                    │
        └─────────┬──────────┘
                  ▼
           ┌───────────┐
           │ Aggregator│
           └───────────┘
```

**Jurisdictional separation**: Place shareholders in different legal jurisdictions to make coordinated subpoenas difficult.

#### 3.3.5 Communication Cost

| Config | Client Upload | ShareHolder→Aggregator |
|--------|---------------|------------------------|
| 3-of-3 additive | 3 × data_size | 3 × sum_size |
| 3-of-5 Shamir | 5 × data_size | 3+ × sum_size |
| 4-of-7 Shamir | 7 × data_size | 4+ × sum_size |

**Trade-off**: ~2x communication increase for significantly better security.

**Recommendation**: Use 3-of-5 Shamir sharing as the new default. This provides dropout tolerance and requires 3 parties to collude (vs 2 currently).

---

### 3.4 Phase 4: Byzantine Robustness

**Problem**: Malicious clients can send invalid shares to corrupt training or probe for information.

**Solution**: Verifiable secret sharing with commitment schemes.

#### 3.4.1 Pedersen Commitments

```python
from dataclasses import dataclass

# Elliptic curve group parameters (use a real library in production)
# G, H are generator points where discrete log relationship is unknown

@dataclass
class PedersenCommitment:
    """C = g^v * h^r where v is value, r is randomness."""
    commitment: bytes  # EC point

    @classmethod
    def commit(cls, value: float, randomness: bytes) -> "PedersenCommitment":
        # C = G * value + H * randomness
        ...

    def verify_sum(self, other_commitments: list["PedersenCommitment"],
                   expected_sum: float, sum_randomness: bytes) -> bool:
        """Verify that commitments sum to expected value."""
        # Product of commitments = commitment to sum
        # C1 * C2 * ... = G * (v1+v2+...) + H * (r1+r2+...)
        ...


class VerifiableClient(Client):
    def submit_verified_shares(self, value: float) -> tuple[list[float], list[PedersenCommitment]]:
        """Generate shares with commitments for verification."""
        shares = secret_share_scalar(value, self.n_shareholders)
        randomness = [secrets.token_bytes(32) for _ in range(self.n_shareholders)]
        commitments = [
            PedersenCommitment.commit(share, rand)
            for share, rand in zip(shares, randomness)
        ]
        return shares, commitments


class VerifyingAggregator(Aggregator):
    def _verify_reconstruction(self, shareholder_sums: list[float],
                                shareholder_commitments: list[PedersenCommitment],
                                expected_total: PedersenCommitment) -> bool:
        """Verify that shareholder sums are consistent with commitments."""
        # Sum of commitments should equal commitment to sum
        ...
```

#### 3.4.2 Zero-Knowledge Range Proofs

Ensure shared values are within valid ranges (prevents overflow attacks):

```python
class RangeProof:
    """Prove that committed value is in [0, 2^n) without revealing it."""

    @classmethod
    def prove(cls, value: float, commitment: PedersenCommitment,
              range_bits: int = 64) -> "RangeProof":
        # Bulletproofs or similar ZK range proof
        ...

    def verify(self, commitment: PedersenCommitment, range_bits: int) -> bool:
        ...
```

#### 3.4.3 Gradient Clipping

Simpler alternative - bound gradient contributions:

```python
class ClippedClient(Client):
    def __init__(self, *args, clip_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_norm = clip_norm

    def _clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradient to bounded L2 norm."""
        norm = np.linalg.norm(gradient)
        if norm > self.clip_norm:
            return gradient * (self.clip_norm / norm)
        return gradient
```

Combined with differential privacy, clipping provides both privacy (bounded sensitivity) and robustness (bounded influence per client).

**Recommendation**: Start with gradient clipping (simple, compatible with DP). Add commitments for high-security deployments.

---

## 4. Alternative Architectures

### 4.1 Homomorphic Encryption Approach

Replace secret sharing with additively homomorphic encryption:

```
Client:     E(g_i) ──────────────────────> Aggregator
                                               │
Aggregator: E(Σg_i) = Π E(g_i) ───────────────┘
                                               │
Key Holder: Decrypt(E(Σg_i)) = Σg_i <──────────┘
```

#### 4.1.1 Implementation with Paillier

```python
from phe import paillier  # python-paillier library

class HEClient:
    def __init__(self, public_key: paillier.PaillierPublicKey):
        self.pk = public_key

    def encrypt_gradient(self, gradient: np.ndarray) -> list:
        """Encrypt gradient vector element-wise."""
        return [self.pk.encrypt(float(g)) for g in gradient]


class HEAggregator:
    def __init__(self, public_key: paillier.PaillierPublicKey):
        self.pk = public_key

    def aggregate(self, encrypted_gradients: list[list]) -> list:
        """Sum encrypted gradients (homomorphic addition)."""
        n_elements = len(encrypted_gradients[0])
        result = [self.pk.encrypt(0)] * n_elements
        for enc_grad in encrypted_gradients:
            for i, enc_val in enumerate(enc_grad):
                result[i] = result[i] + enc_val  # Homomorphic add
        return result


class KeyHolder:
    def __init__(self):
        self.pk, self.sk = paillier.generate_paillier_keypair(n_length=2048)

    def decrypt_sum(self, encrypted_sum: list) -> np.ndarray:
        """Decrypt aggregated result."""
        return np.array([self.sk.decrypt(enc) for enc in encrypted_sum])
```

#### 4.1.2 Trade-offs

| Aspect | Secret Sharing | Homomorphic Encryption |
|--------|----------------|------------------------|
| Computation | Fast (native ops) | Slow (100-1000x) |
| Communication | O(n × parties) | O(n) ciphertexts |
| Ciphertext size | Native floats | ~2KB per value |
| Trust model | Multiple shareholders | Single key holder |
| Collusion | 2+ parties | Key holder only |

**Recommendation**: HE is better for small n with strong trust requirements. Secret sharing scales better for cross-device FL.

### 4.2 Hybrid: Local Trees + Global Ensemble

Reduce communication by training local models:

```python
class LocalTreeClient:
    def __init__(self, features: np.ndarray, target: float):
        self.X = features.reshape(1, -1)
        self.y = np.array([target])
        self.local_stumps: list[DecisionStump] = []

    def train_local_stump(self, residual: float) -> DecisionStump:
        """Train depth-1 tree on local data."""
        # With one sample, this is trivial - just memorize
        # But with local batching (multiple samples per client), valuable
        stump = DecisionStump()
        stump.fit(self.X, np.array([residual]))
        return stump

    def share_stump_parameters(self) -> dict:
        """Secret-share stump parameters for aggregation."""
        # Share: feature_idx, threshold, left_value, right_value
        ...


class EnsembleAggregator:
    def aggregate_stumps(self, stump_shares: list[dict]) -> GlobalTree:
        """Combine local stumps into global tree."""
        # Option 1: Majority vote on split decisions
        # Option 2: Weighted average of leaf values
        # Option 3: Distillation - train global tree on stump predictions
        ...
```

**Use case**: When clients have multiple samples (not single-sample setting).

### 4.3 Split Learning for Gradient Boosting

Partition computation between client and server:

```
Client (has data):          Server (has model):
    │                            │
    │── features ──────────────> │
    │                            ├── forward pass
    │ <──── activations ─────────│
    │                            │
    ├── compute gradients        │
    │                            │
    │── gradient shares ───────> │
    │                            ├── update model
```

For XGBoost, this would mean:
- Clients compute and share gradient/hessian
- Server finds splits and broadcasts thresholds
- Clients apply splits locally

This is essentially what privateboost already does, but split learning terminology emphasizes the neural network heritage.

---

## 5. Implementation Roadmap

### 5.1 Phase 1: Foundation (Weeks 1-2)

- [ ] Add differential privacy to Aggregator
  - [ ] Implement privacy accountant
  - [ ] Add Laplace noise to `_reconstruct_stats()`
  - [ ] Add Laplace noise to `_reconstruct_histograms()`
  - [ ] Add Laplace noise to `compute_leaf_values()`
  - [ ] Parameterize epsilon, test accuracy impact

- [ ] Increase default bins from 10 to 32
  - [ ] Update `DEFAULT_N_BINS`
  - [ ] Benchmark communication increase
  - [ ] Measure gain retention improvement

### 5.2 Phase 2: Strengthen ShareHolders

- [ ] Implement Shamir secret sharing
  - [ ] Polynomial-based share generation
  - [ ] Lagrange interpolation for reconstruction
  - [ ] Verify linearity (sum of shares = share of sum)

- [ ] Upgrade to 3-of-5 threshold scheme
  - [ ] Update Client to generate 5 shares
  - [ ] Update Aggregator to reconstruct from any 3
  - [ ] Add dropout detection and handling

- [ ] Shareholder distribution strategy
  - [ ] Document deployment across cloud providers
  - [ ] Consider jurisdictional separation

### 5.3 Phase 3: Robustness (Weeks 5-6)

- [ ] Add gradient clipping
  - [ ] Configurable clip norm
  - [ ] Per-client gradient bounds

- [ ] Add input validation
  - [ ] Verify share counts match
  - [ ] Detect NaN/Inf values
  - [ ] Timeout handling for dropped clients

### 5.4 Phase 4: Advanced (Future)

- [ ] Verifiable secret sharing (Pedersen commitments)
- [ ] Federated quantile sketches for adaptive binning
- [ ] Hierarchical aggregation for scale
- [ ] Multi-tree parallelism

---

## 6. Evaluation Plan

### 6.1 Accuracy Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Gain retention | 98% | >97% with DP |
| Test accuracy (Heart) | 85% | >83% with DP |
| Test accuracy (Breast Cancer) | 93% | >91% with DP |

### 6.2 Privacy Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Epsilon (DP) | ∞ (none) | 1.0-2.0 per tree |
| Collusion threshold | 2-of-3 | n-1 (SecAgg) |

### 6.3 Efficiency Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Messages per round | 3n (to SHs) | n (SecAgg) |
| Bytes per client | ~1KB | ~3KB (more bins) |
| Rounds per tree | depth+1 | depth+1 (unchanged) |

### 6.4 Robustness Tests

- [ ] Byzantine client injection (random gradients)
- [ ] Client dropout (10%, 30%, 50%)
- [ ] Aggregator crash recovery
- [ ] Collusion simulation (SecAgg vs ShareHolder)

---

## 7. Open Questions

1. **Privacy budget allocation**: How to optimally distribute ε across rounds and features?

2. **Adaptive clipping**: Should clip norm be learned or fixed?

3. **Quantile bins**: Is the complexity worth it vs. more equal-width bins?

4. **Hierarchical aggregation**: At what scale does two-level aggregation help?

5. **Asynchronous training**: Can we relax synchronization requirements?

---

## 8. References

1. Bonawitz et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017)

2. Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)

3. Chen & Guestrin. "XGBoost: A Scalable Tree Boosting System" (KDD 2016)

4. Cheng et al. "SecureBoost: A Lossless Federated Learning Framework" (IEEE 2021)

5. Karnin, Lang, Liberty. "Optimal Quantile Approximation in Streams" (FOCS 2016)

6. Pedersen. "Non-Interactive and Information-Theoretic Secure Verifiable Secret Sharing" (CRYPTO 1991)