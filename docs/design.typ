#set document(title: "privateboost Design", author: "privateboost")
#set page(margin: 2.5cm, numbering: "1")
#set text(font: "Source Sans Pro", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)

#show heading.where(level: 1): it => {
  v(1em)
  text(size: 18pt, weight: "bold", it)
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 14pt, weight: "semibold", it)
  v(0.3em)
}

#align(center)[
  #text(size: 24pt, weight: "bold")[privateboost Design]
  #v(0.5em)
  #text(size: 12pt, fill: gray)[Privacy-Preserving Federated XGBoost via Shamir Secret Sharing]
  #v(2em)
]

= Overview

*privateboost* implements privacy-preserving federated XGBoost using Shamir secret sharing with commitments. Multiple data owners (clients) train a gradient boosting model without revealing their individual data—each client splits their values into Shamir shares distributed across multiple shareholders, who forward only aggregated sums to a central aggregator. The aggregator learns statistical properties needed for training but never sees individual contributions or client identities.

= Architecture

#figure(
  image("figures/architecture.png", width: 90%),
  caption: [Protocol architecture. Clients send Shamir shares with commitments to shareholders, who forward sums to the aggregator.]
)

The protocol uses *m-of-n threshold secret sharing*: any m shareholders can reconstruct the aggregate, but fewer than m learn nothing. For example, with 2-of-3 sharing, the system tolerates one shareholder being unavailable while requiring at least two colluding parties to compromise privacy.

= Shamir Secret Sharing

*Polynomial Construction.* To share a value $v$ with threshold $m$, create a polynomial of degree $m-1$ with $v$ as the constant term:

#align(center)[
  $f(x) = v + a_1 x + a_2 x^2 + ... + a_(m-1) x^(m-1)$
]

where $a_1, ..., a_(m-1)$ are random coefficients. Each shareholder $i$ receives the point $(i, f(i))$.

*Reconstruction.* Given any $m$ points, Lagrange interpolation recovers $f(0) = v$:

#align(center)[
  $v = sum_(j=1)^m y_j product_(k != j) frac(-x_k, x_j - x_k)$
]

*Linearity.* Shamir sharing is additively homomorphic. If shareholders sum their shares from multiple clients, the result is a valid sharing of the sum:

#align(center)[
  $"share"_i (v_1) + "share"_i (v_2) = "share"_i (v_1 + v_2)$
]

This allows shareholders to aggregate without reconstructing individual values.

= Commitment Scheme

*Purpose.* Commitments ensure shareholders process the same set of clients without revealing client identities to the aggregator.

*Construction.* Each client generates a fresh commitment per round:

#align(center)[
  $"commitment" = "SHA256"("round_id" || "client_id" || "nonce")$
]

The commitment is sent to all shareholders along with the Shamir share. Since the nonce is random, different rounds produce different commitments—preventing cross-round correlation.

*Verification.* The aggregator collects commitments from all shareholders and computes the intersection. Clients whose commitments appear in all (or enough) shareholders are included in aggregation. The aggregator never sees client_id, only opaque hashes.

= Protocol Flow

*Phase 1: Client Submission.* Each client computes its values (statistics or gradients), creates Shamir shares, and sends each share with the same commitment to the corresponding shareholder.

*Phase 2: Commitment Collection.* The aggregator requests the set of commitments from each shareholder. No shares are transferred yet—only commitment hashes.

*Phase 3: Shareholder Selection.* The aggregator finds the $m$ shareholders with the largest commitment overlap. This determines which clients will be included and ensures reconstruction is possible.

*Phase 4: Share Request.* The aggregator sends the list of valid commitments to the selected shareholders. Shareholders enforce a minimum threshold (e.g., N ≥ 10 clients) before responding—rejecting requests for too few clients to prevent individual value extraction.

*Phase 5: Aggregation.* Each selected shareholder sums the shares for the requested commitments and sends the total to the aggregator.

*Phase 6: Reconstruction.* The aggregator uses Lagrange interpolation on the summed shares to recover the aggregate value.

= Histogram Construction

*Bin Definition.* After the statistics round, the aggregator knows the global mean ($mu$) and standard deviation ($sigma$) for each feature. It defines histogram bins spanning $mu plus.minus 3 sigma$, divided into `n_bins` equal-width intervals, plus underflow and overflow bins for outliers:

#align(center)[
  $"edges" = [-infinity, mu - 3 sigma, ..., mu + 3 sigma, +infinity]$
]

This creates `n_bins + 2` total bins per feature.

*Gradient Histograms.* Clients contribute their gradient and hessian to the appropriate bin. Each client computes:
- $"gradient" = "prediction" - "target"$ (for squared loss)
- $"hessian" = 1.0$ (or $p(1-p)$ for logistic loss)

Then places these values in the bin corresponding to each feature value, creating per-feature gradient/hessian vectors that are Shamir-shared. The aggregator reconstructs summed gradients per bin, enabling it to find the optimal split threshold by evaluating cumulative gain across bin boundaries.

= Security Guarantees

*Confidentiality.* No single party learns individual client values. Shareholders see only random polynomial evaluations; the aggregator sees only aggregate sums across N+ clients.

*Threshold Security.* With m-of-n sharing, any m-1 colluding shareholders cannot recover values. For 2-of-3, this means a single compromised shareholder reveals nothing.

*Minimum N Enforcement.* Shareholders reject requests for fewer than N commitments, preventing the aggregator from isolating individual clients.

*Commitment Privacy.* The aggregator sees only commitment hashes, never client IDs. Fresh nonces each round prevent correlation across rounds.

*Threat Model.* The protocol assumes _honest-but-curious_ adversaries—parties follow the protocol correctly but may try to learn extra information. Trust assumption: fewer than m shareholders collude with the aggregator.

= Split Quality Analysis

*Gain Formula.* XGBoost evaluates splits using information gain:

#align(center)[
  $ "gain" = frac((sum_(i in L) g_i)^2, sum_(i in L) h_i + lambda) + frac((sum_(i in R) g_i)^2, sum_(i in R) h_i + lambda) - frac((sum_i g_i)^2, sum_i h_i + lambda) $
]

Where $g_i$ = gradient, $h_i$ = hessian for sample $i$, and $lambda$ = L2 regularization.

#figure(
  image("figures/gain_retention.png", width: 100%),
  caption: [Gain retention by feature on UCI Heart Disease dataset (n=237 training samples).]
)

*Results.* privateboost achieves *98.1% mean gain retention* across all features. 10 of 13 features achieve 100% retention—the histogram bins capture the optimal split exactly.

= Learning Performance

#figure(
  image("figures/learning_curves.png", width: 100%),
  caption: [Learning curves on Breast Cancer Wisconsin dataset (n=569).]
)

*Results.* privateboost achieves *96.5% test accuracy*, comparable to standard XGBoost. The learning curves show similar convergence patterns—privacy preservation does not degrade model quality.

= Limitations and Future Directions

*Byzantine protection.* The protocol does not detect shareholders or aggregators that deviate from correct behavior. Two extensions address this: (1) _multiple aggregators_ with majority voting detect corruption at cost of redundant computation, or (2) _secure multi-party computation (MPC)_ among shareholders eliminates the single aggregator entirely. MPC provides stronger guarantees but adds significant latency and message overhead.

*Differential privacy.* Aggregate statistics are revealed exactly, allowing inference attacks with auxiliary knowledge. Adding calibrated noise provides formal guarantees, but noise calibration requires knowing value bounds. For gradient histograms, classification naturally bounds gradients to approximately [-1, 1]. For statistics, either clip contributions or use domain knowledge about feature bounds.

*Communication overhead.* Messages scale as $O("clients" times "shareholders" times "rounds")$. Hierarchical aggregation groups clients under local aggregators that pre-aggregate before forwarding, reducing message count to $O("clients" + "local aggregators" times "shareholders")$—but local aggregators become additional trust points.

*Collusion threshold.* If m shareholders collude with the aggregator, individual values can be reconstructed. Increasing the threshold (e.g., 3-of-5 instead of 2-of-3) raises the bar. Requiring shareholders from different organizations or jurisdictions adds social and legal barriers beyond cryptographic ones.

*Minimum client requirement.* Rounds require at least N clients (e.g., N=10) to preserve privacy. This is fundamental to aggregate-based privacy—with fewer participants, individual contributions become identifiable regardless of cryptographic protections.

*Node path leakage.* During tree construction, clients submit gradients tagged with their current node ID. This reveals to shareholders which tree path each client follows—leaking information about feature values (e.g., "client A went left at the age≤50 split"). While shareholders cannot see the actual gradient values (only Shamir shares), the node membership pattern is visible.

_Mitigation: Zero-padding._ Clients submit gradient shares for _all_ possible nodes at each depth, with zeros for nodes they are not in. Since Shamir shares of zero look random, shareholders cannot distinguish real from dummy submissions. Shareholders sum position-wise without learning node membership.

_Overhead._ Communication increases by $O(2^d)$ where $d$ is tree depth. For typical depths:
- Depth 3: 8× overhead
- Depth 4: 16× overhead
- Depth 5: 32× overhead

This is bounded by `max_depth`, a hyperparameter already constrained for model complexity reasons.

_Alternative: Secure computation._ More advanced techniques (garbled circuits, function secret sharing) could partition gradient sums obliviously—shareholders aggregate without learning which clients belong to which partition. This eliminates the $O(2^d)$ overhead but requires complex MPC protocols with higher latency. For most deployments, the zero-padding approach offers a practical trade-off.
