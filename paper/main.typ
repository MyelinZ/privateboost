// Privacy-Preserving Federated XGBoost for Cross-Device Medical Applications
// ArXiv preprint

#set document(
  title: "Privacy-Preserving Federated XGBoost for Cross-Device Medical Applications",
  author: "Author Names",
)

#set page(
  paper: "a4",
  margin: (x: 1.5cm, y: 2cm),
  columns: 2,
)

#set text(
  font: "New Computer Modern",
  size: 10pt,
)

#set par(
  justify: true,
  leading: 0.65em,
)

#set heading(numbering: "1.1")

// Title and abstract span both columns
#place(top + center, float: true, scope: "parent")[
  #block(width: 100%)[
    #align(center)[
      #text(size: 16pt, weight: "bold")[
        Privacy-Preserving Federated XGBoost for Cross-Device Medical Applications
      ]

      #v(0.8em)

      #text(size: 11pt)[
        Author Names \
        #text(size: 9pt, style: "italic")[Institution]
      ]

      #v(1em)

      #block(width: 85%)[
        #align(left)[
          #text(weight: "bold")[Abstract.]
          #text(size: 9pt)[
            We present privateboost, a privacy-preserving federated XGBoost system for the extreme non-IID setting where each client holds exactly one data sample. This cross-device setting arises naturally in medical applications where each patient controls their own record. Our protocol uses m-of-n Shamir secret sharing with commitment-based anonymous aggregation: raw feature values never leave the client, shareholders see only secret shares, and the aggregator reconstructs only aggregate gradient sums. Unlike SecAgg-based approaches requiring client-to-client coordination for pairwise key agreement, our architecture uses a fixed set of stateless shareholders — clients communicate only with shareholders, never with each other. We evaluate on UCI medical datasets, demonstrating 98% split gain retention compared to centralized XGBoost with less than 2% accuracy degradation, while maintaining resilience to client dropout. We discuss extensions including differential privacy and k-anonymous tree structures.
          ]
        ]
      ]

      #v(1em)
    ]
  ]
]

= Introduction

Federated learning enables machine learning on distributed data without centralizing raw samples @kairouz2021. However, most existing work assumes a _cross-silo_ setting where each client (typically an organization) holds multiple data samples and can perform local training @mcmahan2017.

In medical settings, "federated learning" typically means hospitals or research institutions each manage patient cohorts @rieke2020 @sheller2020 — the institution remains a trusted aggregation point for its patients' data. True _cross-device_ medical federated learning — where individual patients control their own records without institutional intermediaries — remains largely unexplored. Cross-device settings have been studied for applications like mobile keyboard prediction @hard2018, but medical applications present unique challenges.

This setting presents a fundamental challenge: _extreme non-IID data_ where each client holds exactly one sample. A patient participating in a federated study contributes their single health record. There are no local batches, no local gradients to average — each participant is a single data point.

Existing privacy-preserving approaches face practical barriers in this setting. Secure Aggregation (SecAgg) @bonawitz2017 requires client-to-client coordination for pairwise key agreement, which is problematic when clients are intermittently online mobile devices. Homomorphic encryption @aono2017 enables computation on encrypted data but introduces substantial computational overhead.

We present _privateboost_, a privacy-preserving federated XGBoost system designed for the one-sample-per-client setting. Our contributions:

+ A protocol using m-of-n Shamir secret sharing with commitment-based client anonymity
+ A three-party architecture (clients → shareholders → aggregator) requiring no client-to-client communication
+ Privacy guarantees: raw data never leaves the client; shareholders see only shares; the aggregator sees only aggregate sums
+ Empirical validation on medical datasets demonstrating 98% split gain retention with dropout resilience

#v(1em)

= Related Work

== Secure Aggregation

Secure Aggregation (SecAgg) @bonawitz2017 enables a server to compute sums of client updates without learning individual values. The protocol uses pairwise masking: clients agree on random masks that cancel when summed. This requires $O(n)$ key agreements per client, and clients must coordinate to handle dropouts. Subsequent work has improved the communication complexity @bell2020, but the fundamental requirement for client coordination remains.

In cross-device settings where clients are intermittently available mobile devices, this coordination is challenging. Our approach eliminates client-to-client communication entirely.

== Federated Gradient Boosting

Gradient boosting @friedman2001 builds ensembles of weak learners, with XGBoost @chen2016 and LightGBM @ke2017 achieving state-of-the-art performance through histogram-based split finding.

SecureBoost @cheng2021 addresses federated XGBoost in the _vertical_ federated learning setting, where different parties hold different features for the same samples. This assumes institutional participants with aligned datasets. Federated Forest @liu2020fedforest similarly targets cross-silo settings with multiple samples per participant.

Federated tree learning has also been explored with differential privacy @li2020ppgbdt, but typically assumes clients hold multiple samples for local histogram computation.

Our work addresses the orthogonal _horizontal_ federated setting with extreme non-IID: one sample per client.

== Secret Sharing in Machine Learning

Shamir secret sharing @shamir1979 is a foundational primitive for secure multi-party computation @benor1988. Verifiable extensions @gennaro1998 enable detection of malicious behavior. These techniques have been applied in MPC frameworks for machine learning inference, but less explored for federated training.

The key property we exploit is _additive homomorphism_: the sum of shares equals shares of the sum, enabling aggregation without reconstruction of individual values.

#v(1em)

= Protocol

== System Model

We consider $n$ clients, each holding a single labeled sample $(x_i, y_i)$. A set of $k$ _shareholders_ (e.g., $k = 3$) act as intermediate aggregation points. A single _aggregator_ coordinates the training process.

#figure(
  image("figures/architecture.png", width: 100%),
  caption: [System architecture. Clients distribute Shamir shares to shareholders, who sum shares and forward to the aggregator for reconstruction.]
)

*Trust model:* We assume honest-but-curious parties. Shareholders do not collude (or at most $m-1$ collude for $m$-of-$k$ threshold). The aggregator follows the protocol but attempts to learn individual values from aggregates.

== Building Blocks

*Shamir Secret Sharing.* To share a value $s$, sample a random polynomial $f(x)$ of degree $m-1$ with $f(0) = s$. Distribute shares $(i, f(i))$ for $i = 1, ..., k$. Any $m$ shares can reconstruct $s$ via Lagrange interpolation; fewer than $m$ shares reveal nothing.

*Additive Homomorphism.* For shares of values $a$ and $b$, the sum of shares equals shares of $(a + b)$. Shareholders can sum their received shares without learning individual values.

*Commitments.* Each client generates a commitment $c = H("round" || "client_id" || "nonce")$ using a fresh nonce per round. The aggregator sees commitments but cannot link them to client identities or across rounds.

== Protocol Phases

*Phase 1: Statistics.* Clients secret-share their feature values $x$ and squared values $x^2$ with commitments. The aggregator reconstructs sums to compute per-feature mean and variance, then defines histogram bin edges.

*Phase 2: Gradient Rounds.* For each tree and depth level:
- Clients compute gradients $(g_i, h_i)$ based on current predictions
- Clients secret-share gradients with fresh commitments per bin
- Shareholders sum shares for each (feature, bin) combination
- Aggregator reconstructs gradient sums and finds optimal splits

*Phase 3: Tree Completion.* The aggregator broadcasts split decisions. Clients update their node assignments and predictions. The tree is added to the ensemble.

#v(1em)

= Security Analysis

== Privacy Guarantees

#table(
  columns: (auto, auto, auto),
  [*Party*], [*Observes*], [*Learns*],
  [Client], [Own raw data], [Nothing new],
  [Shareholder], [Shares + commitments], [Nothing (threshold not met)],
  [Aggregator], [Sums + commitment hashes], [Aggregate statistics only],
)

The aggregator learns split thresholds (necessary for prediction) and gradient sums per bin. It learns _how many_ clients fall into each tree branch but not _which_ clients.

== Limitations

*Collusion threshold.* With 2-of-3 sharing, any two colluding shareholders can reconstruct individual values. Deployments should distribute shareholders across independent parties.

*Branch counts.* The aggregator learns client counts per tree branch. For branches with very few clients, this narrows the anonymity set. Combined with auxiliary information, this could enable inference attacks. This motivates k-anonymous tree constraints (Section 6).

*No differential privacy.* The protocol reveals exact aggregate sums. Future work addresses formal differential privacy guarantees.

*Malicious clients.* The protocol assumes honest-but-curious clients. In practice, mobile deployments can leverage platform attestation mechanisms — such as App Check tokens on Android and App Attest on iOS — to verify that participating clients are running legitimate, unmodified application code. This provides a practical defense against Sybil attacks and gradient poisoning without requiring cryptographic verification of individual contributions.

#v(1em)

= Experiments

== Setup

We evaluate on two UCI medical datasets:

#table(
  columns: (auto, auto, auto, auto),
  [*Dataset*], [*Samples*], [*Features*], [*Task*],
  [Heart Disease], [237], [13], [Binary classification],
  [Breast Cancer], [455], [30], [Binary classification],
)

Configuration: 2-of-3 Shamir threshold, 10 bins per feature (equal-width from aggregated statistics), max depth 3, 15 trees. Baseline: XGBoost with matched hyperparameters.

== Results

#figure(
  image("figures/gain_retention.png", width: 100%),
  caption: [Split gain retention per feature. Mean retention 98% across features.]
)

#figure(
  image("figures/learning_curves.png", width: 100%),
  caption: [Learning curves comparing privateboost to XGBoost. Final accuracy gap less than 2%.]
)

#figure(
  image("figures/dropout_resilience.png", width: 100%),
  caption: [Test accuracy under varying client dropout rates. Stable up to ~30% dropout.]
)

== Discussion

The 98% gain retention demonstrates that histogram binning, not secret sharing, is the primary source of accuracy loss. Features with skewed distributions (e.g., `oldpeak`) show 90-93% retention, suggesting adaptive binning could improve results.

Dropout resilience emerges naturally from commitment-based aggregation: the aggregator works with whichever clients participate in each round.

#v(1em)

= Future Work

== Differential Privacy

Adding calibrated Laplace noise to gradient sums before reconstruction would provide formal $(epsilon, delta)$-differential privacy @dwork2014. Similar techniques have been applied to deep learning @abadi2016. The noise scale can be calibrated using the mean and variance statistics already computed during the binning phase — no additional privacy cost for calibration. We suggest per-tree budgets of $epsilon = 1.0 - 2.0$ with composition accounting across the ensemble.

== K-Anonymous Tree Structure

To prevent membership inference via branch analysis, splits could be constrained to ensure at least $k$ clients in every resulting node, inspired by k-anonymity principles @sweeney2002. This extends the existing minimum-client enforcement in shareholders to the tree structure itself. Splits that would create nodes with fewer than $k$ clients would be rejected, trading some model expressiveness for stronger privacy guarantees.

== Private Path Hiding

In the current protocol, the aggregator learns how many clients traverse each branch of the tree, even though it cannot identify which specific clients. This branch count information could narrow the anonymity set, particularly for branches with few clients.

A stronger privacy guarantee would hide even the branch membership. We sketch an extension using Shamir secret sharing for path indicators.

*Approach.* At each tree depth, clients secret-share a binary indicator for each active node: 1 if the client belongs to that node, 0 otherwise. Shareholders sum these indicators alongside the gradient shares. The aggregator reconstructs only the total count per node, learning nothing about individual assignments beyond what the count reveals.

*Communication overhead.* Let $d$ be the tree depth, $k$ the number of shareholders, and $n$ the number of clients. At depth $d$, there are up to $2^d$ active nodes. Each client must share an indicator for each node, resulting in:

$ O(2^d times k times n) "shares per gradient round" $

For concrete parameters (depth $d = 3$, shareholders $k = 3$, clients $n = 200$):
- Current protocol: $approx 200 times 3 = 600$ gradient shares per round
- With path hiding: $approx 8 times 200 times 3 = 4,800$ shares per round

This represents approximately 8× overhead at depth 3. The overhead grows exponentially with depth, making this extension most practical for shallow trees ($d <= 3$), which are common in gradient boosting where individual trees are intentionally weak learners.

*Tradeoff.* Path hiding provides stronger privacy — the aggregator learns only aggregate branch counts through reconstruction, not through direct observation — at the cost of increased communication. For applications with stringent privacy requirements and moderate tree depths, this tradeoff may be acceptable.

#v(1em)

= Conclusion

We presented privateboost, a privacy-preserving federated XGBoost system for the extreme non-IID setting where each client holds exactly one sample. Our protocol ensures raw data never leaves the client, requires no client-to-client coordination, and provides resilience to participant dropout.

Evaluation on medical datasets demonstrates 98% split gain retention compared to centralized XGBoost with less than 2% accuracy degradation. The system enables true cross-device medical federated learning where patients can participate directly without institutional intermediaries.

#v(2em)

#bibliography("references.bib", style: "ieee")
