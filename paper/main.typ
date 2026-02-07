// PrivateBoost: Privacy-Preserving Federated Gradient Boosting for Cross-Device Medical Data
// ArXiv preprint

#import "@preview/abbr:0.3.0"
#show: abbr.show-rule

#abbr.make(
  ("IID", "independent and identically distributed"),
  ("MPC", "multi-party computation"),
  ("SecAgg", "Secure Aggregation"),
  ("UCI", "University of California, Irvine"),
)

#set document(
  title: "PrivateBoost: Privacy-Preserving Federated Gradient Boosting for Cross-Device Medical Data",
  author: ("Bernhard Specht", "Samaher Garbaya", "Orhan Ermis", "Reinhard Schneider", "Ricardo Chavarriaga", "Djamel Khadraoui", "Zied Tayeb"),
)

#set page(
  paper: "a4",
  margin: (x: 1.5cm, y: 2cm),
  columns: 2,
  numbering: "1",
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
        PrivateBoost: Privacy-Preserving Federated Gradient Boosting for Cross-Device Medical Data
      ]

      #v(0.8em)

      #text(size: 11pt)[
        Bernhard Specht#super[1,2,\*],
        Samaher Garbaya#super[1],
        Orhan Ermis#super[3],
        Reinhard Schneider#super[4],
        Ricardo Chavarriaga#super[5],
        Djamel Khadraoui#super[3],
        Zied Tayeb#super[1,6]
      ]

      #v(0.5em)

      #text(size: 8pt)[
        #super[1]MyelinZ, 125 Deansgate, Manchester M3 2BY, United Kingdom \
        #super[2]University of Luxembourg, Esch-sur-Alzette, Luxembourg \
        #super[3]Luxembourg Institute of Science and Technology (LIST), Esch-sur-Alzette, Luxembourg \
        #super[4]Luxembourg Centre for Systems Biomedicine (LCSB), Esch-sur-Alzette, Luxembourg \
        #super[5]ZHAW Zurich University of Applied Sciences, Winterthur, Switzerland \
        #super[6]University of Lincoln, Lincoln, United Kingdom \
        #v(0.3em)
        #super[\*]Corresponding author: #raw("bernhard.specht@myelinz.com")
      ]

      #v(1em)

      #block(width: 85%)[
        #align(left)[
          #text(weight: "bold")[Abstract.]
          #text(size: 9pt)[
            Cross-device medical federated learning---where individual patients, rather than institutions, participate directly---poses a unique challenge: each client holds only a few samples, often just one (e.g., a single diagnostic record), leaving insufficient local data for gradient computation or secure pairwise aggregation. Existing approaches such as #[@SecAgg:lsf] require client-to-client coordination impractical for intermittently available mobile devices, while homomorphic encryption introduces substantial computational overhead. We present _privateboost_, a federated XGBoost system that addresses this setting through m-of-n Shamir secret sharing with commitment-based anonymous aggregation. Clients distribute shares to a fixed set of shareholders---requiring no client-to-client communication---and the aggregator reconstructs only aggregate gradient sums via Lagrange interpolation, never observing individual values or client identities. We evaluate on @UCI:lsf medical datasets, demonstrating 98% split gain retention relative to centralized XGBoost and accuracy resilient to up to 80% client dropout.
          ]
        ]
      ]

      #v(1em)
    ]
  ]
]

= Introduction

Federated learning enables machine learning on distributed data without centralizing raw samples @mcmahan2017. However, most existing work assumes a _cross-silo_ setting where each client (typically an organization) holds multiple data samples and can perform local training @kairouz2021. Popular frameworks such as Flower @beutel2020flower and NVIDIA FLARE @roth2022nvidia support both settings but are most commonly deployed in cross-silo scenarios.

In medical settings, "federated learning" typically means hospitals or research institutions each manage patient cohorts @rieke2020 @sheller2020, with the institution serving as a trusted aggregation point for its patients' data. True _cross-device_ medical federated learning, where individual patients control their own records without institutional intermediaries, remains largely unexplored. Cross-device settings have been studied for applications like mobile keyboard prediction @hard2018, but medical applications present unique challenges.

First, _extreme non-@IID:s data_: each client holds exactly one sample. A patient participating in a federated study contributes their single health record. There are no local batches, no local gradients to average; each participant is a single data point. Second, tree-based methods like XGBoost @chen2016 require global statistics (gradient histograms across all samples) to find optimal split points. Without local batches large enough for meaningful statistics or secure aggregation, clients cannot contribute to these histograms without revealing their individual values.

Existing privacy-preserving approaches face practical barriers in this setting. @SecAgg:s @bonawitz2017 requires client-to-client coordination for pairwise key agreement, which is problematic when clients are intermittently online mobile devices. Homomorphic encryption @aono2017 enables computation on encrypted data but introduces substantial computational overhead.

We present _privateboost_, a privacy-preserving federated XGBoost system designed for the one-sample-per-client setting:

+ A protocol using m-of-n Shamir secret sharing with commitment-based client anonymity
+ A three-party architecture (clients → shareholders → aggregator) requiring no client-to-client communication
+ Privacy guarantees: raw data never leaves the client; shareholders see only shares; the aggregator sees only aggregate sums
+ Empirical validation on medical datasets demonstrating 98% split gain retention with dropout resilience

#v(1em)

= Related Work

== Secure Aggregation

@SecAgg:s @bonawitz2017 enables a server to compute sums of client updates without learning individual values. The protocol uses pairwise masking: each pair of clients agrees on a random mask via Diffie-Hellman key exchange, and these masks cancel when the server sums all contributions. This requires $O(n)$ pairwise key agreements per client and coordination to handle dropouts. Subsequent work has improved the communication complexity @bell2020, but the fundamental requirement for client-to-client coordination remains.

Our approach uses a star topology where clients distribute shares to a fixed set of shareholders without any client-to-client communication. This is better suited to cross-device settings where clients are intermittently available.

== Federated Gradient Boosting

Gradient boosting @friedman2001 builds ensembles of weak learners. XGBoost @chen2016 and LightGBM @ke2017 are widely adopted implementations that use histogram-based split finding for efficiency.

SecureBoost @cheng2021 addresses federated XGBoost in the _vertical_ federated learning setting, where different parties hold different features for the same samples. This assumes institutional participants with aligned datasets. Federated Forest @liu2020fedforest similarly targets cross-silo settings with multiple samples per participant. Recent work has extended secret sharing to vertical federated XGBoost @xie2022fedxgb, and FedTree @li2023fedtree provides a comprehensive system supporting both horizontal and vertical settings with configurable privacy techniques.

For _horizontal_ federated learning, SimFL @li2020simfl uses locality-sensitive hashing (LSH) to share similarity information between parties. However, SimFL requires each party to hold many samples: each party builds trees using only its local instances, weighted by gradients from similar samples at other parties. This fundamentally precludes the one-sample-per-client setting. Additionally, LSH-based approaches have weaker privacy guarantees than cryptographic methods: parties learn which samples are similar across institutions, which could enable inference attacks on sensitive medical data.

Federated tree learning has also been explored with differential privacy @li2020ppgbdt, but typically assumes clients hold multiple samples for local histogram computation.

Our work addresses the orthogonal _horizontal_ federated setting with extreme non-@IID:s: one sample per client, providing information-theoretic security via secret sharing rather than the computational security of LSH-based approaches.

== Secret Sharing in Machine Learning

Shamir secret sharing @shamir1979 is a foundational primitive for @MPC @benor1988. Verifiable extensions @gennaro1998 enable detection of malicious behavior. These techniques have been applied in @MPC:s frameworks for machine learning inference, but less explored for federated training.

The key property we exploit is _additive homomorphism_: the sum of shares equals shares of the sum, enabling aggregation without reconstruction of individual values.

#v(1em)

= Protocol

== Background: Histogram-Based Split Finding

XGBoost @chen2016 builds decision trees by greedily selecting splits that maximize a gain function. For each node, the algorithm considers every feature and every possible split threshold. To make this tractable, histogram-based methods discretize continuous features into $B$ bins and compute gradient statistics per bin.

Each sample $i$ contributes a gradient $g_i = partial L / partial hat(y)_i$ and Hessian $h_i = partial^2 L / partial hat(y)_i^2$ based on the current prediction $hat(y)_i$ and true label $y_i$. For a candidate split that partitions samples into left ($L$) and right ($R$) sets, the gain is:

$ "Gain" = 1/2 [ G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - (G_L + G_R)^2 / (H_L + H_R + lambda) ] $

where $G_L = sum_(i in L) g_i$, $H_L = sum_(i in L) h_i$, and $lambda$ is a regularization parameter. The key insight is that split finding requires only _aggregate_ gradient sums per bin, not individual gradients. Our protocol computes these sums without revealing individual contributions.

== System Model

We consider $c$ clients, each holding a single labeled sample $(x_i, y_i)$. A set of $n$ _shareholders_ (e.g., $n = 3$) act as intermediate aggregation points. A single _aggregator_ coordinates the training process.

#place(top, float: true, scope: "parent")[
  #figure(
    image("figures/architecture.png", width: 90%),
    caption: [System architecture. Clients distribute Shamir shares to all shareholders. Shareholders sum received shares and forward partial sums to the aggregator, which reconstructs aggregate statistics (Σx, Σx² for binning) and gradient sums (ΣG, ΣH for splits). The aggregator broadcasts bin configurations and split decisions back to clients.]
  )
]

We assume honest-but-curious parties. Shareholders do not collude (or at most $m-1$ collude for $m$-of-$n$ threshold). The aggregator follows the protocol but may attempt to learn individual values from aggregates.

== Building Blocks

*Shamir Secret Sharing.* To share a secret value $s$, sample a random polynomial $f(x)$ of degree $m-1$ with $f(0) = s$. Distribute shares $(i, f(i))$ for $i = 1, ..., n$ to $n$ shareholders. Any $m$ shares can reconstruct $s$ via Lagrange interpolation; fewer than $m$ shares reveal nothing about $s$.

*Additive Homomorphism.* For shares of values $a$ and $b$, the sum of shares equals shares of $(a + b)$. This allows shareholders to sum their received shares without learning individual values, and the aggregator reconstructs only the sum.

*Commitments.* Each client generates a commitment $c = H("round" || "client_id" || "nonce")$ using a fresh nonce per round. Shareholders only sum shares that have matching commitment hashes, ensuring consistent aggregation: if a client's share reaches some shareholders but not others, those shares are excluded rather than causing corrupted reconstruction. The aggregator sees only commitment hashes, not client identifiers.

== Protocol Flow

The protocol has two phases: a one-time statistics phase to define histogram bins, followed by repeated gradient phases to build trees.

*Phase 1: Statistics.* Before training, the aggregator needs feature statistics to define histogram bin edges. Each client $i$ creates Shamir shares of their feature values $x_(i,f)$ and squared values $x_(i,f)^2$ for each feature $f$, along with a commitment. Clients send shares to shareholders. The aggregator requests sums from shareholders, reconstructs $sum x$ and $sum x^2$, computes per-feature mean $mu$ and variance $sigma^2$, and defines $B$ equal-width bins spanning $mu plus.minus 3sigma$. The aggregator broadcasts bin edges to all clients.

*Phase 2: Gradient Rounds.* For each tree and each depth level, the aggregator needs gradient sums per (feature, bin) combination. Each client $i$:

+ Computes their gradient $g_i$ and Hessian $h_i$ from current ensemble prediction
+ For each feature $f$, determines which bin $b$ their value $x_(i,f)$ falls into
+ Creates Shamir shares of $(g_i, h_i)$ with a fresh commitment
+ Sends shares to shareholders, tagged with $(f, b, "commitment")$

Shareholders accumulate shares grouped by (feature, bin, commitment). When the aggregator requests sums for a set of commitments, each shareholder returns summed shares per (feature, bin). The aggregator reconstructs gradient sums $G_(f,b) = sum g_i$ and $H_(f,b) = sum h_i$ for each bin, evaluates the gain formula for each possible split, selects the best split, and broadcasts the decision (feature, threshold) to clients. Clients update their node assignments, and the process repeats for the next depth level.

After reaching maximum depth, leaf values are computed from gradient sums and the tree is added to the ensemble. The entire gradient phase repeats for each tree.

#v(1em)

= Security Analysis

== Privacy Guarantees

*Clients* hold their raw feature values and labels locally. Each client submits Shamir shares along with a pseudonymous client identifier (to prevent duplicate submissions) to shareholders. This identifier is linked to an account but does not reveal the client's real identity. Since shares are information-theoretically secure, no single shareholder can learn anything about the original values.

*Shareholders* receive shares tagged with client identifiers and commitment hashes. They use identifiers to prevent duplicate submissions but do not forward them to the aggregator. Shareholders cannot reconstruct values without colluding with at least $m-1$ other shareholders. The collusion threshold is configurable: with m-of-n sharing, at least m shareholders must collude to reconstruct individual values. Increasing from 2-of-3 to 3-of-5 or higher provides stronger guarantees while maintaining dropout tolerance. Shareholders sum shares for requested commitments and return only these aggregated sums.

*The aggregator* receives summed shares from shareholders and reconstructs aggregate gradient sums via Lagrange interpolation. Crucially, it never sees client identifiers, only opaque commitment hashes. It learns split thresholds (necessary for prediction) and gradient sums per bin. It learns how many clients fall into each tree branch but not which specific clients.

== Limitations

*Branch counts.* The aggregator learns client counts per tree branch. For branches with very few clients, this narrows the anonymity set. Combined with auxiliary information, this could enable inference attacks. This motivates k-anonymous tree constraints (Section 6.2).

*Shareholder path visibility.* Shareholders observe which bin each client submits gradients to, revealing individual tree paths. This can be addressed using the path hiding technique described in Section 6.3: clients submit shares for all possible paths, with only the true path containing non-zero values. However, this incurs communication overhead that grows exponentially with tree depth ($O(2^d)$ per client per round).

*Cross-round linkability.* Commitment hashes are fresh per round (using a new random nonce each time), preventing the aggregator from linking submissions across rounds. However, shareholders can link a client's submissions across rounds via transport-layer identity (e.g., network address). This is a transport-level concern, not a protocol-level one, and is straightforwardly mitigated by routing client submissions through an anonymous communication channel.

*No differential privacy.* The protocol reveals exact aggregate sums. Future work addresses formal differential privacy guarantees.

*Malicious clients.* The protocol assumes honest-but-curious clients. In practice, mobile deployments can leverage platform attestation mechanisms (such as App Check tokens on Android and App Attest on iOS) to verify that participating clients are running legitimate, unmodified application code. This provides a practical defense against Sybil attacks and gradient poisoning without requiring cryptographic verification of individual contributions.

*Single aggregator.* The current design assumes a single honest-but-curious aggregator. A malicious aggregator could corrupt split decisions or model outputs. This can be mitigated by running multiple independent aggregators that reconstruct the same values and cross-verify results, or by using verifiable computation techniques.

#v(1em)

= Experiments

We evaluate on three medical datasets from the @UCI:s Machine Learning Repository @uci2019. Heart Disease @detrano1989 (297 samples) predicts coronary artery disease from 13 clinical features. Breast Cancer @street1993 (569 samples) classifies tumors using 30 cell nuclei measurements. Pima Diabetes @smith1988 (768 samples) predicts diabetes onset from 8 diagnostic measurements. We use an 80/20 train/test split for all datasets.

We use 2-of-3 Shamir threshold, 10 bins per feature, max depth 3, 15 trees, learning rate 0.15, and regularization $lambda = 2.0$. Bins are equal-width, spanning $mu plus.minus 3 sigma$ computed from aggregated statistics. We compare against XGBoost with matched hyperparameters and XGBoost with default settings.

#figure(
  image("figures/learning_curves.png", width: 100%),
  caption: [Learning curves comparing _privateboost_ to XGBoost across three medical datasets.],
  placement: auto,
  scope: "parent"
)

Across all three datasets, _privateboost_ achieves competitive test accuracy. On Heart Disease, _privateboost_ achieves 88.3% compared to 83.3% for XGBoost with matched hyperparameters and 76.7% for XGBoost with defaults. On Breast Cancer, all three methods achieve 95.6% accuracy. On Pima Diabetes, _privateboost_ achieves 71.4% compared to 73.4% for both XGBoost configurations.

The Heart Disease result is notable: _privateboost_ outperforms both XGBoost configurations, which we attribute to a regularization effect from histogram binning that reduces overfitting on smaller datasets.

*Split gain retention.* To quantify the information loss from histogram binning, we define _split gain retention_ per feature $f$ as:

$ R_f = G_f^"hist" / G_f^"exact" $

where $G_f^"exact"$ is the optimal split gain obtained by exhaustive search over all unique thresholds in feature $f$ (equivalent to centralized XGBoost with no binning), and $G_f^"hist"$ is the best split gain when restricted to the $B$ histogram bin edges derived from the privacy-preserving statistics phase. Both gains use the same gradient and Hessian values, evaluated at the root level using initial predictions. We report mean retention $macron(R) = 1/F sum_(f=1)^F R_f$ across all $F$ features.

#figure(
  image("figures/gain_retention.png", width: 100%),
  caption: [Split gain retention $R_f$ per feature on Heart Disease. Mean retention $macron(R) = 98%$ across 13 features.]
)

On Heart Disease, mean retention is $macron(R) = 98.1%$ across 13 features, with 10 features achieving $R_f = 100%$. Histogram bins are constructed using aggregated statistics from the first protocol round, requiring no additional privacy cost. Features with skewed distributions may have suboptimal bin placement, with most data concentrated in few bins. This explains the 90-93% retention observed on features like `oldpeak`.

In cross-device settings, clients are intermittently available: mobile devices go offline, patients miss check-ins, or network connectivity fails. The protocol must tolerate such dropout without requiring all clients to participate in every round.

#figure(
  image("figures/dropout_resilience.png", width: 100%),
  caption: [Test accuracy under varying client dropout rates.]
)

The protocol naturally handles dropout: the aggregator reconstructs from whichever clients participate in each round. We simulate dropout by randomly excluding a fraction of clients each training round, with independent sampling per round. Accuracy remains stable up to 80% dropout across all datasets, demonstrating robustness to intermittent client availability.

#v(1em)

= Future Work

== Differential Privacy

Adding calibrated Laplace noise to gradient sums before reconstruction would provide formal $(epsilon, delta)$-differential privacy @dwork2014. Similar techniques have been applied to deep learning @abadi2016. The noise scale can be calibrated using the mean and variance statistics already computed during the binning phase, incurring no additional privacy cost. We suggest per-tree budgets of $epsilon = 1.0 - 2.0$ with composition accounting across the ensemble.

== K-Anonymous Tree Structure

To prevent membership inference via branch analysis, splits could be constrained to ensure at least $k$ clients in every resulting node, inspired by k-anonymity principles @sweeney2002. This extends the existing minimum-client enforcement in shareholders to the tree structure itself. Splits that would create nodes with fewer than $k$ clients would be rejected, trading some model expressiveness for stronger privacy guarantees.

== Private Path Hiding

In the current protocol, the aggregator learns how many clients traverse each branch of the tree, even though it cannot identify which specific clients. This branch count information could narrow the anonymity set, particularly for branches with few clients.

A stronger privacy guarantee would hide even the branch membership. At each tree depth, clients secret-share a binary indicator for each active node: 1 if the client belongs to that node, 0 otherwise. Shareholders sum these indicators alongside the gradient shares. The aggregator reconstructs only the total count per node, learning nothing about individual assignments beyond what the count reveals.

This extension incurs communication overhead of $O(2^d times n times c)$ shares per gradient round, where $d$ is tree depth, $n$ is shareholders, and $c$ is clients. For $d = 3$, $n = 3$, and $c = 200$:

- Current protocol: $200 times 3 = 600$ shares per round
- With path hiding: $8 times 200 times 3 = 4,800$ shares per round

This 8× overhead grows exponentially with depth, making the extension most practical for shallow trees ($d <= 3$), which are common in gradient boosting. For applications with stringent privacy requirements, this tradeoff may be acceptable.

#v(1em)

= Conclusion

We presented _privateboost_, a privacy-preserving federated XGBoost system for the extreme non-@IID:s setting where each client holds exactly one sample. Our protocol ensures raw data never leaves the client, requires no client-to-client coordination, and provides resilience to participant dropout.

Evaluation on medical datasets demonstrates 98% split gain retention with accuracy comparable to centralized XGBoost. The system enables true cross-device medical federated learning where patients can participate directly without institutional intermediaries.

=== Acknowledgement
The authors would like to thank the Ministry of the Economy in Luxembourg and its digital health directorate for supporting this research. Similarly, we would like to thank the Luxembourg National Research Fund (FNR) for funding this research.

=== Funding
This work was supported in part by a PhD grant from the Luxembourg National Research Fund (FNR) under the project reference 17223919/MMS/Industrial Fellowship.

#v(2em)

#bibliography("references.bib", style: "ieee")
