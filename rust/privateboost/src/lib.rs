pub mod crypto;
pub mod csv_io;
pub mod math;
pub mod metrics;
pub mod model;
pub mod protocol;
pub mod split;
pub mod traffic;

pub use crypto::{Commitment, F, Share};
pub use crypto::{commit, generate_nonce};
pub use crypto::{decode, decode_all, encode, encode_all};
pub use crypto::{reconstruct, share};
pub use csv_io::{Dataset, read_csv, write_results};
pub use metrics::{accuracy, auc_roc, f1_score};
pub use model::{Model, Tree, TreeNode};
pub use protocol::{
    Aggregator, AggregatorBuilder, BinConfiguration, BinMethod, Client, ClientBuilder, CommittedGradientShare,
    CommittedStatsShare, Loss, NodeTotals, RoundContext, ShareHolder, SplitDecision,
};
pub use split::{stratified_split, SplitData};
pub use traffic::{TrafficLog, TrafficEntry, Direction};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("threshold {threshold} exceeds n_parties {n_parties}")]
    ThresholdExceedsParties { threshold: usize, n_parties: usize },

    #[error("need at least {needed} shares, got {got}")]
    InsufficientShares { needed: usize, got: usize },

    #[error("need at least {needed} clients, got {got}")]
    InsufficientClients { needed: usize, got: usize },

    #[error("field element has no inverse (zero)")]
    FieldInverse,

    #[error("unknown commitment")]
    UnknownCommitment,

    #[error("no shares for node {0}")]
    NoSharesForNode(usize),

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
