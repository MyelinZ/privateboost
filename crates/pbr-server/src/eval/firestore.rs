//! Firestore writer for `paperSimTreeMetrics`, built on the `firestore`
//! crate's `FirestoreDb` (gRPC, auth via `gcloud-sdk`). The write is admin
//! and bypasses security rules, which is why the `paperSimTreeMetrics` rule
//! locks out all client access: only the aggregator (and the
//! ADC-authenticated analysis fetch) may touch the collection. A write
//! failure is returned to the caller, which logs and continues so a dropped
//! metric never blocks the round loop.

use super::{MetricSink, TreeMetricDoc};
use firestore::{FirestoreDb, FirestoreDbOptions, firestore_document_from_serializable};
use std::time::Duration;

/// The one place a `TreeMetricDoc` becomes a Firestore document. The name
/// argument must stay empty: a non-empty name becomes `Document.name`, which
/// the create API rejects alongside `generate_document_id`.
pub(crate) fn tree_metric_document(
    doc: &TreeMetricDoc,
) -> firestore::FirestoreResult<firestore::FirestoreDocument> {
    firestore_document_from_serializable("", doc)
}

/// Whole-request timeout for one document write, so a hung connection cannot
/// park a spawned write task (and its socket) indefinitely.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct FirestoreWriter {
    db: FirestoreDb,
}

impl FirestoreWriter {
    /// `service_account_path = None` uses Application Default Credentials
    /// (`FirestoreDb::new`); `Some(path)` loads and signs with that
    /// service-account JSON key file directly, the same two token sources
    /// as `crate::fcm::FcmSender::from_config`. The key needs Firestore
    /// write (`roles/datastore.user`); an FCM-only key gets a permission
    /// error on the first write.
    pub async fn from_config(
        project_id: String,
        service_account_path: Option<String>,
    ) -> anyhow::Result<Self> {
        let db = match service_account_path {
            Some(path) => {
                FirestoreDb::with_options_service_account_key_file(
                    FirestoreDbOptions::new(project_id),
                    path.into(),
                )
                .await?
            }
            None => FirestoreDb::new(&project_id).await?,
        };
        Ok(Self { db })
    }
}

#[tonic::async_trait]
impl MetricSink for FirestoreWriter {
    async fn write(&self, doc: &TreeMetricDoc) -> anyhow::Result<()> {
        let firestore_doc = tree_metric_document(doc)?;
        tokio::time::timeout(
            WRITE_TIMEOUT,
            self.db
                .fluent()
                .insert()
                .into("paperSimTreeMetrics")
                .generate_document_id()
                .document(firestore_doc)
                .execute(),
        )
        .await??;
        Ok(())
    }
}
