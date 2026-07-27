use serde::Deserialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use tonic::transport::{Identity, ServerTlsConfig};

#[derive(Deserialize, Clone)]
pub struct ShareholderConfig {
    pub x_coord: u64,
    pub min_clients: usize,
    pub listen: SocketAddr,
    pub internal_listen: SocketAddr,
    pub auth: AuthConfig,
    /// For the client-facing listener only; absent leaves it plaintext. The
    /// internal loopback plane is never wrapped in TLS.
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

/// PEM cert and key. Both roles share this shape; each `serve()` applies it to
/// its client-facing listener alone.
#[derive(Deserialize, Clone)]
pub struct TlsConfig {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
}

impl TlsConfig {
    /// Reads the PEM cert and key, installing the process default rustls
    /// crypto provider first (see [`ensure_crypto_provider`]).
    pub fn server_tls_config(&self) -> anyhow::Result<ServerTlsConfig> {
        ensure_crypto_provider();
        let cert = std::fs::read(&self.cert_path)?;
        let key = std::fs::read(&self.key_path)?;
        Ok(ServerTlsConfig::new().identity(Identity::from_pem(cert, key)))
    }
}

static INSTALL_CRYPTO: Once = Once::new();

/// Install the ring provider as rustls's process-wide default, exactly once.
/// The dependency tree links both `ring` and `aws-lc-rs`, so rustls cannot pick
/// for itself and `ServerConfig::builder()` would panic. The install error
/// means another component won the race, and any installed provider will do.
pub fn ensure_crypto_provider() {
    INSTALL_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

#[derive(Deserialize, Clone)]
pub struct AuthConfig {
    pub issuer: String,
    pub audience: String,
    pub static_keys: Vec<StaticKey>,
    /// Google's JWKS endpoint for Firebase `securetoken` keys. When set,
    /// `serve()` fetches it once before accepting traffic and refreshes it
    /// periodically, and `static_keys` may be empty.
    #[serde(default)]
    pub google_jwks_url: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct StaticKey {
    pub kid: String,
    pub public_key_pem_path: PathBuf,
}

impl ShareholderConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let cfg = toml::from_str::<ShareholderConfig>(&std::fs::read_to_string(path)?)?;
        if cfg.x_coord == 0 {
            anyhow::bail!("x_coord must be >= 1 (Shamir evaluation points start at 1)");
        }
        Ok(cfg)
    }
}

impl AuthConfig {
    pub fn build_verifier(&self) -> anyhow::Result<crate::auth::Verifier> {
        let mut keys = Vec::new();
        for k in &self.static_keys {
            keys.push(crate::auth::VerifierKey {
                kid: k.kid.clone(),
                pem: std::fs::read(&k.public_key_pem_path)?,
            });
        }
        crate::auth::Verifier::from_static(self.issuer.clone(), self.audience.clone(), keys)
    }

    /// Build the verifier. With `google_jwks_url` set, Firebase's key set is
    /// fetched once before returning, so the first real token verifies, and a
    /// background task then refreshes it every
    /// `auth::GOOGLE_JWKS_REFRESH_INTERVAL`. A failed background refresh logs
    /// and keeps the previous keys; the initial fetch is not backgrounded, so
    /// failing to reach Google fails `serve()` rather than coming up unable to
    /// verify anything. Unset, only the static keys are loaded.
    ///
    /// The returned [`RefreshHandle`] owns the task and aborts it on drop, so
    /// the refresh cannot outlive the server it authenticates for.
    pub async fn build_and_refresh_verifier(&self) -> anyhow::Result<RefreshingVerifier> {
        let verifier = Arc::new(self.build_verifier()?);
        let refresh = match self.google_jwks_url.clone() {
            Some(url) => {
                verifier.refresh_from_jwks_url(&url).await?;
                let refresh_verifier = verifier.clone();
                let task = tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(crate::auth::GOOGLE_JWKS_REFRESH_INTERVAL).await;
                        if let Err(e) = refresh_verifier.refresh_from_jwks_url(&url).await {
                            tracing::warn!(error = %e, "Firebase key refresh failed; keeping previous keys");
                        }
                    }
                });
                RefreshHandle(Some(task.abort_handle()))
            }
            None => RefreshHandle(None),
        };
        Ok(RefreshingVerifier { verifier, refresh })
    }
}

/// Keep `refresh` alive as long as the verifier is in use: dropping it aborts
/// the background Firebase-key refresh.
pub struct RefreshingVerifier {
    pub verifier: Arc<crate::auth::Verifier>,
    pub refresh: RefreshHandle,
}

/// Aborts the background Firebase-key refresh on drop, tying it to the server's
/// lifetime. `None` when no `google_jwks_url` was configured.
pub struct RefreshHandle(Option<tokio::task::AbortHandle>);

impl Drop for RefreshHandle {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.abort();
        }
    }
}
