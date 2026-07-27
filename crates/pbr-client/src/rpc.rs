use crate::wire_metrics::{CountingIo, WireCounters};
use http::Uri;
use hyper_util::rt::TokioIo;
use pbr_proto::v1::shareholder_service_client::ShareholderServiceClient;
use std::sync::Arc;
use std::sync::Once;
use std::time::Duration;
use tokio::net::TcpStream;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};
use tonic::{Request, Status};

/// Connect budget for every `client_endpoint` channel: generous for a mobile
/// radio yet finite, so a blackholed peer (packets dropped, no TCP RST) fails
/// its own connect instead of parking in retransmission backoff.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Per-RPC deadline on every `client_endpoint` call. The fan-out awaits its
/// channels concurrently, so this bounds the whole fan-out however many
/// shareholders there are, and an unresponsive one cannot stall the rest. On
/// the aggregator channel it stops a half-open connection hanging a poll.
/// Tight enough that an entirely blackholed fan-out returns inside a wake
/// budget.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

static INSTALL_CRYPTO: Once = Once::new();

/// Install the ring provider as rustls's process-wide default, exactly once.
/// The dependency tree links both `ring` and `aws-lc-rs`, so rustls cannot pick
/// for itself and `ClientConfig::builder()` would panic. The install error
/// means another component won the race, and any installed provider will do.
fn ensure_crypto_provider() {
    INSTALL_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Pin `ca_pem` as the sole trust root: only a server cert chaining to it is
/// accepted, and rustls matches the endpoint host against the cert's SANs.
pub(crate) fn client_tls_config(ca_pem: &[u8]) -> ClientTlsConfig {
    ensure_crypto_provider();
    ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_pem))
}

/// A client-plane endpoint carrying the shared deadlines, TLS-pinned to
/// `ca_pem` when supplied, in which case the URL must be `https://`.
pub(crate) fn client_endpoint(url: &str, ca_pem: Option<&[u8]>) -> anyhow::Result<Endpoint> {
    let mut endpoint = Endpoint::from_shared(url.to_string())?
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT);
    if let Some(ca) = ca_pem {
        endpoint = endpoint.tls_config(client_tls_config(ca))?;
    }
    Ok(endpoint)
}

/// A custom tonic connector that opens the TCP stream and wraps it in
/// [`CountingIo`] before returning it. tonic layers its own rustls
/// `TlsConnector` on top of this IO when the endpoint carries a `tls_config`
/// and the URI is `https://`, so `counters` tallies ciphertext, not plaintext
/// gRPC bodies. The connector resolves host/port from the endpoint URI and
/// re-enables `TCP_NODELAY`, which the default tonic HTTP connector sets but a
/// custom connector otherwise drops.
fn counting_connector(
    counters: Arc<WireCounters>,
) -> impl tower::Service<
    Uri,
    Response = TokioIo<CountingIo<TcpStream>>,
    Error = std::io::Error,
    Future: Send,
> + Clone {
    tower::service_fn(move |uri: Uri| {
        let counters = counters.clone();
        async move {
            let host = uri
                .host()
                .ok_or_else(|| std::io::Error::other("endpoint uri has no host"))?
                .to_string();
            let port = uri
                .port_u16()
                .unwrap_or(if uri.scheme_str() == Some("https") {
                    443
                } else {
                    80
                });
            let tcp = TcpStream::connect((host.as_str(), port)).await?;
            tcp.set_nodelay(true).ok();
            Ok::<_, std::io::Error>(TokioIo::new(CountingIo::new(tcp, counters)))
        }
    })
}

/// Eagerly connect `endpoint` through the [`counting_connector`], so the
/// returned channel's socket bytes accrue to `counters`. Preserves the
/// endpoint's `connect_timeout`, per-RPC `timeout`, and TLS config.
pub(crate) async fn connect_counted(
    endpoint: Endpoint,
    counters: Arc<WireCounters>,
) -> anyhow::Result<Channel> {
    Ok(endpoint
        .connect_with_connector(counting_connector(counters))
        .await?)
}

/// Like [`connect_counted`] but lazy: the channel connects on first use, which
/// is what the best-effort fan-out needs so a shareholder that is down at
/// connect time does not abort the session.
pub(crate) fn connect_lazy_counted(endpoint: Endpoint, counters: Arc<WireCounters>) -> Channel {
    endpoint.connect_with_connector_lazy(counting_connector(counters))
}

#[derive(Clone)]
pub struct Bearer {
    header: MetadataValue<tonic::metadata::Ascii>,
}

impl Bearer {
    /// An interceptor that stamps `Bearer <token>` onto every RPC made over
    /// the channel it is attached to via `with_interceptor`.
    pub(crate) fn new(token: &str) -> anyhow::Result<Self> {
        Ok(Self {
            header: format!("Bearer {token}").parse()?,
        })
    }
}

impl Interceptor for Bearer {
    fn call(&mut self, mut req: Request<()>) -> Result<Request<()>, Status> {
        req.metadata_mut()
            .insert("authorization", self.header.clone());
        Ok(req)
    }
}

pub struct Shareholders {
    pub(crate) channels: Vec<
        ShareholderServiceClient<tonic::service::interceptor::InterceptedService<Channel, Bearer>>,
    >,
}

impl Shareholders {
    /// Connect lazily: each channel connects on first use, so a shareholder
    /// down now or dead later just fails its own RPCs, which the best-effort
    /// fan-out tolerates. Only a malformed endpoint URI or token is fatal here.
    /// `endpoints[i]` serves Shamir evaluation point x = i + 1, and `ca_pem`
    /// pins TLS on every channel.
    ///
    /// Every channel tallies its socket bytes into `counters`, the session's
    /// shared [`WireCounters`], which the aggregator channel also holds, so a
    /// round's delta spans the whole fan-out.
    pub fn connect_best_effort(
        endpoints: &[String],
        bearer_token: String,
        ca_pem: Option<&[u8]>,
        counters: Arc<WireCounters>,
    ) -> anyhow::Result<Self> {
        let bearer = Bearer::new(&bearer_token)?;
        let mut channels = Vec::with_capacity(endpoints.len());
        for ep in endpoints {
            let endpoint = client_endpoint(ep, ca_pem)?;
            let channel = connect_lazy_counted(endpoint, counters.clone());
            channels.push(ShareholderServiceClient::with_interceptor(
                channel,
                bearer.clone(),
            ));
        }
        Ok(Self { channels })
    }
}
