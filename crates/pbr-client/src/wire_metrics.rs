//! Client-side wire byte counters. [`crate::rpc`]'s connector wraps each TCP
//! stream in [`CountingIo`] *below* rustls, so the counters tally what actually
//! goes on the socket: on an `https://` endpoint the ciphertext, meaning TLS
//! records, the handshake and HTTP/2 framing rather than the gRPC bodies, since
//! tonic layers its `TlsConnector` above this IO. A plaintext endpoint has no
//! TLS layer, so the count is raw HTTP/2. The measured deployment is TLS.
//!
//! Counters are per session, not process-global: one process can host several
//! at once (the app's foreground UI and background isolate share one native
//! library), and a per-round delta must not absorb another session's traffic.
//! One `Arc` spans a session's aggregator channel and every shareholder
//! channel, so the delta covers poll and submit across all sockets.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

/// Byte tallies for every socket of one session; read the delta around a round
/// to measure its wire cost.
#[derive(Debug, Default)]
pub struct WireCounters {
    tx: AtomicU64,
    rx: AtomicU64,
}

impl WireCounters {
    /// Total bytes this session wrote to the network.
    pub fn tx(&self) -> u64 {
        self.tx.load(Ordering::Relaxed)
    }

    /// Total bytes this session read from the network.
    pub fn rx(&self) -> u64 {
        self.rx.load(Ordering::Relaxed)
    }
}

/// Wraps an IO, counting bytes read/written into the owning session's counters.
pub struct CountingIo<S> {
    inner: S,
    counters: Arc<WireCounters>,
}

impl<S> CountingIo<S> {
    pub fn new(inner: S, counters: Arc<WireCounters>) -> Self {
        Self { inner, counters }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for CountingIo<S> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let r = Pin::new(&mut self.inner).poll_read(cx, buf);
        if let Poll::Ready(Ok(())) = &r {
            self.counters
                .rx
                .fetch_add((buf.filled().len() - before) as u64, Ordering::Relaxed);
        }
        r
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for CountingIo<S> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let r = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = &r {
            self.counters.tx.fetch_add(*n as u64, Ordering::Relaxed);
        }
        r
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        let r = Pin::new(&mut self.inner).poll_write_vectored(cx, bufs);
        if let Poll::Ready(Ok(n)) = &r {
            self.counters.tx.fetch_add(*n as u64, Ordering::Relaxed);
        }
        r
    }
    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}
