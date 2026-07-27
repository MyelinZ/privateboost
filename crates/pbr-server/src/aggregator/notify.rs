//! Round-open push: persisted device registry, session enrollments, and the
//! throttled notify tick.
//!
//! `RegisterDevice` upserts `devices` (verified uid to FCM token, plus the
//! per-account `last_notified_at` stamp); `EnrollSession` records
//! `enrollments`. The notify tick reads both, pushes to every enrolled account
//! whose floor has elapsed while a session has open work, and stamps before
//! sending so a crash or slow send can never re-burst pushes.

use crate::auth;
use crate::fcm::FcmSender;
use pbr_proto::v1::SessionPhase;
use sqlx::{Row, SqlitePool};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The registry key: the verified issuer+subject from the JWT interceptor's
/// `Identity`, never a client-supplied value.
pub(super) fn device_uid(identity: &auth::Identity) -> String {
    format!("{}|{}", identity.issuer, identity.subject)
}

pub struct DeviceRow {
    pub uid: String,
    pub fcm_token: String,
    pub platform: i32,
    pub last_notified_at: i64,
}

pub(super) async fn upsert_device(
    store: &SqlitePool,
    uid: &str,
    fcm_token: &str,
    platform: i32,
    now_secs: i64,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO devices (uid, fcm_token, platform, updated_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(uid) DO UPDATE SET
             fcm_token = excluded.fcm_token,
             platform = excluded.platform,
             updated_at = excluded.updated_at",
    )
    .bind(uid)
    .bind(fcm_token)
    .bind(platform)
    .bind(now_secs)
    .execute(store)
    .await?;
    Ok(())
}

pub(super) async fn record_enrollment(
    store: &SqlitePool,
    session_id: &str,
    uid: &str,
    now_secs: i64,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO enrollments (session_id, uid, enrolled_at)
         VALUES (?, ?, ?)
         ON CONFLICT(session_id, uid) DO UPDATE SET
             enrolled_at = excluded.enrolled_at",
    )
    .bind(session_id)
    .bind(uid)
    .bind(now_secs)
    .execute(store)
    .await?;
    Ok(())
}

pub(super) async fn all_devices(store: &SqlitePool) -> anyhow::Result<Vec<DeviceRow>> {
    let rows = sqlx::query(
        "SELECT uid, fcm_token, platform, last_notified_at FROM devices",
    )
    .fetch_all(store)
    .await?;
    rows.into_iter()
        .map(|r| {
            Ok(DeviceRow {
                uid: r.try_get(0)?,
                fcm_token: r.try_get(1)?,
                platform: r.try_get(2)?,
                last_notified_at: r.try_get(3)?,
            })
        })
        .collect()
}

pub(super) async fn all_enrollments(
    store: &SqlitePool,
) -> anyhow::Result<Vec<(String, String)>> {
    let rows = sqlx::query("SELECT session_id, uid FROM enrollments")
        .fetch_all(store)
        .await?;
    rows.into_iter()
        .map(|r| Ok((r.try_get(0)?, r.try_get(1)?)))
        .collect()
}

/// One UPDATE per uid on the single-connection pool: the due set is small and
/// per-statement writes stay serialized, as session rows are.
pub(super) async fn stamp_notified(
    store: &SqlitePool,
    uids: &[String],
    now_secs: i64,
) -> anyhow::Result<()> {
    for uid in uids {
        sqlx::query("UPDATE devices SET last_notified_at = ? WHERE uid = ?")
            .bind(now_secs)
            .bind(uid)
            .execute(store)
            .await?;
    }
    Ok(())
}

/// `(uid, fcm_token)` pairs due a wake: enrolled in a notifiable session and at
/// or past the per-account floor. Order follows `devices`, and a uid appears
/// once however many sessions it is enrolled in, since the woken app steps them
/// all itself. Pure over its inputs, so the throttle policy is testable without
/// a store or sender; the caller stamps the returned uids before any send.
pub(super) fn due_for_notify(
    notifiable: &HashSet<String>,
    enrollments: &[(String, String)],
    devices: &[DeviceRow],
    now_secs: i64,
    floor_secs: i64,
) -> Vec<(String, String)> {
    let interested: HashSet<&str> = enrollments
        .iter()
        .filter(|(sid, _)| notifiable.contains(sid))
        .map(|(_, uid)| uid.as_str())
        .collect();
    devices
        .iter()
        .filter(|d| interested.contains(d.uid.as_str()))
        .filter(|d| now_secs.saturating_sub(d.last_notified_at) >= floor_secs)
        .map(|d| (d.uid.clone(), d.fcm_token.clone()))
        .collect()
}

/// FCM data values are always strings. `sentAt` is the aggregator's
/// epoch-millisecond send stamp, which the device subtracts from its own clock
/// to report perceived wake latency, clock skew included. `kind` selects the
/// app's round_open handlers and must stay `"round_open"`. Nothing else: the
/// woken app lists sessions itself, so per-round fields would be dead weight
/// once more than one session is notifiable.
pub(super) fn wake_data(sent_at_ms: u128) -> HashMap<String, String> {
    HashMap::from([
        ("kind".to_string(), "round_open".to_string()),
        ("sentAt".to_string(), sent_at_ms.to_string()),
    ])
}

/// How often the notify loop re-plans. Worst-case first-push latency after a
/// round opens; the floor, not this, sets the send rate.
pub(super) const NOTIFY_TICK_PERIOD: Duration = Duration::from_secs(60);

/// One planning pass: while any session has open work, push to every enrolled
/// account at or past the floor. `last_notified_at` is stamped for the whole
/// due set before the first send, so a slow send, a failed send or a crash
/// in between costs at most one missed wake rather than a burst. Sends run
/// sequentially on this task, each bounded by the sender's request timeout,
/// and nothing waits on the tick, so a slow FCM endpoint only delays the next
/// one.
pub(super) async fn run_notify_tick(
    store: &SqlitePool,
    sessions: &[(String, SessionPhase)],
    fcm: &FcmSender,
    now_secs: i64,
    floor_secs: i64,
) -> anyhow::Result<usize> {
    let notifiable: HashSet<String> = sessions
        .iter()
        .filter(|(_, phase)| {
            matches!(phase, SessionPhase::StatsPending | SessionPhase::Training)
        })
        .map(|(id, _)| id.clone())
        .collect();
    if notifiable.is_empty() {
        return Ok(0);
    }

    let enrollments = all_enrollments(store).await?;
    let devices = all_devices(store).await?;
    let due = due_for_notify(&notifiable, &enrollments, &devices, now_secs, floor_secs);
    if due.is_empty() {
        return Ok(0);
    }

    let uids: Vec<String> = due.iter().map(|(uid, _)| uid.clone()).collect();
    stamp_notified(store, &uids, now_secs).await?;

    let sent_at_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut failed = 0usize;
    for (_, fcm_token) in &due {
        if let Err(e) = fcm.send_data(fcm_token, wake_data(sent_at_ms)).await {
            failed += 1;
            tracing::warn!(error = %e, "round_open push failed for a device; continuing");
        }
    }
    if failed > 0 {
        tracing::warn!(failed, total = due.len(), "some round_open pushes failed");
    }
    Ok(due.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregator::manager::SessionManager;
    use std::path::Path;

    async fn mem() -> SessionManager {
        SessionManager::load(Path::new(":memory:")).await.unwrap()
    }

    #[tokio::test]
    async fn reregister_preserves_last_notified_at() {
        let m = mem().await;
        upsert_device(m.store(), "u1", "tok-a", 1, 100).await.unwrap();
        stamp_notified(m.store(), &["u1".to_string()], 500).await.unwrap();

        // Re-register with a rotated token: the stamp must survive, or a
        // re-register would bypass the floor.
        upsert_device(m.store(), "u1", "tok-b", 1, 600).await.unwrap();

        let devices = all_devices(m.store()).await.unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].fcm_token, "tok-b");
        assert_eq!(devices[0].last_notified_at, 500);
    }

    #[tokio::test]
    async fn enrollment_upsert_refreshes_without_duplicating() {
        let m = mem().await;
        record_enrollment(m.store(), "s1", "u1", 100).await.unwrap();
        record_enrollment(m.store(), "s1", "u1", 200).await.unwrap();
        record_enrollment(m.store(), "s2", "u1", 300).await.unwrap();

        let mut rows = all_enrollments(m.store()).await.unwrap();
        rows.sort();
        assert_eq!(
            rows,
            vec![
                ("s1".to_string(), "u1".to_string()),
                ("s2".to_string(), "u1".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn stamp_notified_touches_only_the_given_uids() {
        let m = mem().await;
        upsert_device(m.store(), "u1", "t1", 1, 0).await.unwrap();
        upsert_device(m.store(), "u2", "t2", 1, 0).await.unwrap();
        stamp_notified(m.store(), &["u1".to_string()], 900).await.unwrap();

        let devices = all_devices(m.store()).await.unwrap();
        let by_uid: std::collections::HashMap<_, _> =
            devices.into_iter().map(|d| (d.uid.clone(), d)).collect();
        assert_eq!(by_uid["u1"].last_notified_at, 900);
        assert_eq!(by_uid["u2"].last_notified_at, 0);
    }

    fn dev(uid: &str, tok: &str, last: i64) -> DeviceRow {
        DeviceRow {
            uid: uid.into(),
            fcm_token: tok.into(),
            platform: 1,
            last_notified_at: last,
        }
    }

    #[test]
    fn due_excludes_unenrolled_and_cooling_accounts() {
        let notifiable: HashSet<String> = ["s1".to_string()].into();
        let enrollments = vec![
            ("s1".to_string(), "due".to_string()),
            ("s1".to_string(), "cooling".to_string()),
            ("s-finished".to_string(), "other-session-only".to_string()),
        ];
        let devices = vec![
            dev("due", "tok-due", 0),
            dev("cooling", "tok-cool", 950),
            dev("other-session-only", "tok-other", 0),
            dev("never-enrolled", "tok-never", 0),
        ];
        // now 1000, floor 900: "due" (1000-0 >= 900) is in; "cooling" (50 < 900)
        // is out; the other two are not enrolled in a notifiable session.
        let due = due_for_notify(&notifiable, &enrollments, &devices, 1_000, 900);
        assert_eq!(due, vec![("due".to_string(), "tok-due".to_string())]);
    }

    #[test]
    fn due_dedupes_across_sessions_and_treats_the_floor_as_inclusive() {
        let notifiable: HashSet<String> = ["s1".to_string(), "s2".to_string()].into();
        let enrollments = vec![
            ("s1".to_string(), "u1".to_string()),
            ("s2".to_string(), "u1".to_string()),
        ];
        let devices = vec![dev("u1", "t1", 100)];
        // Exactly floor seconds elapsed (1000 - 100 = 900) is due, once, even
        // though u1 is enrolled in two notifiable sessions.
        let due = due_for_notify(&notifiable, &enrollments, &devices, 1_000, 900);
        assert_eq!(due, vec![("u1".to_string(), "t1".to_string())]);
        // One second earlier it is not.
        assert!(due_for_notify(&notifiable, &enrollments, &devices, 999, 900).is_empty());
    }

    #[test]
    fn wake_data_is_exactly_kind_and_parseable_sent_at() {
        let data = wake_data(1_700_000_000_123);
        assert_eq!(data.len(), 2, "payload is exactly kind + sentAt");
        assert_eq!(data["kind"], "round_open");
        let sent_at: u128 = data["sentAt"].parse().expect("sentAt must parse");
        assert_eq!(sent_at, 1_700_000_000_123);
    }

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One request at a time: counts hits, answers 200 `{}`.
    async fn stub_fcm(hits: Arc<AtomicUsize>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut data = Vec::new();
                let mut buf = vec![0u8; 1024];
                loop {
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    data.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&data);
                    if let Some(hdr_end) = text.find("\r\n\r\n") {
                        let len = text
                            .lines()
                            .find_map(|l| {
                                l.to_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|v| v.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if data.len() >= hdr_end + 4 + len {
                            break;
                        }
                    }
                }
                hits.fetch_add(1, Ordering::SeqCst);
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
                    .await;
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn tick_sends_once_then_respects_the_floor() {
        let m = mem().await;
        upsert_device(m.store(), "u1", "tok1", 1, 0).await.unwrap();
        record_enrollment(m.store(), "s1", "u1", 0).await.unwrap();

        let hits = Arc::new(AtomicUsize::new(0));
        let base = stub_fcm(hits.clone()).await;
        let fcm = FcmSender::for_tests("proj", &base, "tok", Duration::from_secs(900));
        let training = vec![("s1".to_string(), SessionPhase::Training)];

        // First tick: due (1000 - 0 >= 900), one push, stamped at 1000.
        assert_eq!(
            run_notify_tick(m.store(), &training, &fcm, 1_000, 900).await.unwrap(),
            1
        );
        // Inside the floor: silent.
        assert_eq!(
            run_notify_tick(m.store(), &training, &fcm, 1_500, 900).await.unwrap(),
            0
        );
        // Floor elapsed: pushes again.
        assert_eq!(
            run_notify_tick(m.store(), &training, &fcm, 2_000, 900).await.unwrap(),
            1
        );
        // No notifiable session: silent regardless of the clock.
        let done = vec![("s1".to_string(), SessionPhase::Completed)];
        assert_eq!(
            run_notify_tick(m.store(), &done, &fcm, 9_999, 900).await.unwrap(),
            0
        );
        assert_eq!(hits.load(Ordering::SeqCst), 2, "exactly two pushes hit FCM");
    }

    #[tokio::test]
    async fn a_failed_send_still_stamps_at_most_once() {
        let m = mem().await;
        upsert_device(m.store(), "u1", "tok1", 1, 0).await.unwrap();
        record_enrollment(m.store(), "s1", "u1", 0).await.unwrap();
        // Nothing listens here: every send fails.
        let fcm = FcmSender::for_tests("proj", "http://127.0.0.1:1", "tok", Duration::from_secs(900));
        let training = vec![("s1".to_string(), SessionPhase::Training)];

        // The failed attempt still counts as this window's push...
        assert_eq!(
            run_notify_tick(m.store(), &training, &fcm, 1_000, 900).await.unwrap(),
            1
        );
        // ...so the account stays on cooldown rather than re-bursting.
        assert_eq!(
            run_notify_tick(m.store(), &training, &fcm, 1_100, 900).await.unwrap(),
            0
        );
    }
}
