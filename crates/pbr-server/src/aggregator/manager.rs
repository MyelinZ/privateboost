//! Every session this aggregator process is hosting.
//!
//! One `SessionState` and one round-loop task per session, keyed by the
//! session's id. The manager is the only thing that knows a session exists:
//! the gRPC service resolves an incoming `session_id` through it, and
//! `serve`'s shutdown aborts every round loop through it. Sessions leave
//! the map only through `remove`; short of that, a completed or failed
//! session stays listable so a device can see how a session it contributed
//! to turned out.
//!
//! A SQLite store persists just enough of that list (id, phase, spec, creation
//! time) to survive a restart: `persist` upserts one row, `persist_removal`
//! deletes one so a removed session does not relist, `load` rebuilds the map.
//! Everything else a live session holds (the published `RoundContext`, the
//! shareholder pools, the in-memory `pbr_core::Aggregator`) dies with the
//! process, so `load` demotes anything still `StatsPending` or `Training` to
//! `Failed` rather than resurrecting a session that can never resume.

use super::service::{SessionState, SharedSession};
use super::SessionSpec;
use pbr_proto::v1::SessionPhase;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::task::AbortHandle;

/// The handle is `None` for a session with no round loop (test fixtures).
struct SessionEntry {
    state: SharedSession,
    round_loop: Option<AbortHandle>,
}

/// The fields `ListSessions` reports, as plain data so the manager does not
/// depend on the proto message type.
pub(super) struct SessionSummaryData {
    pub session_id: String,
    pub phase: SessionPhase,
    pub n_features: u32,
    /// Empty for a dataset-less session; see `SessionSpec::dataset_id`.
    pub dataset_id: String,
    pub created_at: SystemTime,
}

/// What the delete handler needs once the entry is already out of the map, all
/// captured under the one lock acquisition that removed it.
pub(super) struct RemovedSession {
    pub state: SharedSession,
    pub round_loop: Option<AbortHandle>,
    pub was_live: bool,
}

/// `resolve` needs `Zero` and `Many` distinguished: their error messages tell
/// an operator two different things to do.
pub(super) enum Sole {
    One(SharedSession),
    Zero,
    Many(usize),
}

/// Not yet terminal. Shared by `live_count`, `try_insert_new` and `sole` so
/// all three reason about the same set.
fn is_live(e: &SessionEntry) -> bool {
    matches!(
        e.state.lock().unwrap().phase,
        SessionPhase::StatsPending | SessionPhase::Training
    )
}

#[derive(Clone)]
pub(super) struct SessionManager {
    sessions: Arc<Mutex<HashMap<String, SessionEntry>>>,
    /// Single-connection `SqlitePool`; see `load` for why one connection,
    /// never recycled.
    store: SqlitePool,
}

impl SessionManager {
    /// Shared with the notify module's `devices` and `enrollments` tables, so
    /// those callers inherit the same serialized-writes invariant.
    pub(super) fn store(&self) -> &SqlitePool {
        &self.store
    }

    /// Register a session directly, no cap check. Tests only.
    #[cfg(test)]
    pub(super) fn insert(
        &self,
        session_id: String,
        state: SharedSession,
        round_loop: Option<AbortHandle>,
    ) {
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id, SessionEntry { state, round_loop });
    }

    /// Callers turn `None` into `NOT_FOUND` rather than falling back to some
    /// "current" session, which would misattribute the caller's contributions.
    pub(super) fn get(&self, session_id: &str) -> Option<SharedSession> {
        self.sessions.lock().unwrap().get(session_id).map(|e| e.state.clone())
    }

    /// `was_live` is captured under the same lock acquisition as the removal,
    /// so the caller's abort-and-cleanup decision reflects the phase the
    /// session had when it left the map.
    pub(super) fn remove(&self, session_id: &str) -> Option<RemovedSession> {
        let mut map = self.sessions.lock().unwrap();
        let entry = map.remove(session_id)?;
        let was_live = is_live(&entry);
        Some(RemovedSession {
            state: entry.state,
            round_loop: entry.round_loop,
            was_live,
        })
    }

    /// Attach the round loop's abort handle to a registered session. `false`
    /// means a concurrent `remove` dropped it between registration and the
    /// spawn, and the caller must abort the loop rather than re-register it.
    pub(super) fn attach_round_loop(&self, session_id: &str, handle: AbortHandle) -> bool {
        match self.sessions.lock().unwrap().get_mut(session_id) {
            Some(e) => {
                e.round_loop = Some(handle);
                true
            }
            None => false,
        }
    }

    /// The one live session `resolve` serves an empty selector, else a count.
    /// Terminal sessions are excluded because they reload across restarts,
    /// which would make the empty selector ambiguous on any deployment that
    /// ever hosted two. Count and fetch share one lock acquisition, so a
    /// concurrent `CreateSession` cannot change the set between them.
    pub(super) fn sole(&self) -> Sole {
        let map = self.sessions.lock().unwrap();
        let live: Vec<&SessionEntry> = map.values().filter(|e| is_live(e)).collect();
        match live.len() {
            0 => Sole::Zero,
            1 => Sole::One(live[0].state.clone()),
            n => Sole::Many(n),
        }
    }

    /// What `CreateSession`'s cap counts against: each live session holds a
    /// share pool, a task and internal-plane connections open, while a terminal
    /// one is history and holds nothing.
    pub(super) fn live_count(&self) -> usize {
        self.sessions.lock().unwrap().values().filter(|e| is_live(e)).count()
    }

    /// Register a session unless that would push live sessions above `cap`.
    /// Count and insert share one lock acquisition, so callers racing
    /// `CreateSession` cannot each observe room and together overshoot.
    pub(super) fn try_insert_new(&self, cap: usize, session_id: String, state: SharedSession) -> bool {
        let mut map = self.sessions.lock().unwrap();
        if map.values().filter(|e| is_live(e)).count() >= cap {
            return false;
        }
        map.insert(session_id, SessionEntry { state, round_loop: None });
        true
    }

    pub(super) fn summaries(&self) -> Vec<SessionSummaryData> {
        self.sessions
            .lock()
            .unwrap()
            .values()
            .map(|e| {
                let s = e.state.lock().unwrap();
                SessionSummaryData {
                    session_id: s.session_id.clone(),
                    phase: s.phase,
                    n_features: s.n_features,
                    dataset_id: s.spec.dataset_id.clone(),
                    created_at: s.created_at,
                }
            })
            .collect()
    }

    /// Stop every round loop. Called from `AggregatorHandle::shutdown`.
    pub(super) fn abort_all(&self) {
        for e in self.sessions.lock().unwrap().values() {
            if let Some(h) = &e.round_loop {
                h.abort();
            }
        }
    }

    /// Upsert one session's row: id (primary key), phase, spec as JSON, and
    /// creation time in Unix seconds. A phase only advances and only its own
    /// task writes it, so each upsert is correct standing alone in its implicit
    /// transaction; there is no cross-session write order to coordinate.
    pub(super) async fn persist(&self, session: &SharedSession) -> anyhow::Result<()> {
        // A `std::sync::MutexGuard` must not be held across the await below.
        let (session_id, phase, spec, created_at) = {
            let s = session.lock().unwrap();
            (
                s.session_id.clone(),
                s.phase as i32,
                serde_json::to_string(&s.spec)?,
                s.created_at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
            )
        };
        sqlx::query(
            "INSERT INTO sessions (session_id, phase, spec, created_at)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(session_id) DO UPDATE SET
                 phase = excluded.phase,
                 spec = excluded.spec,
                 created_at = excluded.created_at",
        )
        .bind(session_id)
        .bind(phase)
        .bind(spec)
        .bind(created_at)
        .execute(&self.store)
        .await?;
        Ok(())
    }

    /// Delete one session's row, and its `enrollments` rows in the same
    /// transaction so a deleted session cannot keep waking devices. Idempotent,
    /// since `DELETE` of an absent id is a no-op: the `DeleteSession` handler
    /// and `spawn_session`'s race compensation may both target the same id.
    /// Without the delete the row would reload on the next `load` and relist.
    pub(super) async fn persist_removal(&self, session_id: &str) -> anyhow::Result<()> {
        let mut tx = self.store.begin().await?;
        sqlx::query("DELETE FROM sessions WHERE session_id = ?")
            .bind(session_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM enrollments WHERE session_id = ?")
            .bind(session_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Open the store at `path`, migrate, and rebuild the session map. A
    /// missing DB file is a normal first run; a migrate failure is fatal.
    ///
    /// A session recorded `StatsPending` or `Training` reloads as `Failed`: its
    /// share pools and in-memory aggregator died with the process, so reporting
    /// it live would strand devices polling a session that cannot advance.
    pub(super) async fn load(path: &Path) -> anyhow::Result<Self> {
        // `create_if_missing` makes a missing DB file but not its parent
        // directory, so a fresh checkout boots without a manual mkdir.
        // `:memory:` has no parent, and an empty one is the current directory.
        if path != Path::new(":memory:")
            && let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);

        // Two pool invariants, correctness rather than tuning. One connection
        // serializes per-row upserts, so each stays atomic with no SQLITE_BUSY
        // to reason about; disabling recycling pins that connection for the
        // process lifetime, which `":memory:"` requires because every new
        // connection there is a fresh empty database.
        let store = SqlitePoolOptions::new()
            .max_connections(1)
            .idle_timeout(None)
            .max_lifetime(None)
            .connect_with(options)
            .await?;

        sqlx::migrate!().run(&store).await?;

        let rows = sqlx::query("SELECT session_id, phase, spec, created_at FROM sessions")
            .fetch_all(&store)
            .await?;

        let mut sessions = HashMap::new();
        for row in rows {
            let session_id: String = row.try_get(0)?;
            let phase: i32 = row.try_get(1)?;
            let spec: String = row.try_get(2)?;
            let created_at: i64 = row.try_get(3)?;
            let phase = match SessionPhase::try_from(phase).unwrap_or(SessionPhase::Unspecified) {
                SessionPhase::StatsPending | SessionPhase::Training => SessionPhase::Failed,
                other => other,
            };
            let spec: SessionSpec = serde_json::from_str(&spec)?;
            let state: SharedSession = Arc::new(Mutex::new(SessionState {
                phase,
                ctx: None,
                bin_edges: Vec::new(),
                n_features: 0,
                session_id: session_id.clone(),
                spec,
                created_at: UNIX_EPOCH + Duration::from_secs(created_at as u64),
            }));
            sessions.insert(session_id, SessionEntry { state, round_loop: None });
        }

        Ok(Self {
            sessions: Arc::new(Mutex::new(sessions)),
            store,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field values are arbitrary: no test here reads them.
    fn test_spec() -> SessionSpec {
        SessionSpec {
            dataset_id: "d".into(),
            title: "t".into(),
            n_trees: 1,
            max_depth: 1,
            n_bins: 4,
            learning_rate: 0.1,
            lambda: 1.0,
            min_clients: 1,
            target_clients: 1,
            submission_window_ms: 1000,
        }
    }

    fn session_in(id: &str, phase: SessionPhase) -> SharedSession {
        Arc::new(Mutex::new(SessionState {
            phase,
            ctx: None,
            bin_edges: Vec::new(),
            n_features: 0,
            session_id: id.into(),
            spec: test_spec(),
            created_at: SystemTime::now(),
        }))
    }

    fn new_shared_session(id: &str) -> SharedSession {
        session_in(id, SessionPhase::StatsPending)
    }

    /// Backed by an ephemeral, file-less store.
    async fn in_memory() -> SessionManager {
        SessionManager::load(Path::new(":memory:")).await.unwrap()
    }

    #[tokio::test]
    async fn manager_keeps_sessions_separate_and_lists_them() {
        let mgr = in_memory().await;
        let a = new_shared_session("sess-a");
        let b = new_shared_session("sess-b");
        mgr.insert("sess-a".into(), a.clone(), None);
        mgr.insert("sess-b".into(), b.clone(), None);

        // Publishing into one session must not be visible through the other.
        a.lock().unwrap().phase = SessionPhase::Training;

        assert_eq!(
            mgr.get("sess-a").unwrap().lock().unwrap().phase,
            SessionPhase::Training
        );
        assert_eq!(
            mgr.get("sess-b").unwrap().lock().unwrap().phase,
            SessionPhase::StatsPending,
            "one session's phase change must not leak into another"
        );
        assert!(mgr.get("sess-missing").is_none());

        let mut ids: Vec<String> = mgr.summaries().into_iter().map(|s| s.session_id).collect();
        ids.sort();
        assert_eq!(ids, vec!["sess-a".to_string(), "sess-b".to_string()]);
    }

    #[tokio::test]
    async fn live_count_excludes_completed_and_failed_sessions() {
        let mgr = in_memory().await;

        mgr.insert("sess-training".into(), session_in("sess-training", SessionPhase::Training), None);
        mgr.insert("sess-completed".into(), session_in("sess-completed", SessionPhase::Completed), None);
        mgr.insert("sess-failed".into(), session_in("sess-failed", SessionPhase::Failed), None);

        assert_eq!(
            mgr.live_count(),
            1,
            "only the Training session is live; Completed and Failed are history"
        );
    }

    #[tokio::test]
    async fn sole_ignores_terminal_sessions_so_the_empty_selector_still_resolves() {
        // Terminal sessions persist across restarts and reload into the map,
        // so a manager that has ever hosted more than one session would leave
        // the empty selector ambiguous forever unless `sole` counts only live
        // sessions.
        let mgr = in_memory().await;
        mgr.insert("live".into(), new_shared_session("live"), None);
        mgr.insert("done".into(), session_in("done", SessionPhase::Completed), None);
        mgr.insert("failed".into(), session_in("failed", SessionPhase::Failed), None);
        mgr.insert("done2".into(), session_in("done2", SessionPhase::Completed), None);

        match mgr.sole() {
            Sole::One(state) => assert_eq!(
                state.lock().unwrap().session_id,
                "live",
                "the one live session is the empty selector's target"
            ),
            Sole::Zero => panic!("one live session must resolve; saw none"),
            Sole::Many(n) => panic!("one live session must resolve; saw {n} candidates"),
        }
    }

    #[tokio::test]
    async fn try_insert_new_refuses_once_the_cap_is_reached() {
        let mgr = in_memory().await;
        assert!(mgr.try_insert_new(2, "a".into(), new_shared_session("a")));
        assert!(mgr.try_insert_new(2, "b".into(), new_shared_session("b")));
        assert!(
            !mgr.try_insert_new(2, "c".into(), new_shared_session("c")),
            "a third live session must be refused once the cap of 2 is already hosted"
        );
        assert_eq!(mgr.live_count(), 2, "the refused session must not have been inserted");
    }

    #[tokio::test]
    async fn concurrent_try_insert_new_calls_never_overshoot_the_cap() {
        // N callers racing `CreateSession` while the cluster already hosts
        // `cap - 1` live sessions must not all observe room below the cap and
        // together overshoot it. `try_insert_new` counts and inserts under one
        // lock acquisition, so this holds regardless of thread scheduling, not
        // just probabilistically.
        let mgr = in_memory().await;
        let cap = 8;
        let attempts = 64;
        let handles: Vec<_> = (0..attempts)
            .map(|i| {
                let mgr = mgr.clone();
                std::thread::spawn(move || {
                    let id = format!("sess-{i}");
                    mgr.try_insert_new(cap, id.clone(), new_shared_session(&id))
                })
            })
            .collect();
        let successes = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&ok| ok)
            .count();
        assert_eq!(successes, cap, "exactly cap insertions may succeed, no matter how many racers");
        assert_eq!(mgr.live_count(), cap);
    }

    #[tokio::test]
    async fn a_session_in_flight_at_shutdown_reloads_as_failed() {
        // Its shareholder pools and in-memory aggregator state are gone, so it
        // can never be resumed; reporting it as still training would strand
        // devices polling a session that will never advance. A finished
        // session keeps its recorded outcome, and its spec round-trips.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sessions.sqlite");

        let mgr = SessionManager::load(&path).await.unwrap();
        let live = session_in("live", SessionPhase::Training);
        let done = session_in("done", SessionPhase::Completed);
        mgr.insert("live".into(), live.clone(), None);
        mgr.insert("done".into(), done.clone(), None);
        mgr.persist(&live).await.unwrap();
        mgr.persist(&done).await.unwrap();

        let reloaded = SessionManager::load(&path).await.unwrap();
        let summaries = reloaded.summaries();
        let by_id: HashMap<_, _> = summaries
            .iter()
            .map(|s| (s.session_id.clone(), s.phase))
            .collect();
        assert_eq!(by_id["live"], SessionPhase::Failed, "in-flight cannot survive a restart");
        assert_eq!(by_id["done"], SessionPhase::Completed, "a finished session keeps its outcome");
        assert!(
            summaries.iter().all(|s| s.dataset_id == "d"),
            "the spec column must round-trip through the store"
        );
    }

    #[tokio::test]
    async fn loading_a_missing_state_file_is_an_empty_manager_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = SessionManager::load(&dir.path().join("nope.sqlite")).await.unwrap();
        assert!(mgr.summaries().is_empty(), "a first run has no history");
    }

    #[tokio::test]
    async fn a_pre_sqlx_store_without_the_migrations_table_adopts_cleanly() {
        // A store written before schema migrations existed has the `sessions`
        // table but no `_sqlx_migrations` bookkeeping. The first migration must
        // adopt that table in place (`CREATE TABLE IF NOT EXISTS`) rather than
        // fail on the pre-existing table, and must record itself so a later
        // process sees the schema as versioned.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.sqlite");

        // Seed a store holding the `sessions` table alone, no migration ledger.
        {
            let options = SqliteConnectOptions::new()
                .filename(&path)
                .create_if_missing(true);
            let pool = SqlitePoolOptions::new()
                .max_connections(1)
                .connect_with(options)
                .await
                .unwrap();
            sqlx::query(
                "CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    phase INTEGER NOT NULL,
                    spec TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )",
            )
            .execute(&pool)
            .await
            .unwrap();
            pool.close().await;
        }

        // Adoption: the load succeeds against the pre-existing table and reads
        // it back empty.
        let mgr = SessionManager::load(&path).await.unwrap();
        assert!(mgr.summaries().is_empty());

        // The migrations are now recorded.
        let options = SqliteConnectOptions::new().filename(&path);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();
        let applied: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM _sqlx_migrations")
            .fetch_one(&pool)
            .await
            .unwrap();
        pool.close().await;
        assert_eq!(applied, 2, "adopting a pre-sqlx store records all migrations");
    }

    #[tokio::test]
    async fn concurrent_persists_through_one_store_all_succeed() {
        // The checkpoint call sites (session creation, round-loop terminal,
        // panic monitor) can persist different sessions concurrently through
        // the one shared store; the single-connection pool serializes them, so
        // every write lands and none errors.
        let mgr = in_memory().await;
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let mgr = mgr.clone();
                tokio::spawn(async move {
                    let id = format!("sess-{i}");
                    let s = new_shared_session(&id);
                    mgr.insert(id, s.clone(), None);
                    mgr.persist(&s).await
                })
            })
            .collect();
        for h in handles {
            h.await.unwrap().expect("a concurrent persist must not fail");
        }
    }

    #[tokio::test]
    async fn remove_drops_the_session_and_reports_liveness_at_removal_time() {
        let mgr = in_memory().await;
        mgr.insert("live".into(), session_in("live", SessionPhase::Training), None);
        mgr.insert("done".into(), session_in("done", SessionPhase::Completed), None);

        let removed = mgr.remove("done").expect("a hosted session must be removable");
        assert!(!removed.was_live, "Completed is terminal");
        let removed = mgr.remove("live").expect("a live session must be removable too");
        assert!(removed.was_live, "Training is live");

        assert!(mgr.remove("live").is_none(), "a second remove must find nothing");
        assert!(mgr.summaries().is_empty(), "both sessions must be gone from the list");
    }

    #[tokio::test]
    async fn a_removed_session_stays_gone_across_persist_and_reload() {
        // A per-row store keeps every row until something deletes it, so a
        // `remove` that only drops the map entry would let the row reload and
        // relist after a restart. Dropping the `persist_removal("gone")` call
        // below makes "gone" reload here.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sessions.sqlite");

        let mgr = SessionManager::load(&path).await.unwrap();
        let keep = session_in("keep", SessionPhase::Completed);
        let gone = session_in("gone", SessionPhase::Failed);
        mgr.insert("keep".into(), keep.clone(), None);
        mgr.insert("gone".into(), gone.clone(), None);
        mgr.persist(&keep).await.unwrap();
        mgr.persist(&gone).await.unwrap();

        mgr.remove("gone").expect("must be removable");
        mgr.persist_removal("gone").await.unwrap();

        let ids: Vec<String> = SessionManager::load(&path)
            .await
            .unwrap()
            .summaries()
            .into_iter()
            .map(|s| s.session_id)
            .collect();
        assert_eq!(ids, vec!["keep".to_string()], "only the kept session may reload");
    }

    #[tokio::test]
    async fn creation_upsert_compensates_when_the_session_was_removed_concurrently() {
        // The create/delete race `spawn_session` closes: the creation upsert
        // runs as the loop task's first await, so a concurrent `DeleteSession`
        // can remove the map entry ahead of it. The post-upsert compensation
        // check must leave no row, or the session resurrects on reload.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sessions.sqlite");
        let mgr = SessionManager::load(&path).await.unwrap();

        // The loop task's creation upsert lands...
        let s = new_shared_session("racer");
        mgr.insert("racer".into(), s.clone(), None);
        mgr.persist(&s).await.unwrap();
        // ...but a concurrent DeleteSession removed the map entry in between.
        mgr.remove("racer").expect("must be removable");
        // The compensation check `spawn_session` runs after the upsert: the
        // session is no longer registered, so the row it just wrote is deleted.
        if mgr.get("racer").is_none() {
            mgr.persist_removal("racer").await.unwrap();
        }

        let reloaded = SessionManager::load(&path).await.unwrap();
        assert!(
            reloaded.summaries().is_empty(),
            "a session removed while its creation upsert was in flight must not resurrect"
        );
    }

    #[tokio::test]
    async fn terminal_upsert_compensates_when_the_session_was_removed_concurrently() {
        // The terminal flavor of the create/delete race: run_session publishes
        // the phase, broadcasts EndSession, then upserts the row, and
        // `DeleteSession` does not abort a terminal loop, so a delete can land
        // in that window. The post-upsert compensation check must leave no row
        // under either interleaving; see `checkpoint_and_compensate`.
        let dir = tempfile::tempdir().unwrap();

        // Ordering A: upsert, then the handler's remove and row-delete, then
        // the compensation clears the resurrected row.
        let path_a = dir.path().join("a.sqlite");
        let mgr = SessionManager::load(&path_a).await.unwrap();
        let s = session_in("racer", SessionPhase::Completed);
        mgr.insert("racer".into(), s.clone(), None);
        mgr.persist(&s).await.unwrap();
        mgr.remove("racer").expect("must be removable");
        mgr.persist_removal("racer").await.unwrap();
        if mgr.get("racer").is_none() {
            mgr.persist_removal("racer").await.unwrap();
        }
        assert!(
            SessionManager::load(&path_a).await.unwrap().summaries().is_empty(),
            "a terminal session deleted while its upsert was in flight must not resurrect"
        );

        // Ordering B: the handler's remove and row-delete land first, then the
        // upsert, whose own compensation clears it.
        let path_b = dir.path().join("b.sqlite");
        let mgr = SessionManager::load(&path_b).await.unwrap();
        let s = session_in("racer", SessionPhase::Failed);
        mgr.insert("racer".into(), s.clone(), None);
        mgr.remove("racer").expect("must be removable");
        mgr.persist_removal("racer").await.unwrap();
        mgr.persist(&s).await.unwrap();
        if mgr.get("racer").is_none() {
            mgr.persist_removal("racer").await.unwrap();
        }
        assert!(
            SessionManager::load(&path_b).await.unwrap().summaries().is_empty(),
            "the inverse ordering must also converge to no row"
        );
    }

    #[tokio::test]
    async fn attach_round_loop_refuses_a_session_removed_in_between() {
        // A `remove` landing between registration and the loop spawn must not
        // be undone by the attach: it must report the session gone so the
        // caller aborts the loop instead.
        let mgr = in_memory().await;
        mgr.insert("sess".into(), session_in("sess", SessionPhase::StatsPending), None);

        let task = tokio::spawn(async {});
        assert!(
            mgr.attach_round_loop("sess", task.abort_handle()),
            "attaching to a registered session must succeed"
        );

        mgr.remove("sess").expect("must be removable");
        let late = tokio::spawn(async {});
        assert!(
            !mgr.attach_round_loop("sess", late.abort_handle()),
            "attaching to a removed session must fail so the caller aborts the loop"
        );
        assert!(mgr.summaries().is_empty(), "the failed attach must not re-register anything");
    }

    #[tokio::test]
    async fn persist_removal_purges_the_sessions_enrollments() {
        let m = in_memory().await;
        sqlx::query(
            "INSERT INTO enrollments (session_id, uid, enrolled_at)
             VALUES ('s1', 'u1', 0), ('s1', 'u2', 0), ('s2', 'u1', 0)",
        )
        .execute(m.store())
        .await
        .unwrap();

        m.persist_removal("s1").await.unwrap();

        let left: Vec<(String, String)> =
            sqlx::query_as("SELECT session_id, uid FROM enrollments")
                .fetch_all(m.store())
                .await
                .unwrap();
        assert_eq!(
            left,
            vec![("s2".to_string(), "u1".to_string())],
            "only the other session's enrollment survives"
        );
    }
}
