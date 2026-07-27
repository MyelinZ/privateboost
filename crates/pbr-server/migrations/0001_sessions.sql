CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    phase INTEGER NOT NULL,
    spec TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
