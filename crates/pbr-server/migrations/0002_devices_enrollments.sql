CREATE TABLE devices (
    uid              TEXT PRIMARY KEY,
    fcm_token        TEXT NOT NULL,
    platform         INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL,
    last_notified_at INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE enrollments (
    session_id  TEXT NOT NULL,
    uid         TEXT NOT NULL,
    enrolled_at INTEGER NOT NULL,
    PRIMARY KEY (session_id, uid)
);
