import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

const _storeFileName = 'joined_sessions.json';
const _lockFileName = 'joined_sessions.lock';

/// Cross-isolate exclusion for [JoinedSessions]'s read-modify-write, on the
/// same atomic-exclusive-create primitive as `ParticipationLock`, which
/// explains why an in-process flag and [RandomAccessFile.lock] both fail here.
/// Its own lock file, not the participation lock: that one means "a
/// participation is in flight", while [JoinedSessions.join] runs from a UI tap
/// with none, so sharing it would block an unrelated wake.
///
/// A mutate is a few bytes of JSON rather than a minutes-long participation, so
/// both timeouts are far smaller: a lock older than [_staleAfter] is an isolate
/// that died mid-write, and [acquire] gives up after [_maxWait].
class _StoreLock {
  _StoreLock._(this._file, this._token);

  final File _file;
  final String _token;

  static const Duration _staleAfter = Duration(seconds: 10);
  static const Duration _maxWait = Duration(seconds: 2);
  static const Duration _retryDelay = Duration(milliseconds: 10);
  static const Duration _reclaimSettle = Duration(milliseconds: 50);

  /// Retries while a live isolate holds it. Returns null, never throwing and
  /// never waiting past [_maxWait], if the holder does not free it in time.
  static Future<_StoreLock?> acquire(Directory directory) async {
    final lockFile = File('${directory.path}/$_lockFileName');
    final deadline = DateTime.now().add(_maxWait);
    while (true) {
      final lock = await _tryOnce(lockFile);
      if (lock != null) return lock;
      if (DateTime.now().isAfter(deadline)) return null;
      await Future<void>.delayed(_retryDelay);
    }
  }

  static Future<_StoreLock?> _tryOnce(File lockFile) async {
    final token = _mintToken();
    try {
      await lockFile.create(exclusive: true);
      await lockFile.writeAsString(token, flush: true);
      return _StoreLock._(lockFile, token);
    } on FileSystemException {
      // Held by another isolate. Reclaim only if it looks abandoned; a fresh
      // lock means a real mutate is in flight, so the caller retries instead.
      try {
        final age = DateTime.now().difference(await lockFile.lastModified());
        if (age < _staleAfter) return null;
        return await _reclaim(lockFile, token);
      } on FileSystemException {
        return null;
      }
    }
  }

  /// Steals an abandoned lock without stealing one that turned fresh
  /// underneath us: the delete/create/settle/reread sequence
  /// `ParticipationLock._reclaim` explains.
  static Future<_StoreLock?> _reclaim(File lockFile, String token) async {
    try {
      await lockFile.delete();
      await lockFile.create(exclusive: true);
      await lockFile.writeAsString(token, flush: true);
    } on FileSystemException {
      return null;
    }
    await Future<void>.delayed(_reclaimSettle);
    try {
      if (await lockFile.readAsString() != token) return null;
    } on FileSystemException {
      return null;
    }
    return _StoreLock._(lockFile, token);
  }

  static String _mintToken() {
    final rng = Random.secure();
    final bytes = List<int>.generate(16, (_) => rng.nextInt(256));
    return bytes.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
  }

  /// Releases the lock so the next caller can proceed. Idempotent. Deletes
  /// the file only while it still carries our token, so a lock already
  /// reclaimed as stale by another isolate is left for its new owner.
  Future<void> release() async {
    try {
      if (await _file.readAsString() == _token) await _file.delete();
    } on FileSystemException {
      // Already gone (e.g. reclaimed as stale by another isolate).
    }
  }
}

/// Durable, per-account record of which sessions this device has joined.
///
/// Joining is the user's standing consent to contribute, so the record must
/// survive restarts and be readable from all three isolates that act on it
/// (foreground UI, FCM handler, WorkManager task) without a platform channel of
/// its own, hence a plain JSON file rather than `SharedPreferences`. Keyed by
/// Firebase uid, so signing in as another account starts empty instead of
/// inheriting the previous one's consent.
///
/// [ids] is permanent once joined, so the UI can still show "you contributed to
/// this" afterwards; [active] also excludes sessions [markFinished] recorded as
/// done, which must drop out of the wake loop.
///
/// On disk: `{uid: {sessionId: {"finished": bool}}}`. The per-session map is
/// not a bare bool so another value can join it later without a format
/// migration.
///
/// Consent bookkeeping must never break participation, so a missing or corrupt
/// file reads as an empty store and a failed write is logged and swallowed.
///
/// [join], [markFinished] and [clear] hold [_StoreLock] across their whole
/// read-modify-write. The three isolates share no memory, so without it a
/// `markFinished` racing a `join` reads the same snapshot and one write
/// clobbers the other, losing exactly the consent this class records. A caller
/// that cannot get the lock drops its write and can retry.
class JoinedSessions {
  JoinedSessions({@visibleForTesting this._directory});

  final Directory? _directory;

  Future<Directory> _dir() async =>
      _directory ?? await getApplicationSupportDirectory();

  Future<File> _file() async => File('${(await _dir()).path}/$_storeFileName');

  Future<Set<String>> ids(String uid) async {
    final store = await _read();
    return store[uid]?.keys.toSet() ?? {};
  }

  Future<Set<String>> active(String uid) async {
    final store = await _read();
    final sessions = store[uid] ?? {};
    return {
      for (final entry in sessions.entries)
        if (entry.value['finished'] != true) entry.key,
    };
  }

  Future<void> join(String uid, String sessionId) async {
    await _mutate((store) {
      final sessions = store.putIfAbsent(uid, () => {});
      sessions.putIfAbsent(sessionId, () => {'finished': false});
    });
  }

  Future<void> markFinished(String uid, String sessionId) async {
    await _mutate((store) {
      final session = store[uid]?[sessionId];
      if (session != null) session['finished'] = true;
    });
  }

  Future<void> clear(String uid) async {
    await _mutate((store) => store.remove(uid));
  }

  Future<Map<String, Map<String, Map<String, dynamic>>>> _read() async {
    try {
      final file = await _file();
      if (!await file.exists()) return {};
      final decoded = jsonDecode(await file.readAsString());
      if (decoded is! Map) return {};
      return decoded.map((uid, sessions) {
        final sessionMap = sessions is Map ? sessions : const {};
        return MapEntry(
          uid as String,
          sessionMap.map((sessionId, value) {
            final fields = value is Map ? value : const {};
            return MapEntry(
              sessionId as String,
              Map<String, dynamic>.from(fields),
            );
          }),
        );
      });
    } catch (e) {
      debugPrint('privateboost joined sessions: failed to read store: $e');
      return {};
    }
  }

  Future<void> _mutate(
    void Function(Map<String, Map<String, Map<String, dynamic>>> store) update,
  ) async {
    final dir = await _dir();
    final lock = await _StoreLock.acquire(dir);
    if (lock == null) {
      debugPrint('privateboost joined sessions: lock busy, dropping a write');
      return;
    }
    try {
      final store = await _read();
      update(store);
      await _writeAtomically(await _file(), jsonEncode(store));
    } catch (e) {
      debugPrint('privateboost joined sessions: failed to write store: $e');
    } finally {
      await lock.release();
    }
  }
}

/// Makes [_writeAtomically]'s temp file name unique per call, so overlapping
/// writes cannot truncate the same temp path out from under each other.
int _writeCounter = 0;

/// Writes via a sibling temp file renamed into place, so a process killed
/// mid-write leaves either the old or the new [target], never a half-written
/// one that [JoinedSessions._read]'s corrupt-file guard would read as empty.
Future<void> _writeAtomically(File target, String content) async {
  final tmp = File('${target.path}.${pid}_${_writeCounter++}.tmp');
  await tmp.writeAsString(content, flush: true);
  await tmp.rename(target.path);
}
