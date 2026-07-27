import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

const _lockFileName = 'participation.lock';

/// A best-effort mutex that lets at most one participation run at a time
/// across the two triggers that drive a round (the FCM `round_open` wake and
/// the WorkManager periodic task).
///
/// Each trigger runs in its own isolate, so an in-process flag cannot exclude
/// them; but both isolates share one OS process, which rules out
/// [RandomAccessFile.lock] too (Dart's file lock is a per-process `fcntl`
/// record lock, so two sibling isolates both acquire it and it excludes
/// nothing). Exclusion is therefore built on atomic exclusive file creation
/// ([File.create] with `exclusive: true`): the create succeeds for exactly
/// one caller and fails for every other, across both isolates and separate
/// processes.
///
/// Each held lock carries a unique token in the file. It closes the reclaim
/// race, since two contenders re-creating the same stale lock leave only the
/// last writer's token, and it lets [release] delete the file only while the
/// lock is still ours.
///
/// This stops overlapping participations enrolling the device twice. Two
/// caveats: it protects only callers routed through [tryAcquire], and an
/// isolate killed without calling [release] leaves the file behind, so a lock
/// older than [_staleAfter] is reclaimed as abandoned.
class ParticipationLock {
  ParticipationLock._(this._file, this._token);

  final File _file;
  final String _token;

  /// A held lock older than this is assumed abandoned and may be reclaimed.
  /// Comfortably longer than any real participation: the OS kills background
  /// wakes within about 10 minutes.
  static const Duration _staleAfter = Duration(minutes: 30);

  /// How long a reclaim waits before re-reading the lock it re-created. By then
  /// a contender racing the same stale lock has finished its own write, so the
  /// re-read sees the loser's token and one backs off. Reclaims only follow a
  /// crashed isolate, so this delay is off every hot path.
  static const Duration _reclaimSettle = Duration(milliseconds: 50);

  /// Acquires the lock, or returns null if another participation holds it.
  static Future<ParticipationLock?> tryAcquire() async {
    final dir = await getApplicationSupportDirectory();
    return tryAcquireAt(File('${dir.path}/$_lockFileName'));
  }

  @visibleForTesting
  static Future<ParticipationLock?> tryAcquireAt(
    File lockFile, {
    Duration staleAfter = _staleAfter,
    Duration reclaimSettle = _reclaimSettle,
  }) async {
    final token = _mintToken();
    try {
      await lockFile.create(exclusive: true);
      await lockFile.writeAsString(token, flush: true);
      return ParticipationLock._(lockFile, token);
    } on FileSystemException {
      // Held by another isolate. Reclaim only if it looks abandoned; a fresh
      // lock means a real participation is in flight, so back off.
      try {
        final age = DateTime.now().difference(await lockFile.lastModified());
        if (age < staleAfter) return null;
        return await _reclaim(lockFile, token, reclaimSettle);
      } on FileSystemException {
        return null;
      }
    }
  }

  /// Steals an abandoned lock without stealing one that turned fresh
  /// underneath. The delete and create are not atomic, so two contenders seeing
  /// the same stale lock can each re-create the file; stamping a token, waiting
  /// [settle] and re-reading leaves the last writer's token, so every earlier
  /// writer reads a foreign one and backs off, and at most one contender
  /// returns a held lock. A live acquirer that took the freed path first fails
  /// the exclusive create instead, so its lock is never stolen.
  static Future<ParticipationLock?> _reclaim(
    File lockFile,
    String token,
    Duration settle,
  ) async {
    try {
      await lockFile.delete();
      await lockFile.create(exclusive: true);
      await lockFile.writeAsString(token, flush: true);
    } on FileSystemException {
      // Another contender deleted the stale lock before us, or a fresh acquire
      // won the exclusive create: its live lock is not ours to take.
      return null;
    }
    await Future<void>.delayed(settle);
    try {
      if (await lockFile.readAsString() != token) return null;
    } on FileSystemException {
      return null;
    }
    return ParticipationLock._(lockFile, token);
  }

  static String _mintToken() {
    final rng = Random.secure();
    final bytes = List<int>.generate(16, (_) => rng.nextInt(256));
    return bytes.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
  }

  /// Releases the lock so the next trigger can proceed. Idempotent. Deletes the
  /// file only while it still carries our token, so a lock already reclaimed as
  /// stale (and re-created) by another isolate is left for its new owner.
  Future<void> release() async {
    try {
      if (await _file.readAsString() == _token) await _file.delete();
    } on FileSystemException {
      // Already gone (e.g. reclaimed as stale by another isolate).
    }
  }
}
