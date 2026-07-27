import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:privateboost_app/dataset.dart';
import 'package:privateboost_app/demo_config.dart';
import 'package:privateboost_app/joined_sessions.dart';
import 'package:privateboost_app/pilot_ca.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart' as rust;
import 'package:privateboost_app/src/rust/api/mobile.dart'
    show EnrollResult, RoundStepOutcome, RoundStepResult, SessionListResult;

/// One round-step wake, shared by the FCM `round_open` push and the
/// WorkManager periodic task: list the hosted sessions, advance every session
/// this device has joined and not finished by exactly one round, and persist
/// enough state to resume next time.
///
/// A round can outlast the screen, so this never loops waiting for a session
/// to finish: each trigger fires it once and returns, and the aggregator holds
/// the round open until the next wake.
///
/// Every session status acted on comes from this wake's own [WakeBridge]
/// calls, never from whatever woke the isolate. The `round_open` payload
/// carries no session id, so it can only mean "go check".
///
/// Callers must hold [ParticipationLock] for the whole call, so no sibling
/// trigger enrolls this device twice, and must pass a fresh `uid` and
/// `idToken`. Keeping both outside this function is what makes it testable
/// with no network and no platform channel.
///
/// [onRoundStep] runs after every step, whatever its outcome, and does nothing
/// by default; `runBackgroundWake` passes the real hook, so this loop stays
/// callable with no Firestore in a test.
Future<WakeResult> runWake({
  required String trigger,
  required String uid,
  required String idToken,
  required String aggEndpoint,
  JoinedSessions? joinedSessions,
  SessionWatermarks? watermarks,
  WakeBridge? bridge,
  int? wakeLatencyMs,
  WakeStepHook onRoundStep = _noopStepHook,
}) async {
  final sessions = joinedSessions ?? JoinedSessions();
  final marks = watermarks ?? SessionWatermarks();
  final wakeBridge = bridge ?? const RustWakeBridge();

  final active = (await sessions.active(uid)).toList()..sort();
  if (active.isEmpty) return const WakeResult(stepped: []);

  final listed = await wakeBridge.listSessions(aggEndpoint: aggEndpoint, idToken: idToken);
  if (listed.lastError != null) {
    return WakeResult(stepped: const [], lastError: listed.lastError);
  }
  final hosted = {for (final s in listed.sessions) s.sessionId: s};

  final stepped = <SessionStepOutcome>[];
  for (final sessionId in active) {
    final summary = hosted[sessionId];
    if (summary == null) continue; // not (or no longer) hosted; nothing to do this wake

    final datasetId = summary.datasetId;
    if (datasetId.isEmpty || !_bundledDatasetIds.contains(datasetId)) {
      debugPrint(
        'privateboost wake: session $sessionId names dataset "$datasetId", which this '
        'device does not bundle; skipping',
      );
      continue;
    }
    final bundledFeatures = await datasetFeatureCount(datasetId);
    if (bundledFeatures != summary.nFeatures) {
      debugPrint(
        'privateboost wake: session $sessionId names dataset $datasetId '
        '($bundledFeatures features) but reports ${summary.nFeatures}; skipping',
      );
      continue;
    }

    final rows = batchSlice(await loadTrainRows(datasetId), batchId, batchCount).rows;
    final lastSeen = await marks.get(uid, sessionId);
    final result = await wakeBridge.stepSession(
      aggEndpoint: aggEndpoint,
      idToken: idToken,
      sessionId: sessionId,
      records: rows,
      hidePath: true,
      lastSeenRoundId: lastSeen,
    );

    // `stepSession` returns the input watermark unchanged on every outcome but
    // `Submitted`, so a step this device did not finish never advances what it
    // resumes from. Gating on the outcome here too means a bridge that
    // misreports cannot defeat that. Persisted before `onRoundStep` runs, so a
    // throwing hook cannot suppress the persist and make the next wake
    // re-submit this round.
    if (result.outcome == RoundStepOutcome.submitted) {
      await marks.set(uid, sessionId, result.lastSeenRoundId);
    }

    await onRoundStep(
      sessionId,
      trigger,
      result,
      datasetId,
      stepped.isEmpty ? wakeLatencyMs : null,
    );

    if (result.outcome == RoundStepOutcome.completed ||
        result.outcome == RoundStepOutcome.failed) {
      await sessions.markFinished(uid, sessionId);
    }

    stepped.add(SessionStepOutcome(sessionId: sessionId, result: result));
  }
  return WakeResult(stepped: stepped);
}

/// Dataset ids `dataset.dart` bundles a train split for. Duplicated because
/// that map is private, so this loop can tell whether a session names a dataset
/// this device has records for without loading every split to ask.
const _bundledDatasetIds = ['heart_disease', 'breast_cancer', 'pima_diabetes', 'cdc_diabetes'];

/// An interface so a test can supply a fake with no network and no platform
/// channel.
abstract class WakeBridge {
  Future<SessionListResult> listSessions({
    required String aggEndpoint,
    required String idToken,
  });

  /// Records this device's interest so the notify tick starts waking it.
  /// Best-effort by contract: a failure means "the next full enroll repairs
  /// it", never a failed join.
  Future<EnrollResult> enrollSession({
    required String aggEndpoint,
    required String idToken,
    required String sessionId,
  });

  Future<RoundStepResult> stepSession({
    required String aggEndpoint,
    required String idToken,
    required String sessionId,
    required List<TrainRow> records,
    required bool hidePath,
    required BigInt lastSeenRoundId,
  });
}

/// Pins the bundled CA for an `https` endpoint as `Facade` does, then calls the
/// generated bridge functions.
class RustWakeBridge implements WakeBridge {
  const RustWakeBridge();

  @override
  Future<SessionListResult> listSessions({
    required String aggEndpoint,
    required String idToken,
  }) async {
    final Uint8List? caPem;
    try {
      caPem = await resolvePinnedCa(aggEndpoint);
    } catch (e) {
      return SessionListResult(sessions: const [], lastError: '$e');
    }
    return rust.listSessions(aggEndpoint: aggEndpoint, idToken: idToken, caPem: caPem);
  }

  @override
  Future<EnrollResult> enrollSession({
    required String aggEndpoint,
    required String idToken,
    required String sessionId,
  }) async {
    final Uint8List? caPem;
    try {
      caPem = await resolvePinnedCa(aggEndpoint);
    } catch (e) {
      return EnrollResult(ok: false, lastError: '$e');
    }
    return rust.enrollSession(
      aggEndpoint: aggEndpoint,
      idToken: idToken,
      sessionId: sessionId,
      caPem: caPem,
    );
  }

  @override
  Future<RoundStepResult> stepSession({
    required String aggEndpoint,
    required String idToken,
    required String sessionId,
    required List<TrainRow> records,
    required bool hidePath,
    required BigInt lastSeenRoundId,
  }) async {
    final Uint8List? caPem;
    try {
      caPem = await resolvePinnedCa(aggEndpoint);
    } catch (e) {
      return RoundStepResult(
        outcome: RoundStepOutcome.error,
        roundId: BigInt.zero,
        sessionId: sessionId,
        lastSeenRoundId: lastSeenRoundId,
        lastError: '$e',
      );
    }
    return rust.stepSession(
      aggEndpoint: aggEndpoint,
      idToken: idToken,
      sessionId: sessionId,
      records: records.map((r) => (Float64List.fromList(r.$1), r.$2)).toList(),
      hidePath: hidePath,
      caPem: caPem,
      lastSeenRoundId: lastSeenRoundId,
    );
  }
}

/// Called after every step, successful or not. [datasetId] is the dataset this
/// wake resolved and validated for the session, the same id `records` was built
/// from. [wakeLatencyMs] is non-null only for the first session stepped, since
/// it measures the wake, not the round.
typedef WakeStepHook = Future<void> Function(
  String sessionId,
  String triggerSource,
  RoundStepResult result,
  String datasetId,
  int? wakeLatencyMs,
);

Future<void> _noopStepHook(
  String sessionId,
  String triggerSource,
  RoundStepResult result,
  String datasetId,
  int? wakeLatencyMs,
) async {}

/// [lastError] is set only when the wake could not list sessions at all; a
/// per-session failure lives on that session's [SessionStepOutcome.result].
class WakeResult {
  const WakeResult({required this.stepped, this.lastError});

  final List<SessionStepOutcome> stepped;
  final String? lastError;
}

/// One joined session's outcome for this wake.
class SessionStepOutcome {
  const SessionStepOutcome({required this.sessionId, required this.result});

  final String sessionId;
  final RoundStepResult result;
}

const _watermarkFileName = 'session_watermarks.json';

/// Durable last-seen round id per (uid, sessionId), passed to the next wake's
/// `stepSession` so a resumed session enrolls where this device left off.
///
/// Keyed by session id as well as uid because a watermark means nothing outside
/// the session that issued it: pairing a round number with a session id it no
/// longer matches, after a restart started a new session, is the hazard
/// `RoundStepResult.sessionId` describes.
///
/// Unlike `JoinedSessions` this keeps no lock file: every write happens inside
/// `runWake`, whose callers hold `ParticipationLock` for the whole wake, so no
/// two isolates touch this file at once.
class SessionWatermarks {
  SessionWatermarks({@visibleForTesting this._directory});

  final Directory? _directory;

  Future<Directory> _dir() async => _directory ?? await getApplicationSupportDirectory();

  Future<File> _file() async => File('${(await _dir()).path}/$_watermarkFileName');

  /// The last round id recorded, or zero if none: `enroll_at` then resumes
  /// from the start of the session.
  Future<BigInt> get(String uid, String sessionId) async {
    final store = await _read();
    final raw = store[uid]?[sessionId];
    if (raw == null) return BigInt.zero;
    try {
      return BigInt.parse(raw);
    } catch (e) {
      debugPrint('privateboost wake loop: corrupt watermark for $sessionId: $e');
      return BigInt.zero;
    }
  }

  Future<void> set(String uid, String sessionId, BigInt roundId) async {
    final store = await _read();
    final forUid = store.putIfAbsent(uid, () => {});
    forUid[sessionId] = roundId.toString();
    try {
      await _writeAtomically(await _file(), jsonEncode(store));
    } catch (e) {
      debugPrint('privateboost wake loop: failed to write watermark store: $e');
    }
  }

  Future<Map<String, Map<String, String>>> _read() async {
    try {
      final file = await _file();
      if (!await file.exists()) return {};
      final decoded = jsonDecode(await file.readAsString());
      if (decoded is! Map) return {};
      return decoded.map((uid, forUid) {
        final byUid = forUid is Map ? forUid : const {};
        return MapEntry(
          uid as String,
          byUid.map((sessionId, roundId) => MapEntry(sessionId as String, roundId as String)),
        );
      });
    } catch (e) {
      debugPrint('privateboost wake loop: failed to read watermark store: $e');
      return {};
    }
  }
}

const _roundTimestampsFileName = 'session_round_timestamps.json';

/// Durable last-recorded-round `clientTsMs` per (uid, sessionId), so a
/// wake-driven round's `wallMs` is the true gap since that session's previous
/// round even across wakes hours apart or an app restart, which no in-memory
/// stopwatch spans. Parallel to [SessionWatermarks] rather than folded into it:
/// this is telemetry-only while that gates what `stepSession` resumes from, and
/// separate files mean a bug in one cannot corrupt the other. Same on-disk
/// shape and locking argument as [SessionWatermarks].
///
/// [get] returns `null` for a (uid, sessionId) with no recorded round yet
/// (or a corrupt stored value), unlike [SessionWatermarks.get], it does not
/// fold "unset" into `0`, because `0` is itself a value `clientTsMs` could
/// legitimately hold, and `recordRoundStep` (`metrics/metrics_writer.dart`)
/// needs to tell "no prior round" apart from "a prior round at timestamp 0"
/// to record a session's first round as `wallMs = 0` instead of the raw
/// `clientTsMs`.
class SessionRoundTimestamps {
  SessionRoundTimestamps({@visibleForTesting this._directory});

  final Directory? _directory;

  Future<Directory> _dir() async => _directory ?? await getApplicationSupportDirectory();

  Future<File> _file() async => File('${(await _dir()).path}/$_roundTimestampsFileName');

  Future<int?> get(String uid, String sessionId) async {
    final store = await _read();
    final raw = store[uid]?[sessionId];
    if (raw == null) return null;
    return int.tryParse(raw);
  }

  Future<void> set(String uid, String sessionId, int clientTsMs) async {
    final store = await _read();
    final forUid = store.putIfAbsent(uid, () => {});
    forUid[sessionId] = clientTsMs.toString();
    try {
      await _writeAtomically(await _file(), jsonEncode(store));
    } catch (e) {
      debugPrint('privateboost wake loop: failed to write round timestamp store: $e');
    }
  }

  Future<Map<String, Map<String, String>>> _read() async {
    try {
      final file = await _file();
      if (!await file.exists()) return {};
      final decoded = jsonDecode(await file.readAsString());
      if (decoded is! Map) return {};
      return decoded.map((uid, forUid) {
        final byUid = forUid is Map ? forUid : const {};
        return MapEntry(
          uid as String,
          byUid.map((sessionId, ts) => MapEntry(sessionId as String, ts as String)),
        );
      });
    } catch (e) {
      debugPrint('privateboost wake loop: failed to read round timestamp store: $e');
      return {};
    }
  }
}

/// Per-isolate counter making [_writeAtomically]'s temp file name unique per
/// call, so two overlapping writes never truncate the same temp path out from
/// under each other.
int _writeCounter = 0;

/// Writes [content] to [target] via a sibling temp file that is renamed into
/// place, so a process killed mid-write leaves either the old or the new
/// [target] and never a half-written file that [SessionWatermarks._read]'s
/// corrupt-file guard would silently read back as an empty store.
Future<void> _writeAtomically(File target, String content) async {
  final tmp = File('${target.path}.${pid}_${_writeCounter++}.tmp');
  await tmp.writeAsString(content, flush: true);
  await tmp.rename(target.path);
}
