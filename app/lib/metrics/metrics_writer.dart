import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:privateboost_app/metrics/context.dart';
import 'package:privateboost_app/metrics/round_metric.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart'
    show RoundStepResult, RoundSummary;
import 'package:privateboost_app/wake_loop.dart' show SessionRoundTimestamps;

/// Firestore collection holding one document per submitted training round.
const roundMetricsCollection = 'paperSimRoundMetrics';

/// Writes one round record to [roundMetricsCollection]. Telemetry must never
/// break training, so any failure is logged and swallowed. The backend-ack
/// wait is bounded by [timeout]: `add()`'s future completes on backend ack,
/// not on the local cache write, so an unbounded wait parks the training loop
/// while offline; a timed-out write is still queued by the offline cache and
/// syncs later.
Future<void> recordRound(
  RoundMetric metric, {
  FirebaseFirestore? firestore,
  Duration timeout = const Duration(seconds: 10),
}) async {
  try {
    final db = firestore ?? FirebaseFirestore.instance;
    await db
        .collection(roundMetricsCollection)
        .add(metric.toFirestore())
        .timeout(timeout);
  } catch (e) {
    debugPrint('privateboost metrics: failed to record round: $e');
  }
}

/// Records one `paperSimRoundMetrics` document for a step `runWake` took, from
/// [result]'s summary (a `Submitted` step) or its idle poll (`NothingNew`).
/// `step_session` never pushes onto `createRoundSummaryStream`, so this is the
/// wake loop's only path to a document.
///
/// A `NothingNew` step from the foreground poller is skipped, like the terminal
/// outcomes: that poller ticks at an operator-chosen fixed cadence, so its idle
/// polls say nothing about organic check-in behavior and would firehose the
/// collection. Any other outcome carrying neither a summary nor an idle poll
/// has nothing to record.
///
/// [datasetId] is what the wake loop resolved and validated for this session.
/// `wallMs` is the gap between this step's `clientTsMs` and whatever
/// [timestamps] holds for the same (uid, sessionId): durable across wakes and
/// restarts, so it stays correct when the previous step landed in an earlier
/// wake, and per-session, so it cannot leak between two sessions one wake
/// steps. A background idle check-in reads and updates the same entry a
/// submitted round does, making its `wallMs` the check-in interval that
/// recording idle polls exists to measure. A session with no prior step records
/// `wallMs = 0` rather than a raw `clientTsMs`.
///
/// [clock] supplies `clientTsMs`. `metricsRecordingHook` is what restricts
/// overriding it to tests, so this parameter is plain plumbing and not itself
/// `@visibleForTesting`, which a caller in that seam's library would trip the
/// lint on. Never throws into the wake loop: [recordRound] bounds and swallows
/// a failing write.
Future<void> recordRoundStep({
  required MetricsContext context,
  required String uid,
  required String triggerSource,
  required int batchId,
  required int batchCount,
  required String datasetId,
  required RoundStepResult result,
  required SessionRoundTimestamps timestamps,
  int? wakeLatencyMs,
  FirebaseFirestore? firestore,
  int Function() clock = _systemClientTsMs,
}) async {
  final summary = result.summary;
  final idle = result.idle;
  if (summary == null && (idle == null || triggerSource == 'foreground')) {
    return;
  }
  final sessionId = summary?.sessionId ?? idle!.sessionId;
  final clientTsMs = clock();
  final lastClientTsMs = await timestamps.get(uid, sessionId);
  await timestamps.set(uid, sessionId, clientTsMs);
  await _recordDocument(
    context: context,
    uid: uid,
    triggerSource: triggerSource,
    batchId: batchId,
    batchCount: batchCount,
    datasetId: datasetId,
    sessionId: sessionId,
    summary: summary,
    idlePollUs: idle?.pollUs,
    wallMs: lastClientTsMs == null ? 0 : clientTsMs - lastClientTsMs,
    clientTsMs: clientTsMs,
    wakeLatencyMs: wakeLatencyMs,
    firestore: firestore,
  );
}

int _systemClientTsMs() => DateTime.now().millisecondsSinceEpoch;

Future<void> _recordDocument({
  required MetricsContext context,
  required String uid,
  required String triggerSource,
  required int batchId,
  required int batchCount,
  String? datasetId,
  // The aggregator's real session id, carried on every RoundSummary/IdlePoll,
  // so these documents join the aggregator's per-tree metrics.
  required String sessionId,
  RoundSummary? summary,
  BigInt? idlePollUs,
  required int wallMs,
  required int clientTsMs,
  required int? wakeLatencyMs,
  required FirebaseFirestore? firestore,
}) async {
  final net = await networkType();
  final battery = await batteryState();
  final level = await batteryLevel();
  await recordRound(
    RoundMetric(
      uid: uid,
      deviceId: context.deviceId,
      deviceModel: context.deviceModel,
      appVersion: context.appVersion,
      sessionId: sessionId,
      triggerSource: triggerSource,
      batchId: batchId,
      batchCount: batchCount,
      datasetId: datasetId,
      summary: summary,
      idlePollUs: idlePollUs,
      networkType: net,
      batteryState: battery,
      batteryLevel: level,
      wallMs: wallMs,
      clientTsMs: clientTsMs,
      rssBytes: currentRssBytes(),
      wakeLatencyMs: wakeLatencyMs,
    ),
    firestore: firestore,
  );
}
