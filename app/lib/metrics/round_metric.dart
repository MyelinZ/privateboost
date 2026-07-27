import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart' show RoundSummary;

/// One submitted round or one idle check-in as a `paperSimRoundMetrics`
/// document, told apart by the `outcome` field. A submitted round's
/// measurements come from the Rust-side [RoundSummary]; an idle check-in
/// carries only [idlePollUs] and none of the round-specific fields, nothing
/// having been computed or submitted. Exactly one of [summary] or [idlePollUs]
/// is supplied.
///
/// Everything else is supplied Dart-side for both kinds alike: identity, the
/// trigger, this device's batch and dataset, device and app context, network
/// and power conditions, `wallMs`, `rssBytes`, and a first-push round's
/// `wakeLatencyMs`. `clientTsMs` is the device clock at arrival, since `ts` is
/// stamped at Firestore sync and lags a queued offline write.
class RoundMetric {
  const RoundMetric({
    required this.uid,
    required this.deviceId,
    required this.deviceModel,
    required this.appVersion,
    required this.sessionId,
    required this.triggerSource,
    required this.batchId,
    required this.batchCount,
    this.datasetId,
    this.summary,
    this.idlePollUs,
    required this.networkType,
    required this.batteryState,
    this.batteryLevel,
    required this.wallMs,
    required this.clientTsMs,
    required this.rssBytes,
    this.wakeLatencyMs,
  });

  final String uid;
  final String deviceId;
  final String deviceModel;
  final String appVersion;
  final String sessionId;
  final String triggerSource;
  final int batchId;
  final int batchCount;

  /// The bundled dataset this round's batch came from, e.g. `heart_disease`.
  /// [RoundSummary] carries no dataset id of its own; the wake loop's
  /// `recordRoundStep` (`metrics/metrics_writer.dart`) always supplies one.
  final String? datasetId;

  /// The submitted round's measurements; null for an idle check-in document
  /// (see [idlePollUs]).
  final RoundSummary? summary;

  /// This poll's `PollSession` wall time in microseconds, for an idle
  /// check-in document (`outcome: "idle"`, no [summary]). Null for a
  /// submitted round, whose own poll time lives on [summary].
  final BigInt? idlePollUs;
  final String networkType;
  final String batteryState;

  /// Charge percent 0-100; null (and omitted from the document) when the
  /// level could not be read.
  final int? batteryLevel;
  final int wallMs;
  final int clientTsMs;
  final int rssBytes;
  final int? wakeLatencyMs;

  Map<String, Object?> toFirestore() {
    final s = summary;
    return {
      'ts': FieldValue.serverTimestamp(),
      'uid': uid,
      'deviceId': deviceId,
      'deviceModel': deviceModel,
      'appVersion': appVersion,
      'sessionId': sessionId,
      'triggerSource': triggerSource,
      'batchId': batchId,
      'batchCount': batchCount,
      if (datasetId != null) 'datasetId': datasetId,
      if (s != null) ...{
        'roundId': s.roundId.toInt(),
        'nRecords': s.nRecords,
        if (s.treeIdx != null) 'treeIdx': s.treeIdx,
        if (s.depth != null) 'depth': s.depth,
        'roundKind': s.roundKind.name,
        'outcome': s.outcome.name,
        if (s.lastError != null) 'lastError': s.lastError,
        'nPeersAttempted': s.nPeersAttempted,
        'nPeersAccepted': s.nPeersAccepted,
        'pollUs': s.pollUs.toInt(),
        'computeUs': s.computeUs.toInt(),
        'submitUs': s.submitUs.toInt(),
        'txBytes': s.txBytes.toInt(),
        'rxBytes': s.rxBytes.toInt(),
      } else ...{
        'outcome': 'idle',
        'pollUs': idlePollUs!.toInt(),
      },
      'networkType': networkType,
      'batteryState': batteryState,
      if (batteryLevel != null) 'batteryLevel': batteryLevel,
      'wallMs': wallMs,
      'clientTsMs': clientTsMs,
      'rssBytes': rssBytes,
      if (wakeLatencyMs != null) 'wakeLatencyMs': wakeLatencyMs,
    };
  }
}
