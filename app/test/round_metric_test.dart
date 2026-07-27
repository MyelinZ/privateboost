import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:fake_cloud_firestore/fake_cloud_firestore.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/metrics/metrics_writer.dart';
import 'package:privateboost_app/metrics/round_metric.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart';

RoundSummary _summary({
  BigInt? roundId,
  RoundKind kind = RoundKind.gradient,
  RoundOutcome outcome = RoundOutcome.submitted,
  int? treeIdx = 0,
  int? depth = 4,
  String? lastError,
  int nRecords = 1,
}) =>
    RoundSummary(
      roundId: roundId ?? BigInt.from(5),
      sessionId: 'sess-1',
      nRecords: nRecords,
      roundKind: kind,
      treeIdx: treeIdx,
      depth: depth,
      pollUs: BigInt.from(100),
      computeUs: BigInt.from(40),
      submitUs: BigInt.from(60),
      txBytes: BigInt.from(200000),
      rxBytes: BigInt.from(84000),
      nPeersAttempted: 3,
      nPeersAccepted: 2,
      outcome: outcome,
      lastError: lastError,
    );

RoundMetric _metric(
  RoundSummary summary, {
  String triggerSource = 'workmanager',
  int? wakeLatencyMs,
  String? datasetId,
  int? batteryLevel,
}) =>
    RoundMetric(
      uid: 'u1',
      deviceId: 'd1',
      deviceModel: 'Google Pixel 9 Pro',
      appVersion: '1.0.0+1',
      sessionId: 'abcd',
      triggerSource: triggerSource,
      summary: summary,
      batchId: 2,
      batchCount: 3,
      datasetId: datasetId,
      networkType: 'cellular',
      batteryState: 'charging',
      batteryLevel: batteryLevel,
      wallMs: 220,
      clientTsMs: 1752000000000,
      rssBytes: 123456,
      wakeLatencyMs: wakeLatencyMs,
    );

RoundMetric _idleMetric({
  BigInt? pollUs,
  String triggerSource = 'workmanager',
  String? datasetId,
}) =>
    RoundMetric(
      uid: 'u1',
      deviceId: 'd1',
      deviceModel: 'Google Pixel 9 Pro',
      appVersion: '1.0.0+1',
      sessionId: 'abcd',
      triggerSource: triggerSource,
      idlePollUs: pollUs ?? BigInt.from(15000),
      batchId: 2,
      batchCount: 3,
      datasetId: datasetId,
      networkType: 'cellular',
      batteryState: 'charging',
      wallMs: 220,
      clientTsMs: 1752000000000,
      rssBytes: 123456,
    );

Future<Map<String, dynamic>> _write(RoundMetric metric) async {
  final fake = FakeFirebaseFirestore();
  await recordRound(metric, firestore: fake);
  final docs = await fake.collection(roundMetricsCollection).get();
  expect(docs.docs.length, 1);
  return docs.docs.first.data();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('toFirestore has every camelCase key with the expected values', () async {
    final d = await _write(
      _metric(_summary(),
          triggerSource: 'push', datasetId: 'heart_disease', batteryLevel: 87),
    );
    expect(d['uid'], 'u1');
    expect(d['deviceId'], 'd1');
    expect(d['deviceModel'], 'Google Pixel 9 Pro');
    expect(d['appVersion'], '1.0.0+1');
    expect(d['sessionId'], 'abcd');
    expect(d['triggerSource'], 'push');
    expect(d['datasetId'], 'heart_disease');
    expect(d['roundId'], 5);
    expect(d['nRecords'], 1);
    expect(d['treeIdx'], 0);
    expect(d['depth'], 4);
    expect(d['roundKind'], 'gradient');
    expect(d['outcome'], 'submitted');
    expect(d['nPeersAttempted'], 3);
    expect(d['nPeersAccepted'], 2);
    expect(d['batchId'], 2);
    expect(d['batchCount'], 3);
    expect(d['networkType'], 'cellular');
    expect(d['batteryState'], 'charging');
    expect(d['batteryLevel'], 87);
    expect(d['wallMs'], 220);
    expect(d['clientTsMs'], 1752000000000);
    expect(d['pollUs'], 100);
    expect(d['computeUs'], 40);
    expect(d['submitUs'], 60);
    expect(d['txBytes'], 200000);
    expect(d['rxBytes'], 84000);
    expect(d['rssBytes'], 123456);
    expect(d.containsKey('ts'), isTrue);
    expect(d.containsKey('recordIndex'), isFalse,
        reason: 'a device is one client now; there is no per-record index');
  });

  test('roundKind and outcome are the enum .name strings', () {
    final stats = _metric(_summary(kind: RoundKind.stats)).toFirestore();
    expect(stats['roundKind'], 'stats');
    final below =
        _metric(_summary(outcome: RoundOutcome.belowThreshold)).toFirestore();
    expect(below['outcome'], 'belowThreshold');
  });

  test('nullable fields are omitted when null', () {
    final d = _metric(_summary(treeIdx: null, depth: null)).toFirestore();
    expect(d.containsKey('treeIdx'), isFalse);
    expect(d.containsKey('depth'), isFalse);
    expect(d.containsKey('lastError'), isFalse);
    expect(d.containsKey('wakeLatencyMs'), isFalse);
    expect(d.containsKey('batteryLevel'), isFalse);
  });

  test('nullable fields are present when set', () {
    final d = _metric(
      _summary(treeIdx: 3, depth: 2, lastError: 'boom'),
      wakeLatencyMs: 250,
      batteryLevel: 42,
    ).toFirestore();
    expect(d['treeIdx'], 3);
    expect(d['depth'], 2);
    expect(d['lastError'], 'boom');
    expect(d['wakeLatencyMs'], 250);
    expect(d['batteryLevel'], 42);
  });

  test('nRecords carries the summary value', () {
    final d = _metric(_summary(nRecords: 3)).toFirestore();
    expect(d['nRecords'], 3);
  });

  test('datasetId is omitted when null', () {
    final d = _metric(_summary()).toFirestore();
    expect(d.containsKey('datasetId'), isFalse);
  });

  test('datasetId is present when set', () {
    final d = _metric(_summary(), datasetId: 'breast_cancer').toFirestore();
    expect(d['datasetId'], 'breast_cancer');
  });

  test('ts is a FieldValue.serverTimestamp sentinel', () {
    final d = _metric(_summary()).toFirestore();
    expect(d['ts'], isA<FieldValue>());
  });

  group('idle check-in document (no summary)', () {
    test('has outcome idle, pollUs, and no round-specific keys',
        () async {
      final d = await _write(
        _idleMetric(pollUs: BigInt.from(9876), triggerSource: 'push', datasetId: 'heart_disease'),
      );
      expect(d['uid'], 'u1');
      expect(d['sessionId'], 'abcd');
      expect(d['triggerSource'], 'push');
      expect(d['datasetId'], 'heart_disease');
      expect(d['batchId'], 2);
      expect(d['batchCount'], 3);
      expect(d['outcome'], 'idle');
      expect(d['pollUs'], 9876);
      expect(d['networkType'], 'cellular');
      expect(d['batteryState'], 'charging');
      expect(d['wallMs'], 220);
      expect(d['clientTsMs'], 1752000000000);
      expect(d['rssBytes'], 123456);
      for (final key in [
        'roundId',
        'nRecords',
        'treeIdx',
        'depth',
        'roundKind',
        'nPeersAttempted',
        'nPeersAccepted',
        'computeUs',
        'submitUs',
        'txBytes',
        'rxBytes',
      ]) {
        expect(d.containsKey(key), isFalse, reason: '$key is round-specific; an idle poll has none');
      }
    });
  });
}
