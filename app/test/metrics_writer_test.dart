import 'dart:async';
import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:fake_cloud_firestore/fake_cloud_firestore.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/metrics/context.dart';
import 'package:privateboost_app/metrics/round_metric.dart';
import 'package:privateboost_app/metrics/metrics_writer.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart';
import 'package:privateboost_app/wake_loop.dart' show SessionRoundTimestamps;

RoundMetric _metric() => RoundMetric(
      uid: 'u1',
      deviceId: 'd1',
      deviceModel: 'model',
      appVersion: '1.0.0+1',
      sessionId: 'abcd',
      triggerSource: 'push',
      summary: RoundSummary(
        roundId: BigInt.from(5),
        sessionId: 'sess-1',
        nRecords: 1,
        roundKind: RoundKind.gradient,
        treeIdx: 0,
        depth: 4,
        pollUs: BigInt.from(1),
        computeUs: BigInt.from(1),
        submitUs: BigInt.from(1),
        txBytes: BigInt.from(1),
        rxBytes: BigInt.from(1),
        nPeersAttempted: 3,
        nPeersAccepted: 3,
        outcome: RoundOutcome.submitted,
        lastError: null,
      ),
      batchId: 0,
      batchCount: 1,
      networkType: 'wifi',
      batteryState: 'discharging',
      batteryLevel: 55,
      wallMs: 10,
      clientTsMs: 1752000000000,
      rssBytes: 1,
    );

/// Firestore whose `add` rejects, standing in for a backend/permission
/// failure. recordRound must swallow it rather than throw into the round loop.
class _ThrowingFirestore extends Fake implements FirebaseFirestore {
  @override
  CollectionReference<Map<String, dynamic>> collection(String collectionPath) =>
      _ThrowingCollection();
}

// ignore: subtype_of_sealed_class
class _ThrowingCollection extends Fake
    implements CollectionReference<Map<String, dynamic>> {
  @override
  Future<DocumentReference<Map<String, dynamic>>> add(Object? data) =>
      Future.error(StateError('firestore is down'));
}

/// Firestore whose `add` never completes, standing in for an offline device
/// where the backend ack never arrives. recordRound must give up on its bound.
class _HangingFirestore extends Fake implements FirebaseFirestore {
  @override
  CollectionReference<Map<String, dynamic>> collection(String collectionPath) =>
      _HangingCollection();
}

// ignore: subtype_of_sealed_class
class _HangingCollection extends Fake
    implements CollectionReference<Map<String, dynamic>> {
  @override
  Future<DocumentReference<Map<String, dynamic>>> add(Object? data) =>
      Completer<DocumentReference<Map<String, dynamic>>>().future;
}

/// A `step_session` outcome carrying a summary, as only a `Submitted` step
/// does (see `app/rust/src/api/mobile.rs`'s `round_summary_from`).
RoundStepResult _submittedStep(String sessionId, BigInt roundId) => RoundStepResult(
      outcome: RoundStepOutcome.submitted,
      roundId: roundId,
      sessionId: sessionId,
      lastSeenRoundId: roundId,
      summary: RoundSummary(
        roundId: roundId,
        sessionId: sessionId,
        nRecords: 7,
        roundKind: RoundKind.gradient,
        treeIdx: 0,
        depth: 1,
        pollUs: BigInt.from(5),
        computeUs: BigInt.from(10),
        submitUs: BigInt.from(8),
        txBytes: BigInt.from(900),
        rxBytes: BigInt.from(1500),
        nPeersAttempted: 3,
        nPeersAccepted: 3,
        outcome: RoundOutcome.submitted,
        lastError: null,
      ),
    );

/// A `step_session` outcome with nothing new to report, `summary` is null,
/// as it is for every non-`Submitted` outcome; `idle` carries the poll's
/// session id and wall time.
RoundStepResult _nothingNewStep(String sessionId, {BigInt? pollUs}) => RoundStepResult(
      outcome: RoundStepOutcome.nothingNew,
      roundId: BigInt.zero,
      sessionId: sessionId,
      lastSeenRoundId: BigInt.zero,
      idle: IdlePoll(sessionId: sessionId, pollUs: pollUs ?? BigInt.from(15000)),
    );

/// A step with neither a summary nor an idle poll, as `Completed`, `Failed`,
/// and `Error` outcomes report.
RoundStepResult _terminalStep(String sessionId) => RoundStepResult(
      outcome: RoundStepOutcome.completed,
      roundId: BigInt.zero,
      sessionId: sessionId,
      lastSeenRoundId: BigInt.zero,
    );

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // `FieldValue`'s platform factory is a `static final`, cached on first
  // touch of the class. Constructing a `FakeFirebaseFirestore` here installs
  // the mock factory before any test calls `RoundMetric.toFirestore()` (which
  // reaches `FieldValue.serverTimestamp()`), so a later test's real
  // `FakeFirebaseFirestore` write does not fail casting a factory-created
  // `FieldValue` produced back when no fake was constructed yet.
  setUpAll(() => FakeFirebaseFirestore());

  test('recordRound swallows a throwing firestore', () async {
    var threw = false;
    try {
      await recordRound(_metric(), firestore: _ThrowingFirestore());
    } catch (_) {
      threw = true;
    }
    expect(threw, isFalse);
  });

  test('recordRound honors a bounded timeout on a never-acking write', () async {
    final sw = Stopwatch()..start();
    await recordRound(
      _metric(),
      firestore: _HangingFirestore(),
      timeout: const Duration(milliseconds: 50),
    );
    sw.stop();
    // Returns on the bound, not after hanging on the never-completing add.
    expect(sw.elapsed, lessThan(const Duration(seconds: 5)));
  });

  group('recordRoundStep', () {
    const context = MetricsContext(
      deviceId: 'd1',
      deviceModel: 'model',
      appVersion: '1.0.0+1',
    );

    late Directory tmp;
    late SessionRoundTimestamps timestamps;

    setUp(() async {
      tmp = await Directory.systemTemp.createTemp('pbr_metrics_writer_test');
      timestamps = SessionRoundTimestamps(directory: tmp);
    });

    tearDown(() async {
      if (await tmp.exists()) await tmp.delete(recursive: true);
    });

    /// Closes the gap this task exists to fix: a `runWake`-stepped round with
    /// a summary must produce exactly one `paperSimRoundMetrics` document --
    /// `step_session` never pushes onto `createRoundSummaryStream`, so
    /// `recordRoundStep` is the only path to a document for a wake-driven
    /// round.
    test('a submitted step writes exactly one document, with datasetId and '
        'nRecords and no recordIndex', () async {
      final fake = FakeFirebaseFirestore();

      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'push',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _submittedStep('sess-wake', BigInt.from(3)),
        timestamps: timestamps,
        firestore: fake,
      );

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs.length, 1);
      final d = docs.single.data();
      expect(d['sessionId'], 'sess-wake');
      expect(d['triggerSource'], 'push');
      expect(d['datasetId'], 'heart_disease');
      expect(d['nRecords'], 7);
      expect(d.containsKey('recordIndex'), isFalse);
      // The test host has no battery plugin, so the guarded reads degrade:
      // the state sentinel is recorded, the unreadable level is omitted.
      expect(d['batteryState'], 'unknown');
      expect(d.containsKey('batteryLevel'), isFalse);
    });

    /// A background wake (push, workmanager) whose poll found nothing new
    /// still writes a check-in document: this is the gap this task exists to
    /// close, so idle check-in frequency is measurable too.
    test('a background idle poll writes one document with outcome '
        'idle, pollUs, and no compute/submit keys', () async {
      final fake = FakeFirebaseFirestore();

      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'workmanager',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _nothingNewStep('sess-wake', pollUs: BigInt.from(12345)),
        timestamps: timestamps,
        firestore: fake,
      );

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs.length, 1);
      final d = docs.single.data();
      expect(d['sessionId'], 'sess-wake');
      expect(d['triggerSource'], 'workmanager');
      expect(d['outcome'], 'idle');
      expect(d['pollUs'], 12345);
      expect(d.containsKey('computeUs'), isFalse);
      expect(d.containsKey('submitUs'), isFalse);
      expect(d.containsKey('roundId'), isFalse);
    });

    /// The foreground poller ticks at a fixed operator-chosen cadence, so its
    /// idle polls carry no organic check-in signal and must not be recorded.
    test('a foreground idle poll writes nothing', () async {
      final fake = FakeFirebaseFirestore();

      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'foreground',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _nothingNewStep('sess-wake'),
        timestamps: timestamps,
        firestore: fake,
      );

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs, isEmpty);
    });

    test('a step with neither a summary nor an idle poll (Completed, Failed, Error) '
        'writes nothing', () async {
      final fake = FakeFirebaseFirestore();

      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'workmanager',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _terminalStep('sess-wake'),
        timestamps: timestamps,
        firestore: fake,
      );

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs, isEmpty);
    });

    test('never throws even when the write fails', () async {
      var threw = false;
      try {
        await recordRoundStep(
          context: context,
          uid: 'u1',
          triggerSource: 'push',
          batchId: 0,
          batchCount: 1,
          datasetId: 'heart_disease',
          result: _submittedStep('sess-wake', BigInt.one),
          timestamps: timestamps,
          firestore: _ThrowingFirestore(),
        );
      } catch (_) {
        threw = true;
      }
      expect(threw, isFalse);
    });

    test(
        "the first recorded round of a session has wallMs = 0, not the raw "
        'epoch-ms clientTsMs', () async {
      final fake = FakeFirebaseFirestore();
      const epochClientTsMs = 1752000000000; // realistic epoch-ms clock reading

      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'push',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _submittedStep('sess-first', BigInt.one),
        timestamps: timestamps,
        firestore: fake,
        clock: () => epochClientTsMs,
      );

      final d = (await fake.collection(roundMetricsCollection).get()).docs.single.data();
      expect(d['clientTsMs'], epochClientTsMs);
      expect(
        d['wallMs'],
        0,
        reason: 'no prior round for this session: there is no measurable gap, so wallMs is '
            'recorded as 0',
      );
      expect(
        d['wallMs'],
        isNot(greaterThan(1000)),
        reason: 'must not be the raw epoch-ms clientTsMs (~1.75e12) that a missing-entry '
            'default of 0 for lastClientTsMs would produce',
      );
    });

    test(
        "a session's second wake-driven round has wallMs equal to the gap between the "
        'two rounds\' clientTsMs, not the intra-wake time', () async {
      final fake = FakeFirebaseFirestore();

      // Two separate `recordRoundStep` calls stand in for two separate wakes
      //, in reality hours apart. The clock is pinned per call, not slept,
      // so the asserted gap is exact and the test has no real wall-clock wait.
      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'push',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _submittedStep('sess-A', BigInt.one),
        timestamps: timestamps,
        firestore: fake,
        clock: () => 1_000_000,
      );
      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'push',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _submittedStep('sess-A', BigInt.two),
        timestamps: timestamps,
        firestore: fake,
        clock: () => 4_600_000, // one hour (3_600_000ms) later, a different wake entirely
      );

      final docs = (await fake.collection(roundMetricsCollection).get()).docs
        ..sort((a, b) => (a.data()['clientTsMs'] as int).compareTo(b.data()['clientTsMs'] as int));
      expect(docs.length, 2);
      expect(docs[1].data()['wallMs'], 3_600_000);
    });

    test("two sessions stepped in one wake do not contaminate each other's wallMs",
        () async {
      final fake = FakeFirebaseFirestore();

      // sess-A already has a recorded round; sess-B has none. Stepping both
      // from the same `timestamps` store, standing in for one wake, must
      // leave each session's wallMs keyed on its own history alone.
      await timestamps.set('u1', 'sess-A', 10_000);

      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'push',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _submittedStep('sess-A', BigInt.from(2)),
        timestamps: timestamps,
        firestore: fake,
        clock: () => 10_500,
      );
      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'push',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _submittedStep('sess-B', BigInt.one),
        timestamps: timestamps,
        firestore: fake,
        clock: () => 10_600,
      );

      final docs = {
        for (final d in (await fake.collection(roundMetricsCollection).get()).docs)
          d.data()['sessionId'] as String: d.data(),
      };
      expect(docs['sess-A']!['wallMs'], 500,
          reason: "10_500 - 10_000, sess-A's own prior round");
      expect(
        docs['sess-B']!['wallMs'],
        0,
        reason: 'sess-B has no prior round of its own, so there is no gap to measure: wallMs '
            "is 0, not a gap measured against sess-A's stored timestamp nor the raw "
            "epoch-ms clientTsMs",
      );
    });

    test(
        'a background idle poll both reads and updates the same per-session '
        'timestamp chain a submitted round does', () async {
      final fake = FakeFirebaseFirestore();

      // The submitted round establishes sess-A's last-check-in timestamp...
      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'push',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _submittedStep('sess-A', BigInt.one),
        timestamps: timestamps,
        firestore: fake,
        clock: () => 10_000,
      );
      // ...an idle check-in measures its wallMs against that same entry...
      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'workmanager',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _nothingNewStep('sess-A'),
        timestamps: timestamps,
        firestore: fake,
        clock: () => 10_300,
      );
      // ...and updates it in turn, so the next check-in (idle or submitted)
      // measures the interval since THIS idle poll, not the earlier round.
      await recordRoundStep(
        context: context,
        uid: 'u1',
        triggerSource: 'workmanager',
        batchId: 0,
        batchCount: 1,
        datasetId: 'heart_disease',
        result: _nothingNewStep('sess-A'),
        timestamps: timestamps,
        firestore: fake,
        clock: () => 10_900,
      );

      final docs = (await fake.collection(roundMetricsCollection).get()).docs
        ..sort((a, b) => (a.data()['clientTsMs'] as int).compareTo(b.data()['clientTsMs'] as int));
      expect(docs.length, 3);
      expect(docs[1].data()['wallMs'], 300, reason: '10_300 - 10_000, the submitted round');
      expect(docs[2].data()['wallMs'], 600, reason: '10_900 - 10_300, the first idle poll');
    });
  });
}
