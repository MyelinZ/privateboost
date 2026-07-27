import 'dart:io';

import 'package:fake_cloud_firestore/fake_cloud_firestore.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/dataset.dart';
import 'package:privateboost_app/joined_sessions.dart';
import 'package:privateboost_app/messaging.dart' show metricsRecordingHook;
import 'package:privateboost_app/metrics/context.dart';
import 'package:privateboost_app/metrics/metrics_writer.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart';
import 'package:privateboost_app/wake_loop.dart';

/// Records every call so tests can assert exactly which sessions were
/// stepped, and returns scripted results per session id, one consumed per
/// call, so a test can script a session's first wake differently from its
/// second. A session with no script left returns `NothingNew` naming itself,
/// so a test that forgot to script a call fails with a clear result rather
/// than a null-check crash.
class _FakeWakeBridge implements WakeBridge {
  _FakeWakeBridge({
    this.listSessionsResult = const SessionListResult(sessions: []),
    Map<String, List<RoundStepResult>>? stepResults,
  }) : stepResults = stepResults ?? {};

  final SessionListResult listSessionsResult;
  final Map<String, List<RoundStepResult>> stepResults;

  int listSessionsCallCount = 0;
  final List<String> steppedSessionIds = [];
  final List<BigInt> steppedWithWatermark = [];
  final List<List<TrainRow>> steppedWithRecords = [];

  @override
  Future<SessionListResult> listSessions({
    required String aggEndpoint,
    required String idToken,
  }) async {
    listSessionsCallCount++;
    return listSessionsResult;
  }

  @override
  Future<EnrollResult> enrollSession({
    required String aggEndpoint,
    required String idToken,
    required String sessionId,
  }) async => EnrollResult(ok: true);

  @override
  Future<RoundStepResult> stepSession({
    required String aggEndpoint,
    required String idToken,
    required String sessionId,
    required List<TrainRow> records,
    required bool hidePath,
    required BigInt lastSeenRoundId,
  }) async {
    steppedSessionIds.add(sessionId);
    steppedWithWatermark.add(lastSeenRoundId);
    steppedWithRecords.add(records);
    final queue = stepResults[sessionId];
    if (queue == null || queue.isEmpty) {
      return RoundStepResult(
        outcome: RoundStepOutcome.nothingNew,
        roundId: BigInt.zero,
        sessionId: sessionId,
        lastSeenRoundId: lastSeenRoundId,
      );
    }
    return queue.removeAt(0);
  }
}

RoundStepResult _submitted(String sessionId, BigInt roundId) => RoundStepResult(
      outcome: RoundStepOutcome.submitted,
      roundId: roundId,
      sessionId: sessionId,
      lastSeenRoundId: roundId,
    );

RoundStepResult _nothingNew(String sessionId, BigInt lastSeen) => RoundStepResult(
      outcome: RoundStepOutcome.nothingNew,
      roundId: BigInt.zero,
      sessionId: sessionId,
      lastSeenRoundId: lastSeen,
    );

RoundStepResult _completed(String sessionId, BigInt lastSeen) => RoundStepResult(
      outcome: RoundStepOutcome.completed,
      roundId: BigInt.zero,
      sessionId: sessionId,
      lastSeenRoundId: lastSeen,
    );

RoundStepResult _failed(String sessionId, BigInt lastSeen) => RoundStepResult(
      outcome: RoundStepOutcome.failed,
      roundId: BigInt.zero,
      sessionId: sessionId,
      lastSeenRoundId: lastSeen,
    );

/// An error outcome that deliberately reports a *different* watermark than
/// the one it was called with, so a test can prove the loop does not trust a
/// misbehaving bridge, it must persist nothing on `Error` regardless of
/// what `lastSeenRoundId` comes back.
RoundStepResult _errorReporting(String sessionId, BigInt suspiciousWatermark) => RoundStepResult(
      outcome: RoundStepOutcome.error,
      roundId: BigInt.zero,
      sessionId: sessionId,
      lastSeenRoundId: suspiciousWatermark,
      lastError: 'boom',
    );

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const uid = 'uid-a';
  late Directory tmp;
  late JoinedSessions joined;
  late SessionWatermarks marks;
  late int heartFeatures;
  late int breastFeatures;

  setUpAll(() async {
    heartFeatures = await datasetFeatureCount('heart_disease');
    breastFeatures = await datasetFeatureCount('breast_cancer');
  });

  setUp(() async {
    tmp = await Directory.systemTemp.createTemp('pbr_wake_loop_test');
    joined = JoinedSessions(directory: tmp);
    marks = SessionWatermarks(directory: tmp);
  });

  tearDown(() async {
    if (await tmp.exists()) await tmp.delete(recursive: true);
  });

  SessionSummary hosted(String sessionId, {String datasetId = 'heart_disease', int? nFeatures}) =>
      SessionSummary(
        sessionId: sessionId,
        phase: SessionPhase.training,
        datasetId: datasetId,
        nFeatures: nFeatures ?? (datasetId == 'breast_cancer' ? breastFeatures : heartFeatures),
      );

  Future<WakeResult> wake(
    String trigger,
    _FakeWakeBridge bridge, {
    int? wakeLatencyMs,
    WakeStepHook onRoundStep = _noop,
  }) =>
      runWake(
        trigger: trigger,
        uid: uid,
        idToken: 'tok',
        aggEndpoint: 'http://example.invalid',
        joinedSessions: joined,
        watermarks: marks,
        bridge: bridge,
        wakeLatencyMs: wakeLatencyMs,
        onRoundStep: onRoundStep,
      );

  test('only joined sessions are advanced, even when others are hosted', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A'), hosted('session-B')]),
      stepResults: {'session-A': [_submitted('session-A', BigInt.one)]},
    );

    await wake('push', bridge);

    expect(bridge.steppedSessionIds, ['session-A']);
  });

  test('a session reported COMPLETED is marked finished and not stepped again', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
      stepResults: {'session-A': [_completed('session-A', BigInt.zero)]},
    );

    await wake('push', bridge);
    expect(bridge.steppedSessionIds, ['session-A']);
    expect(await joined.active(uid), isEmpty);

    await wake('push', bridge);
    expect(
      bridge.steppedSessionIds,
      ['session-A'],
      reason: 'a session marked finished locally must not be stepped again',
    );
  });

  test('a session reported FAILED is marked finished and not stepped again', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
      stepResults: {'session-A': [_failed('session-A', BigInt.zero)]},
    );

    await wake('push', bridge);
    expect(await joined.active(uid), isEmpty);

    await wake('push', bridge);
    expect(bridge.steppedSessionIds, ['session-A']);
  });

  test('the watermark is persisted only after a successful step', () async {
    await joined.join(uid, 'session-A');
    await marks.set(uid, 'session-A', BigInt.from(5));
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
      // Reports a different watermark than the input on purpose (see
      // `_errorReporting`'s doc comment).
      stepResults: {'session-A': [_errorReporting('session-A', BigInt.from(999))]},
    );

    await wake('push', bridge);

    expect(
      await marks.get(uid, 'session-A'),
      BigInt.from(5),
      reason: 'a failed step must leave the stored watermark unchanged',
    );
  });

  test('NothingNew also leaves the stored watermark unchanged', () async {
    await joined.join(uid, 'session-A');
    await marks.set(uid, 'session-A', BigInt.from(3));
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
      stepResults: {'session-A': [_nothingNew('session-A', BigInt.from(3))]},
    );

    await wake('push', bridge);

    expect(await marks.get(uid, 'session-A'), BigInt.from(3));
  });

  test('a submitted step persists its watermark and feeds it back as the next lastSeenRoundId',
      () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
      stepResults: {
        'session-A': [_submitted('session-A', BigInt.from(7))],
      },
    );

    await wake('push', bridge);

    expect(await marks.get(uid, 'session-A'), BigInt.from(7));
    expect(bridge.steppedWithWatermark, [BigInt.zero], reason: 'first wake resumes from zero');
  });

  test(
      'a throwing onRoundStep hook does not prevent the watermark from advancing on a '
      'submitted round', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
      stepResults: {'session-A': [_submitted('session-A', BigInt.from(7))]},
    );

    var threw = false;
    try {
      await wake(
        'push',
        bridge,
        onRoundStep: (sessionId, triggerSource, result, datasetId, wakeLatencyMs) async {
          throw StateError('metrics hook boom');
        },
      );
    } catch (_) {
      threw = true;
    }

    expect(threw, isTrue, reason: 'sanity check: the hook really did throw');
    expect(
      await marks.get(uid, 'session-A'),
      BigInt.from(7),
      reason: 'the watermark is persisted before the hook runs, so a hook that throws after a '
          'successful submit can never suppress the persist and cause the next wake to '
          're-submit (and double-count) this round',
    );
  });

  test('one wake advances a session by at most one round', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
      stepResults: {
        'session-A': [_submitted('session-A', BigInt.one), _submitted('session-A', BigInt.two)],
      },
    );

    await wake('push', bridge);

    expect(bridge.steppedSessionIds.length, 1);
  });

  test('all three triggers behave identically, differing only in the recorded triggerSource',
      () async {
    for (final trigger in ['push', 'workmanager', 'foreground']) {
      await joined.join(uid, 'session-A');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
        stepResults: {'session-A': [_submitted('session-A', BigInt.one)]},
      );
      String? seenTrigger;

      await wake(
        trigger,
        bridge,
        onRoundStep: (sessionId, triggerSource, result, datasetId, wakeLatencyMs) async {
          seenTrigger = triggerSource;
        },
      );

      expect(bridge.steppedSessionIds, ['session-A']);
      expect(seenTrigger, trigger);
      await joined.clear(uid);
      await marks.set(uid, 'session-A', BigInt.zero);
    }
  });

  test('a session naming a dataset this device does not bundle is skipped, not crashed',
      () async {
    await joined.join(uid, 'session-A');
    await joined.join(uid, 'session-B');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [
        hosted('session-A', datasetId: 'no_such_dataset', nFeatures: 999999),
        hosted('session-B'),
      ]),
      stepResults: {'session-B': [_submitted('session-B', BigInt.one)]},
    );

    final result = await wake('push', bridge);

    expect(bridge.steppedSessionIds, ['session-B']);
    expect(result.lastError, isNull);
  });

  test('a session whose datasetId is empty (a dataset-less session) is skipped, not guessed',
      () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [hosted('session-A', datasetId: '')]),
    );

    final result = await wake('push', bridge);

    expect(bridge.steppedSessionIds, isEmpty);
    expect(result.lastError, isNull);
  });

  test(
      'a session naming a dataset the device has is stepped with that dataset\'s records',
      () async {
    await joined.join(uid, 'session-A');
    await joined.join(uid, 'session-B');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [
        hosted('session-A', datasetId: 'heart_disease'),
        hosted('session-B', datasetId: 'breast_cancer'),
      ]),
      stepResults: {
        'session-A': [_submitted('session-A', BigInt.one)],
        'session-B': [_submitted('session-B', BigInt.one)],
      },
    );

    await wake('push', bridge);

    expect(bridge.steppedSessionIds, ['session-A', 'session-B']);
    final heartRows = await loadTrainRows('heart_disease');
    final breastRows = await loadTrainRows('breast_cancer');
    expect(bridge.steppedWithRecords[0].length, heartRows.length);
    expect(bridge.steppedWithRecords[0].first.$1.length, heartFeatures);
    expect(bridge.steppedWithRecords[1].length, breastRows.length);
    expect(bridge.steppedWithRecords[1].first.$1.length, breastFeatures);
  });

  test('a datasetId/n_features mismatch is skipped, not submitted', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: SessionListResult(sessions: [
        hosted('session-A', datasetId: 'heart_disease', nFeatures: heartFeatures + 1),
      ]),
    );

    final result = await wake('push', bridge);

    expect(bridge.steppedSessionIds, isEmpty);
    expect(result.lastError, isNull);
  });

  test('a session no longer hosted by the aggregator is simply skipped this wake', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(listSessionsResult: const SessionListResult(sessions: []));

    final result = await wake('push', bridge);

    expect(bridge.steppedSessionIds, isEmpty);
    expect(result.stepped, isEmpty);
  });

  test('listSessions failing entirely steps nothing and surfaces the error', () async {
    await joined.join(uid, 'session-A');
    final bridge = _FakeWakeBridge(
      listSessionsResult: const SessionListResult(sessions: [], lastError: 'network down'),
    );

    final result = await wake('push', bridge);

    expect(bridge.steppedSessionIds, isEmpty);
    expect(result.lastError, 'network down');
  });

  test('no joined sessions: the wake does not even list sessions', () async {
    final bridge = _FakeWakeBridge();

    await wake('push', bridge);

    expect(bridge.listSessionsCallCount, 0);
  });

  group('closing the wake-loop metrics blackout', () {
    // A background wake now writes its own `paperSimRoundMetrics` documents
    // via `onRoundStep: metricsRecordingHook(...)`, the exact hook builder
    // `runBackgroundWake` (`messaging.dart`) calls, exercised here directly
    // rather than through a hand-rolled stand-in closure, so a transposed
    // parameter in that builder would fail this suite.
    const context = MetricsContext(
      deviceId: 'd1',
      deviceModel: 'model',
      appVersion: '1.0.0+1',
    );

    late SessionRoundTimestamps timestamps;

    setUp(() {
      timestamps = SessionRoundTimestamps(directory: tmp);
    });

    RoundStepResult submittedWithSummary(String sessionId, BigInt roundId, int nRecords) =>
        RoundStepResult(
          outcome: RoundStepOutcome.submitted,
          roundId: roundId,
          sessionId: sessionId,
          lastSeenRoundId: roundId,
          summary: RoundSummary(
            roundId: roundId,
            sessionId: sessionId,
            nRecords: nRecords,
            roundKind: RoundKind.gradient,
            treeIdx: 0,
            depth: 0,
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

    /// A realistic `NothingNew` step: no summary, but the idle-poll fields a
    /// real bridge populates for it.
    RoundStepResult nothingNewWithIdle(String sessionId, {BigInt? pollUs}) => RoundStepResult(
          outcome: RoundStepOutcome.nothingNew,
          roundId: BigInt.zero,
          sessionId: sessionId,
          lastSeenRoundId: BigInt.zero,
          idle: IdlePoll(sessionId: sessionId, pollUs: pollUs ?? BigInt.from(15000)),
        );

    WakeStepHook hookFor(FakeFirebaseFirestore fake, {int Function()? clock}) =>
        metricsRecordingHook(
          context: context,
          uid: uid,
          batchId: 0,
          batchCount: 1,
          timestamps: timestamps,
          firestore: fake,
          clock: clock ?? (() => DateTime.now().millisecondsSinceEpoch),
        );

    test('a wake-stepped submitted round writes exactly one paperSimRoundMetrics '
        'document, with datasetId and nRecords and no recordIndex', () async {
      await joined.join(uid, 'session-A');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
        stepResults: {'session-A': [submittedWithSummary('session-A', BigInt.one, 42)]},
      );
      final fake = FakeFirebaseFirestore();

      await wake('push', bridge, onRoundStep: hookFor(fake));

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs.length, 1);
      final d = docs.single.data();
      expect(d['sessionId'], 'session-A');
      expect(d['datasetId'], 'heart_disease');
      expect(d['nRecords'], 42);
      expect(d.containsKey('recordIndex'), isFalse);
    });

    test('a wake that steps two sessions writes one document per submitted round',
        () async {
      await joined.join(uid, 'session-A');
      await joined.join(uid, 'session-B');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [
          hosted('session-A', datasetId: 'heart_disease'),
          hosted('session-B', datasetId: 'breast_cancer'),
        ]),
        stepResults: {
          'session-A': [submittedWithSummary('session-A', BigInt.one, 10)],
          'session-B': [submittedWithSummary('session-B', BigInt.one, 20)],
        },
      );
      final fake = FakeFirebaseFirestore();

      await wake('push', bridge, onRoundStep: hookFor(fake));

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs.length, 2);
      final datasetIds = docs.map((d) => d.data()['datasetId']).toSet();
      expect(datasetIds, {'heart_disease', 'breast_cancer'});
    });

    test('a step with neither a summary nor idle-poll info writes no document', () async {
      await joined.join(uid, 'session-A');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
        // No scripted result -> the fake bridge returns NothingNew with no
        // idle-poll info, same as a Completed/Failed/Error step.
      );
      final fake = FakeFirebaseFirestore();

      await wake('push', bridge, onRoundStep: hookFor(fake));

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs, isEmpty);
    });

    test('a background wake\'s idle poll writes one document with outcome idle',
        () async {
      await joined.join(uid, 'session-A');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
        stepResults: {
          'session-A': [nothingNewWithIdle('session-A', pollUs: BigInt.from(12345))],
        },
      );
      final fake = FakeFirebaseFirestore();

      await wake('push', bridge, onRoundStep: hookFor(fake));

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs.length, 1);
      final d = docs.single.data();
      expect(d['sessionId'], 'session-A');
      expect(d['outcome'], 'idle');
      expect(d['pollUs'], 12345);
      expect(d.containsKey('computeUs'), isFalse);
      expect(d.containsKey('submitUs'), isFalse);
    });

    test("the foreground poller's idle poll writes no document", () async {
      await joined.join(uid, 'session-A');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
        stepResults: {
          'session-A': [nothingNewWithIdle('session-A')],
        },
      );
      final fake = FakeFirebaseFirestore();

      await wake('foreground', bridge, onRoundStep: hookFor(fake));

      final docs = (await fake.collection(roundMetricsCollection).get()).docs;
      expect(docs, isEmpty,
          reason: 'the foreground poller ticks at a fixed cadence, so its idle polls carry no '
              'organic check-in signal and must not be recorded');
    });

    test(
        'the first wake-driven round of a session has wallMs = 0, not the raw '
        'epoch-ms clientTsMs', () async {
      await joined.join(uid, 'session-A');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
        stepResults: {'session-A': [submittedWithSummary('session-A', BigInt.one, 10)]},
      );
      final fake = FakeFirebaseFirestore();

      await wake('push', bridge, onRoundStep: hookFor(fake, clock: () => 1752000000000));

      final d = (await fake.collection(roundMetricsCollection).get()).docs.single.data();
      expect(d['wallMs'], 0,
          reason: 'no prior round for this session: there is no measurable gap');
    });

    test(
        "a session's second wake-driven round has wallMs equal to the gap between the "
        "two rounds' clientTsMs, not the intra-wake time", () async {
      await joined.join(uid, 'session-A');
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [hosted('session-A')]),
        stepResults: {
          'session-A': [
            submittedWithSummary('session-A', BigInt.one, 10),
            submittedWithSummary('session-A', BigInt.two, 10),
          ],
        },
      );
      final fake = FakeFirebaseFirestore();

      // Two separate `wake()` calls, each with the hook this wake would build
      // fresh, standing in for two wakes hours apart. Only `timestamps`
      // (backed by the shared `tmp` directory) survives between them, exactly
      // as the on-disk store does between real wakes.
      await wake('push', bridge, onRoundStep: hookFor(fake, clock: () => 1_000_000));
      // One hour (3_600_000ms) later, a different wake entirely.
      await wake('push', bridge, onRoundStep: hookFor(fake, clock: () => 4_600_000));

      final docs = (await fake.collection(roundMetricsCollection).get()).docs
        ..sort((a, b) => (a.data()['clientTsMs'] as int).compareTo(b.data()['clientTsMs'] as int));
      expect(docs.length, 2);
      expect(docs[1].data()['wallMs'], 3_600_000);
    });

    test("two sessions stepped in one wake do not contaminate each other's wallMs",
        () async {
      await joined.join(uid, 'session-A');
      await joined.join(uid, 'session-B');
      await timestamps.set(uid, 'session-A', 10_000);
      final bridge = _FakeWakeBridge(
        listSessionsResult: SessionListResult(sessions: [
          hosted('session-A', datasetId: 'heart_disease'),
          hosted('session-B', datasetId: 'breast_cancer'),
        ]),
        stepResults: {
          'session-A': [submittedWithSummary('session-A', BigInt.from(2), 10)],
          'session-B': [submittedWithSummary('session-B', BigInt.one, 20)],
        },
      );
      final fake = FakeFirebaseFirestore();
      final clocks = [10_500, 10_600];
      var callCount = 0;

      // A single hook instance, built once as `runBackgroundWake` builds one
      // per wake, stepping both sessions, proving neither reads or writes
      // the other's stored timestamp.
      await wake('push', bridge, onRoundStep: hookFor(fake, clock: () => clocks[callCount++]));

      final docs = {
        for (final d in (await fake.collection(roundMetricsCollection).get()).docs)
          d.data()['sessionId'] as String: d.data(),
      };
      expect(docs['session-A']!['wallMs'], 500,
          reason: "10_500 - 10_000, session-A's own prior round");
      expect(
        docs['session-B']!['wallMs'],
        0,
        reason: 'session-B has no prior round of its own, so there is no gap to measure: '
            "wallMs is 0, not a gap measured against session-A's stored timestamp",
      );
    });
  });

  group('SessionWatermarks', () {
    test('an unset (uid, sessionId) reads as zero', () async {
      expect(await marks.get(uid, 'session-A'), BigInt.zero);
    });

    test('round-trips a set value', () async {
      await marks.set(uid, 'session-A', BigInt.from(42));
      expect(await marks.get(uid, 'session-A'), BigInt.from(42));
    });

    test('a corrupt store file reads as zero rather than throwing', () async {
      final file = File('${tmp.path}/session_watermarks.json');
      await file.writeAsString('{not valid json');

      expect(await marks.get(uid, 'session-A'), BigInt.zero);

      // A write after a corrupt read must still succeed.
      await marks.set(uid, 'session-A', BigInt.from(1));
      expect(await marks.get(uid, 'session-A'), BigInt.from(1));
    });

    test('a corrupt round id value reads as zero rather than throwing', () async {
      final file = File('${tmp.path}/session_watermarks.json');
      await file.writeAsString('{"$uid": {"session-A": "not-a-number"}}');

      expect(await marks.get(uid, 'session-A'), BigInt.zero);
    });

    test('a second instance over the same directory sees the first instance\'s writes', () async {
      final other = SessionWatermarks(directory: tmp);
      await marks.set(uid, 'session-A', BigInt.from(9));
      expect(await other.get(uid, 'session-A'), BigInt.from(9));
    });
  });

  group('SessionRoundTimestamps', () {
    test('a second instance over the same directory sees the first instance\'s writes', () async {
      final timestamps = SessionRoundTimestamps(directory: tmp);
      final other = SessionRoundTimestamps(directory: tmp);
      await timestamps.set(uid, 'session-A', 1234);
      expect(await other.get(uid, 'session-A'), 1234);
    });
  });
}

Future<void> _noop(
  String sessionId,
  String triggerSource,
  RoundStepResult result,
  String datasetId,
  int? wakeLatencyMs,
) async {}
