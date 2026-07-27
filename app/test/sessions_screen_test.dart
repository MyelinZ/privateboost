import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/dataset.dart';
import 'package:privateboost_app/joined_sessions.dart';
import 'package:privateboost_app/providers/auth.dart';
import 'package:privateboost_app/screens/sessions_screen.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart';
import 'package:privateboost_app/wake_loop.dart' show WakeBridge;
import 'package:shared_preferences/shared_preferences.dart';

/// In-memory stand-in for [JoinedSessions]'s disk store. `SessionsScreen`
/// calls [JoinedSessions.ids]/[JoinedSessions.join] itself from `initState`
/// and from a tap handler, so the test has no seam to wrap those particular
/// calls in `tester.runAsync()`; a real file read started inside a widget
/// test's fake-clock zone has no `runAsync` wrapper and so never completes
/// (a wire that is orthogonal to the auto-refresh timer this suite already
/// has to work around). Overriding just the two methods the screen calls
/// keeps everything on plain, immediately-resolving Dart futures instead.
class _InMemoryJoinedSessions extends JoinedSessions {
  final Map<String, Set<String>> _byUid = {};

  @override
  Future<Set<String>> ids(String uid) async => {...?_byUid[uid]};

  @override
  Future<void> join(String uid, String sessionId) async {
    (_byUid[uid] ??= {}).add(sessionId);
  }
}

/// Scripts [listSessions] with one [SessionListResult] per call (the last one
/// repeats), and fails the test outright if [stepSession] is ever called --
/// the sessions screen must only ever join a session via [JoinedSessions.join]
/// and leave advancing it to the wake loop, never drive a round itself.
class _FakeSessionsBridge implements WakeBridge {
  _FakeSessionsBridge(this._results);

  final List<SessionListResult> _results;
  int callCount = 0;

  final enrolled = <String>[];
  EnrollResult enrollResult = EnrollResult(ok: true);

  @override
  Future<SessionListResult> listSessions({
    required String aggEndpoint,
    required String idToken,
  }) async {
    final result = _results[callCount < _results.length ? callCount : _results.length - 1];
    callCount++;
    return result;
  }

  @override
  Future<EnrollResult> enrollSession({
    required String aggEndpoint,
    required String idToken,
    required String sessionId,
  }) async {
    enrolled.add(sessionId);
    return enrollResult;
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
    fail('SessionsScreen must never step a round directly; the wake loop does that');
  }
}

/// Stands in for the on-disk store when a test needs `join` itself to fail
/// (e.g. a network error), rather than merely recording a successful join.
class _ThrowingJoinedSessions extends JoinedSessions {
  @override
  Future<Set<String>> ids(String uid) async => {};

  @override
  Future<void> join(String uid, String sessionId) async {
    throw Exception('join failed');
  }
}

const _uid = 'uid-a';

/// [SessionsScreen] runs a `Timer.periodic` auto-refresh for as long as it is
/// mounted, so `pumpAndSettle()`, which keeps pumping until no frame is
/// scheduled, never converges: it just keeps crossing the refresh period
/// and scheduling another one. Settle with a bounded number of short pumps
/// instead, comfortably enough to drain the screen's own async loads (which
/// resolve through ordinary Futures, not fake-clock timers) without ever
/// advancing virtual time anywhere near the auto-refresh interval.
Future<void> _settle(WidgetTester tester) async {
  for (var i = 0; i < 10; i++) {
    await tester.pump(const Duration(milliseconds: 10));
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late JoinedSessions joined;
  late int heartFeatures;

  setUpAll(() async {
    heartFeatures = await datasetFeatureCount('heart_disease');
  });

  setUp(() {
    joined = _InMemoryJoinedSessions();
    // The screen watches foregroundPollingProvider, which reads SharedPreferences.
    SharedPreferences.setMockInitialValues({});
  });

  Widget wrap(Widget child) => ProviderScope(
        overrides: [
          authStateProvider.overrideWith((ref) => Stream.value(null)),
          credsProvider.overrideWith((ref) async => const Creds(uid: _uid, idToken: 'tok')),
        ],
        child: MaterialApp(home: child),
      );

  SessionSummary hosted(
    String sessionId, {
    SessionPhase phase = SessionPhase.training,
    String datasetId = 'heart_disease',
    int? nFeatures,
  }) =>
      SessionSummary(
        sessionId: sessionId,
        phase: phase,
        datasetId: datasetId,
        nFeatures: nFeatures ?? heartFeatures,
      );

  group('batchBannerText', () {
    test('renders the 0-indexed id 1-based, with count, size and row range', () {
      expect(
        batchBannerText(1, 4, 59, 59, 118),
        'Batch 2 of 4 - 59 records (rows 59-118)',
      );
    });

    test('covers the whole split for the default single batch', () {
      expect(
        batchBannerText(0, 1, 237, 0, 237),
        'Batch 1 of 1 - 237 records (rows 0-237)',
      );
    });
  });

  testWidgets('lists each session with its dataset, id and phase, and a Joined badge when joined',
      (tester) async {
    await joined.join(_uid, 'session-A');
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [
        hosted('session-A'),
        hosted('session-B', phase: SessionPhase.statsPending),
      ]),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.textContaining('session-A'), findsOneWidget);
    expect(find.textContaining('session-B'), findsOneWidget);
    expect(find.textContaining('heart_disease'), findsNWidgets(2));
    expect(find.text('Joined'), findsOneWidget);
  });

  testWidgets(
      'a session naming a dataset this device does not bundle is shown not contributable, '
      'with the reason visible', (tester) async {
    final bridge = _FakeSessionsBridge([
      const SessionListResult(sessions: [
        SessionSummary(
          sessionId: 'session-X',
          phase: SessionPhase.training,
          datasetId: 'no_such_dataset',
          nFeatures: 3,
        ),
      ]),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.text('Contribute'), findsNothing);
    expect(find.textContaining('no_such_dataset'), findsWidgets);
    expect(find.textContaining('does not have'), findsOneWidget);
  });

  testWidgets(
      'a session whose nFeatures does not match this device\'s bundled dataset is also shown '
      'not contributable, with the reason visible', (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A', nFeatures: heartFeatures + 1)]),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.text('Contribute'), findsNothing);
    expect(find.textContaining('features'), findsWidgets);
  });

  testWidgets('tapping Contribute on an open, compatible session joins it', (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A')]),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.text('Contribute'), findsOneWidget);
    await tester.tap(find.text('Contribute'));
    await _settle(tester);

    expect(await joined.ids(_uid), contains('session-A'));
    expect(find.text('Joined'), findsOneWidget);
    expect(find.text('Contribute'), findsNothing);
  });

  testWidgets('join tap fires a best-effort server enrollment', (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A')]),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    await tester.tap(find.text('Contribute'));
    await _settle(tester);

    expect(bridge.enrolled, ['session-A']);
  });

  testWidgets('join succeeds even when enrollment fails', (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A')]),
    ])..enrollResult = EnrollResult(ok: false, lastError: 'down');

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    await tester.tap(find.text('Contribute'));
    await _settle(tester);

    expect(await joined.ids(_uid), contains('session-A'));
    expect(find.text('Joined'), findsOneWidget);
    expect(find.text('Contribute'), findsNothing);
    expect(find.textContaining('failed'), findsNothing);
  });

  testWidgets(
      'a Contribute tap that fails to join surfaces a visible error instead of throwing '
      'out of the test', (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A')]),
    ]);

    await tester.pumpWidget(
      wrap(SessionsScreen(bridge: bridge, joinedSessions: _ThrowingJoinedSessions())),
    );
    await _settle(tester);

    expect(find.text('Contribute'), findsOneWidget);
    await tester.tap(find.text('Contribute'));
    await _settle(tester);

    expect(find.textContaining('failed to join'), findsOneWidget);
    // Not joined: the Contribute button stays so the user can retry.
    expect(find.text('Contribute'), findsOneWidget);
    expect(find.text('Joined'), findsNothing);
  });

  testWidgets('a completed session shows no Contribute control', (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A', phase: SessionPhase.completed)]),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.text('Contribute'), findsNothing);
    expect(find.textContaining('Completed'), findsOneWidget);
  });

  testWidgets('a failed listSessions call shows an error, not an empty "no sessions" list',
      (tester) async {
    final bridge = _FakeSessionsBridge([
      const SessionListResult(sessions: [], lastError: 'network down'),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.textContaining('network down'), findsOneWidget);
    expect(find.textContaining('No sessions'), findsNothing);
  });

  testWidgets(
      'an automatic refresh failure keeps showing a previously loaded list instead of the '
      'error banner', (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A')]),
      const SessionListResult(sessions: [], lastError: 'network down'),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);
    expect(find.textContaining('session-A'), findsOneWidget);

    await tester.pump(const Duration(seconds: 30)); // periodic auto-refresh tick
    await _settle(tester);

    expect(bridge.callCount, 2);
    expect(find.textContaining('session-A'), findsOneWidget);
    expect(find.textContaining('network down'), findsNothing);
  });

  testWidgets('a pull-to-refresh failure surfaces the banner even with a list already shown',
      (tester) async {
    final bridge = _FakeSessionsBridge([
      SessionListResult(sessions: [hosted('session-A')]),
      const SessionListResult(sessions: [], lastError: 'network down'),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);
    expect(find.textContaining('session-A'), findsOneWidget);

    await tester.widget<RefreshIndicator>(find.byType(RefreshIndicator)).onRefresh();
    await _settle(tester);

    expect(find.textContaining('network down'), findsOneWidget);
  });

  testWidgets('a successful refresh after a failure clears the error banner', (tester) async {
    final bridge = _FakeSessionsBridge([
      const SessionListResult(sessions: [], lastError: 'network down'),
      SessionListResult(sessions: [hosted('session-A')]),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);
    expect(find.textContaining('network down'), findsOneWidget);

    await tester.pump(const Duration(seconds: 30)); // periodic auto-refresh tick
    await _settle(tester);

    expect(find.textContaining('network down'), findsNothing);
    expect(find.textContaining('session-A'), findsOneWidget);
  });

  testWidgets('a genuinely empty session list is an explicit empty state, not an error',
      (tester) async {
    final bridge = _FakeSessionsBridge([const SessionListResult(sessions: [])]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.textContaining('No sessions'), findsOneWidget);
  });

  testWidgets('shows the batch-slice banner', (tester) async {
    final bridge = _FakeSessionsBridge([const SessionListResult(sessions: [])]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.textContaining('Batch 1 of 1'), findsOneWidget);
  });

  testWidgets('keeps the sign-out control', (tester) async {
    final bridge = _FakeSessionsBridge([const SessionListResult(sessions: [])]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    expect(find.text('Sign out'), findsOneWidget);
  });

  testWidgets('auto-refreshes the session list while the screen stays visible', (tester) async {
    final bridge = _FakeSessionsBridge([
      const SessionListResult(sessions: []),
      const SessionListResult(sessions: []),
      const SessionListResult(sessions: []),
    ]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);
    expect(bridge.callCount, 1);

    await tester.pump(const Duration(seconds: 30));
    await _settle(tester);
    expect(bridge.callCount, 2);

    await tester.pump(const Duration(seconds: 30));
    await _settle(tester);
    expect(bridge.callCount, 3);
  });

  testWidgets('foreground polling switch renders, defaults off, and persists when toggled on',
      (tester) async {
    final bridge = _FakeSessionsBridge([const SessionListResult(sessions: [])]);

    await tester.pumpWidget(wrap(SessionsScreen(bridge: bridge, joinedSessions: joined)));
    await _settle(tester);

    final switchTile = find.byType(SwitchListTile);
    expect(switchTile, findsOneWidget);
    expect(tester.widget<SwitchListTile>(switchTile).value, isFalse);

    await tester.tap(switchTile);
    await _settle(tester);

    expect(tester.widget<SwitchListTile>(switchTile).value, isTrue);
    final prefs = await SharedPreferences.getInstance();
    expect(prefs.getBool('foreground_polling_enabled'), isTrue);
  });
}
