import 'dart:async';

import 'package:firebase_ui_auth/firebase_ui_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:privateboost_app/dataset.dart';
import 'package:privateboost_app/demo_config.dart';
import 'package:privateboost_app/joined_sessions.dart';
import 'package:privateboost_app/messaging.dart' show runBackgroundWake;
import 'package:privateboost_app/providers/auth.dart';
import 'package:privateboost_app/providers/settings.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart' show SessionPhase, SessionSummary;
import 'package:privateboost_app/wake_loop.dart';

/// How often the screen re-lists while visible, on top of pull-to-refresh.
/// Long enough not to hammer the aggregator from every open phone in the fleet,
/// short enough that a phase change appears without a pull.
const _autoRefreshInterval = Duration(seconds: 30);

/// Which of the fleet's [batchCount] slices this device runs, how many records
/// it holds, and its `[start, end)` range in the train split. [batchId] is the
/// 0-indexed build define, rendered 1-based because "Batch 0 of 3" reads as an
/// off-by-one. Pure, so it is testable without a widget.
String batchBannerText(int batchId, int batchCount, int n, int start, int end) =>
    'Batch ${batchId + 1} of $batchCount - $n records (rows $start-$end)';

/// Signed-in landing screen: every hosted session, so the user chooses which to
/// contribute to. Tapping Contribute records standing consent only; the wake
/// loop advances a joined session round by round from there.
class SessionsScreen extends ConsumerStatefulWidget {
  const SessionsScreen({super.key, this.bridge = const RustWakeBridge(), this.joinedSessions});

  /// Reuses [WakeBridge] rather than a screen-specific interface: it is
  /// already the seam `wake_loop.dart` uses, and one test double is enough.
  final WakeBridge bridge;

  /// Injected in tests to point at a temp directory; `null` in production
  /// resolves the real on-disk store (see [JoinedSessions]).
  final JoinedSessions? joinedSessions;

  @override
  ConsumerState<SessionsScreen> createState() => _SessionsScreenState();
}

class _SessionsScreenState extends ConsumerState<SessionsScreen> {
  late final JoinedSessions _joined = widget.joinedSessions ?? JoinedSessions();

  String _bannerText = 'batch: loading';
  List<SessionSummary>? _sessions;
  Set<String> _joinedIds = {};

  /// Non-null reason a listed session cannot be joined from this device
  /// (dataset not bundled, or bundled with a different feature count),
  /// keyed by session id; a session with no entry here is contributable.
  Map<String, String?> _blockedReasons = {};

  /// Set only when the most recent [listSessions] call itself failed (a
  /// network/auth error), so the UI can tell that apart from a session list
  /// that came back genuinely empty.
  String? _listError;

  final Set<String> _joining = {};

  /// Set only for a session whose most recent Contribute tap failed (a
  /// credential fetch or [JoinedSessions.join] error), keyed by session id, so
  /// the failure is visible instead of the button silently reverting with no
  /// explanation.
  final Map<String, String> _joinErrors = {};
  Timer? _autoRefresh;
  Timer? _fgPoll;

  @override
  void initState() {
    super.initState();
    _loadBanner();
    _refresh(userInitiated: false);
    _autoRefresh =
        Timer.periodic(_autoRefreshInterval, (_) => _refresh(userInitiated: false));
  }

  @override
  void dispose() {
    _autoRefresh?.cancel();
    _fgPoll?.cancel();
    super.dispose();
  }

  /// Starts or stops the foreground poller to match the
  /// [foregroundPollingProvider] switch. Idempotent: an already-running poller
  /// is left in place when re-applied on, so a rebuild never restarts its timer
  /// (and thus never resets the interval mid-flight).
  void _applyForegroundPolling(bool enabled) {
    if (enabled) {
      _fgPoll ??= Timer.periodic(
        Duration(seconds: foregroundPollSeconds),
        (_) => _foregroundWake(),
      );
    } else {
      _fgPoll?.cancel();
      _fgPoll = null;
    }
  }

  /// One tick of the foreground poller, live only while the
  /// [foregroundPollingProvider] switch is on: funnels through the same
  /// [runBackgroundWake] as the push and WorkManager triggers, so the
  /// participation lock serializes it against both and a tick that overlaps a
  /// running wake degrades to a no-op. Refreshes the list afterwards so a
  /// phase change shows without waiting for the next auto-refresh tick.
  Future<void> _foregroundWake() async {
    final result = await runBackgroundWake(trigger: 'foreground');
    debugPrint('privateboost foreground wake: sessions_stepped=${result.stepped.length} '
        'error=${result.lastError}');
    if (mounted) await _refresh(userInitiated: false);
  }

  /// Loads the train split once to fill the batch banner. A bad batch config
  /// (`PBR_BATCH_ID >= PBR_BATCH_COUNT`) or a missing asset degrades to
  /// `batch: unavailable` rather than throwing during the first build.
  Future<void> _loadBanner() async {
    try {
      final slice = batchSlice(await loadTrainRows(), batchId, batchCount);
      if (!mounted) return;
      setState(() {
        _bannerText = batchBannerText(
          batchId,
          batchCount,
          slice.rows.length,
          slice.start,
          slice.end,
        );
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _bannerText = 'batch: unavailable');
    }
  }

  /// Re-lists the aggregator's sessions and this device's joined set. Drives
  /// the initial load, pull-to-refresh, the app-bar refresh action, an
  /// error-banner Retry tap, the periodic auto-refresh, and a foreground-wake
  /// tick alike; [userInitiated] tells a real user tap (pull-to-refresh, the
  /// app-bar icon, Retry) apart from a timer-driven call, which the failure
  /// handling below treats differently.
  Future<void> _refresh({required bool userInitiated}) async {
    try {
      final creds = await ref.read(credsProvider.future);
      if (creds == null) {
        if (!mounted) return;
        if (userInitiated || _sessions == null) setState(() => _listError = 'not signed in');
        return;
      }
      final result =
          await widget.bridge.listSessions(aggEndpoint: aggEndpoint, idToken: creds.idToken);
      final joinedIds = await _joined.ids(creds.uid);
      final reasons =
          result.lastError == null ? await _blockedReasonsFor(result.sessions) : _blockedReasons;
      if (!mounted) return;
      setState(() {
        _joinedIds = joinedIds;
        if (result.lastError == null) {
          _listError = null;
          _sessions = _usefullyOrdered(result.sessions);
          _blockedReasons = reasons;
        } else if (userInitiated || _sessions == null) {
          _listError = result.lastError;
        }
        // else: an automatic refresh failed while a list from an earlier
        // load is still on screen. On a real device, unlocking the phone
        // races the radio waking from Doze, so the fresh dial fails
        // transiently and the next auto-refresh tick usually recovers it;
        // swapping the list for an alarming banner over that would be a
        // false alarm. Leave the list showing and let the next tick retry.
      });
    } catch (e) {
      if (!mounted) return;
      if (userInitiated || _sessions == null) {
        setState(() => _listError = 'failed to load sessions: $e');
      }
    }
  }

  /// Why each of [sessions] cannot be joined from this device, if at all --
  /// the same dataset-availability and feature-count checks `wake_loop.dart`
  /// applies before stepping a joined session, run here up front so an
  /// incompatible session reads as "not contributable" with a stated reason
  /// instead of a Contribute button that would silently do nothing useful.
  Future<Map<String, String?>> _blockedReasonsFor(List<SessionSummary> sessions) async {
    final reasons = <String, String?>{};
    for (final s in sessions) {
      if (!_isOpen(s.phase)) continue; // _action never shows a reason for a closed session
      if (s.datasetId.isEmpty) {
        reasons[s.sessionId] = 'no dataset assigned to this session yet';
        continue;
      }
      try {
        final n = await datasetFeatureCount(s.datasetId);
        reasons[s.sessionId] = n == s.nFeatures
            ? null
            : 'this device\'s "${s.datasetId}" dataset has $n features, but the session '
                'expects ${s.nFeatures}';
      } catch (_) {
        reasons[s.sessionId] = 'this device does not have the "${s.datasetId}" dataset';
      }
    }
    return reasons;
  }

  /// Sessions still open (statsPending/training) before finished ones
  /// (completed/failed): no timestamp crosses the bridge to sort by (the
  /// aggregator returns sessions by session id, a random UUID with no time
  /// signal), so "useful" here means surfacing what a user might still act on
  /// above what is already done.
  List<SessionSummary> _usefullyOrdered(List<SessionSummary> sessions) {
    bool open(SessionSummary s) =>
        s.phase == SessionPhase.statsPending || s.phase == SessionPhase.training;
    return [...sessions.where(open), ...sessions.where((s) => !open(s))];
  }

  /// Records this device's standing consent to contribute to [session] via
  /// [JoinedSessions.join] and nothing else. This deliberately does not call
  /// `stepSession`: the wake loop (`wake_loop.dart`), driven by the FCM push
  /// and WorkManager triggers, is what advances a joined session round by
  /// round, serialized by `ParticipationLock`. If this tap also drove a
  /// round directly, it would race that serialization instead of going
  /// through it, so it only joins and lets the loop take it from there.
  Future<void> _contribute(SessionSummary session) async {
    setState(() {
      _joining.add(session.sessionId);
      _joinErrors.remove(session.sessionId);
    });
    try {
      final creds = await ref.read(credsProvider.future);
      if (creds == null) return; // signed out mid-tap; nothing to join with
      await _joined.join(creds.uid, session.sessionId);
      // Fire-and-forget: the wake loop re-enrolls on every resume, so a
      // failure here only delays the aggregator learning of this join until
      // the next wake, it must never block or fail the join itself.
      unawaited(() async {
        try {
          final r = await widget.bridge.enrollSession(
            aggEndpoint: aggEndpoint,
            idToken: creds.idToken,
            sessionId: session.sessionId,
          );
          if (!r.ok) {
            debugPrint('privateboost: enroll on join failed: ${r.lastError}');
          }
        } catch (e) {
          debugPrint('privateboost: enroll on join failed: $e');
        }
      }());
      if (!mounted) return;
      setState(() => _joinedIds = {..._joinedIds, session.sessionId});
    } catch (e) {
      if (!mounted) return;
      setState(() => _joinErrors[session.sessionId] = 'failed to join: $e');
    } finally {
      if (mounted) setState(() => _joining.remove(session.sessionId));
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = ref.watch(authStateProvider).value;
    ref.listen(foregroundPollingProvider, (_, next) {
      _applyForegroundPolling(next.value ?? false);
    });
    final foregroundPolling = ref.watch(foregroundPollingProvider).value ?? false;

    return Scaffold(
      appBar: AppBar(
        title: const Text('PrivateBoost'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => _refresh(userInitiated: true),
          ),
        ],
      ),
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: () => _refresh(userInitiated: true),
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
              Text(_bannerText, style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 16),
              if (user?.email case final email?) Text('Signed in as $email'),
              const SizedBox(height: 16),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Foreground polling'),
                subtitle: Text(
                  'Advance joined rounds every ${foregroundPollSeconds}s while this screen is open',
                ),
                value: foregroundPolling,
                onChanged: (v) => ref.read(foregroundPollingProvider.notifier).set(v),
              ),
              const SizedBox(height: 16),
              ..._sessionsSection(),
              const SizedBox(height: 24),
              OutlinedButton(
                onPressed: () => FirebaseUIAuth.signOut(context: context),
                child: const Text('Sign out'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  List<Widget> _sessionsSection() {
    if (_listError case final err?) {
      return [_ErrorBanner(message: err, onRetry: () => _refresh(userInitiated: true))];
    }
    final sessions = _sessions;
    if (sessions == null) {
      return [const Center(child: CircularProgressIndicator())];
    }
    if (sessions.isEmpty) {
      return [const Text('No sessions hosted right now.')];
    }
    return [
      for (final s in sessions) ...[_SessionCard(session: s, state: _cardStateFor(s)), const SizedBox(height: 8)],
    ];
  }

  _SessionCardState _cardStateFor(SessionSummary s) => _SessionCardState(
        joined: _joinedIds.contains(s.sessionId),
        blockedReason: _blockedReasons[s.sessionId],
        joining: _joining.contains(s.sessionId),
        joinError: _joinErrors[s.sessionId],
        onContribute: () => _contribute(s),
      );
}

/// What [_SessionCard] needs to render one session, computed by the screen
/// state so the card itself stays a plain, stateless render of its inputs.
class _SessionCardState {
  const _SessionCardState({
    required this.joined,
    required this.blockedReason,
    required this.joining,
    required this.joinError,
    required this.onContribute,
  });

  final bool joined;
  final String? blockedReason;
  final bool joining;
  final String? joinError;
  final VoidCallback onContribute;
}

String _phaseLabel(SessionPhase phase) => switch (phase) {
      SessionPhase.statsPending => 'Preparing',
      SessionPhase.training => 'Training',
      SessionPhase.completed => 'Completed',
      SessionPhase.failed => 'Failed',
    };

bool _isOpen(SessionPhase phase) =>
    phase == SessionPhase.statsPending || phase == SessionPhase.training;

class _SessionCard extends StatelessWidget {
  const _SessionCard({required this.session, required this.state});

  final SessionSummary session;
  final _SessionCardState state;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(session.sessionId, style: Theme.of(context).textTheme.titleMedium),
                ),
                if (state.joined) const Chip(label: Text('Joined')),
              ],
            ),
            Text(
              session.datasetId.isEmpty ? 'Dataset: (none yet)' : 'Dataset: ${session.datasetId}',
            ),
            Text('Phase: ${_phaseLabel(session.phase)}'),
            const SizedBox(height: 8),
            _action(context),
          ],
        ),
      ),
    );
  }

  Widget _action(BuildContext context) {
    if (state.joined) return const SizedBox.shrink();
    if (!_isOpen(session.phase)) return const SizedBox.shrink();
    if (state.blockedReason case final reason?) {
      return Text(
        'Not contributable: $reason',
        style: TextStyle(color: Theme.of(context).colorScheme.error),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        FilledButton(
          onPressed: state.joining ? null : state.onContribute,
          child: state.joining
              ? const SizedBox(
                  height: 16,
                  width: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Text('Contribute'),
        ),
        if (state.joinError case final err?)
          Text(err, style: TextStyle(color: Theme.of(context).colorScheme.error)),
      ],
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.message, required this.onRetry});

  final String message;
  final Future<void> Function() onRetry;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          "Couldn't load sessions: $message",
          style: TextStyle(color: Theme.of(context).colorScheme.error),
        ),
        TextButton(onPressed: onRetry, child: const Text('Retry')),
      ],
    );
  }
}
