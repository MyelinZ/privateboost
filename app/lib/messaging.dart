import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:privateboost_app/demo_config.dart';
import 'package:privateboost_app/facade.dart';
import 'package:privateboost_app/metrics/context.dart';
import 'package:privateboost_app/metrics/metrics_writer.dart';
import 'package:privateboost_app/participation_lock.dart';
import 'package:privateboost_app/src/rust/frb_generated.dart';
import 'package:privateboost_app/wake_loop.dart';

const _facade = Facade();

/// One closure per wake, turning a stepped round into a [recordRoundStep] call
/// over the resolved [context] and the [timestamps] store `wallMs` is measured
/// against. Named rather than inlined at the call site so a test exercises the
/// exact hook production wires in: a hand-rolled stand-in could transpose two
/// parameters and neither the analyzer nor the suite would notice.
///
/// [clock] and [firestore] let a test pin `clientTsMs` and capture writes with
/// no Firestore backend; production supplies neither.
WakeStepHook metricsRecordingHook({
  required MetricsContext context,
  required String uid,
  required int batchId,
  required int batchCount,
  required SessionRoundTimestamps timestamps,
  FirebaseFirestore? firestore,
  @visibleForTesting int Function() clock = _systemClientTsMs,
}) =>
    (sessionId, triggerSource, result, datasetId, wakeLatencyMs) => recordRoundStep(
          context: context,
          uid: uid,
          triggerSource: triggerSource,
          batchId: batchId,
          batchCount: batchCount,
          datasetId: datasetId,
          result: result,
          timestamps: timestamps,
          wakeLatencyMs: wakeLatencyMs,
          firestore: firestore,
          clock: clock,
        );

int _systemClientTsMs() => DateTime.now().millisecondsSinceEpoch;

/// Drives one wake, redoing the setup a background isolate does not inherit
/// (Rust init, Firebase init, a fresh id token) before handing off to
/// [runWake]. Shared by every trigger that drives a round: the FCM background
/// handler, the foreground `round_open` listener, the WorkManager fallback and
/// the sessions screen's poller. The Contribute button records standing consent
/// only and never calls this.
///
/// [trigger] becomes each round document's `triggerSource`. [wakeLatencyMs] is
/// the device-perceived push delay, non-null only for a push wake whose payload
/// carried `sentAt`, and applies to the first session stepped.
///
/// Every stepped session writes its own `paperSimRoundMetrics` document via
/// [metricsRecordingHook]. `wallMs` is measured against a fresh
/// [SessionRoundTimestamps]; the on-disk store behind it, not this object, is
/// what makes the gap correct across wakes and restarts.
///
/// firebase_messaging dispatches every background message to one long-lived
/// isolate and flutter_rust_bridge throws on a second `init()` in it, so
/// without the guard only the first wake per process would run.
///
/// Never throws: signed-out and no-token cases and any setup failure fold into
/// [WakeResult.lastError], so nothing can crash the background isolate or
/// silently fail the WorkManager task. [ParticipationLock] serializes it
/// against sibling triggers, returning early if a wake is already in flight.
@pragma('vm:entry-point')
Future<WakeResult> runBackgroundWake({
  String trigger = 'push',
  int? wakeLatencyMs,
}) async {
  ParticipationLock? lock;
  try {
    lock = await ParticipationLock.tryAcquire();
    if (lock == null) {
      return const WakeResult(stepped: [], lastError: 'another participation in flight');
    }
    if (!RustLib.instance.initialized) await RustLib.init();
    await Firebase.initializeApp();
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      return const WakeResult(stepped: [], lastError: 'not signed in');
    }
    final idToken = await user.getIdToken(true); // force-refresh: a background
    // wake may fire long after the last foreground token refresh.
    if (idToken == null) {
      return const WakeResult(stepped: [], lastError: 'no id token');
    }
    if (wakeLatencyMs != null) {
      debugPrint('privateboost fcm wake latency: ${wakeLatencyMs}ms');
    }
    final metricsContext = await resolveMetricsContext();
    return await runWake(
      trigger: trigger,
      uid: user.uid,
      idToken: idToken,
      aggEndpoint: aggEndpoint,
      wakeLatencyMs: wakeLatencyMs,
      onRoundStep: metricsRecordingHook(
        context: metricsContext,
        uid: user.uid,
        batchId: batchId,
        batchCount: batchCount,
        timestamps: SessionRoundTimestamps(),
      ),
    );
  } catch (e) {
    return WakeResult(stepped: const [], lastError: 'background wake failed: $e');
  } finally {
    await lock?.release();
  }
}

/// Only `{kind: "round_open"}` drives participation; any other `kind`, or
/// none, is ignored so unrelated pushes never enroll the device.
@visibleForTesting
bool isRoundOpenWake(Map<String, dynamic> data) => data['kind'] == 'round_open';

/// [nowMs] at isolate wake minus the server's `sentAt` stamp, in milliseconds.
/// `sentAt` is epoch millis as a string, since FCM data values always are.
/// Null when absent or unparseable, as on a WorkManager wake. The two clocks
/// are unsynchronised, so this includes skew: a device-perceived delay, not a
/// network measurement.
@visibleForTesting
int? wakeLatencyMsFromData(Map<String, dynamic> data, int nowMs) {
  final raw = data['sentAt'];
  if (raw == null) return null;
  final sentAt = int.tryParse(raw.toString());
  if (sentAt == null) return null;
  return nowMs - sentAt;
}

/// The background and terminated silent-push handler: a `round_open` message
/// runs one round in the background isolate. Wake latency is sampled here, the
/// instant the isolate wakes, so it excludes the id-token refresh
/// `runBackgroundWake` does.
@pragma('vm:entry-point')
Future<void> _fcmBackgroundHandler(RemoteMessage message) async {
  if (!isRoundOpenWake(message.data)) return;
  final wakeLatencyMs =
      wakeLatencyMsFromData(message.data, DateTime.now().millisecondsSinceEpoch);
  final result = await runBackgroundWake(trigger: 'push', wakeLatencyMs: wakeLatencyMs);
  debugPrint('privateboost fcm wake: sessions_stepped=${result.stepped.length} '
      'error=${result.lastError}');
}

/// Registers the background handler, re-registers the token on rotation, and
/// drives a round from foreground `round_open` pushes under the same `push`
/// label. With the foreground poller also on, the two race benignly: whichever
/// fires second finds [ParticipationLock] held. Call once at startup, after
/// `Firebase.initializeApp()`.
void initMessaging() {
  FirebaseMessaging.onBackgroundMessage(_fcmBackgroundHandler);
  FirebaseMessaging.instance.onTokenRefresh.listen((_) async {
    // An async listener's exception is otherwise unhandled; a failed
    // re-registration just means the old token stays active until the next
    // rotation, so log-and-continue.
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) return;
      final idToken = await user.getIdToken();
      if (idToken == null) return;
      await const FcmRegistrar().register(idToken);
    } catch (e) {
      debugPrint('privateboost: fcm token refresh registration failed: $e');
    }
  });
  FirebaseMessaging.onMessage.listen((m) async {
    if (!isRoundOpenWake(m.data)) return;
    final wakeLatencyMs =
        wakeLatencyMsFromData(m.data, DateTime.now().millisecondsSinceEpoch);
    // runBackgroundWake never throws, so this async listener cannot leak an
    // unhandled zone error.
    final result =
        await runBackgroundWake(trigger: 'push', wakeLatencyMs: wakeLatencyMs);
    debugPrint('privateboost fcm foreground wake: '
        'sessions_stepped=${result.stepped.length} error=${result.lastError}');
  });
}

/// Needed on Android 13+ to receive pushes while backgrounded, where it shows
/// the runtime dialog. Call it after the first frame: awaited during startup it
/// would block that frame on the dialog.
Future<void> requestNotificationPermission() =>
    FirebaseMessaging.instance.requestPermission();

/// Recorded with the device registration. Nothing branches on it today, since
/// the sender ships both the android and apns blocks on every push, but a
/// hardcoded android would break iOS wakes the moment pushes are keyed per
/// platform.
ClientPlatform currentDevicePlatform() => defaultTargetPlatform == TargetPlatform.iOS
    ? ClientPlatform.ios
    : ClientPlatform.android;

/// Fetches the current FCM token and registers it. Held behind
/// [fcmRegistrarProvider] so a fake can stand in where the platform channel is
/// unavailable. Idempotent server-side.
class FcmRegistrar {
  const FcmRegistrar();
  Future<void> register(String idToken) async {
    final token = await FirebaseMessaging.instance.getToken();
    if (token == null) return;
    final result = await _facade.registerDevice(
      aggEndpoint: aggEndpoint,
      idToken: idToken,
      fcmToken: token,
      platform: currentDevicePlatform(),
    );
    if (!result.ok) {
      debugPrint('privateboost: RegisterDevice failed: ${result.lastError}');
    }
  }
}

final fcmRegistrarProvider = Provider<FcmRegistrar>((ref) => const FcmRegistrar());
