import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:privateboost_app/messaging.dart';
import 'package:workmanager/workmanager.dart';

const _periodicTask = 'pbr-contribute';

/// WorkManager's entry point, in its own background isolate separate from the
/// FCM handler's, driving one wake through `runBackgroundWake`: the periodic
/// fallback for when no push arrives.
@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((task, _) async {
    try {
      final result = await runBackgroundWake(trigger: 'workmanager');
      debugPrint('privateboost workmanager wake: sessions_stepped=${result.stepped.length} '
          'error=${result.lastError}');
    } catch (e) {
      // runBackgroundWake records its own outcome and never throws; this is
      // defense in depth so a surprise throw cannot fail the task and trigger
      // OS backoff retries the telemetry never accounts for.
      debugPrint('privateboost workmanager wake crashed: $e');
    }
    return true;
  });
}

/// Initializes WorkManager and registers [callbackDispatcher]. Call once at
/// app startup, alongside `initMessaging`.
Future<void> initWorkManager() => Workmanager().initialize(callbackDispatcher);

Future<void> registerPeriodicParticipation() => Workmanager().registerPeriodicTask(
      _periodicTask,
      _periodicTask,
      frequency: const Duration(minutes: 15),
      constraints: Constraints(networkType: NetworkType.connected),
    );

Future<void> cancelPeriodicParticipation() => Workmanager().cancelByUniqueName(_periodicTask);

/// Registers/cancels the periodic background task as auth state changes.
/// Held behind [backgroundSchedulerProvider] so a fake can stand in where the
/// WorkManager platform channel is unavailable.
class BackgroundScheduler {
  const BackgroundScheduler();
  Future<void> register() => registerPeriodicParticipation();
  Future<void> cancel() => cancelPeriodicParticipation();
}

final backgroundSchedulerProvider = Provider<BackgroundScheduler>((ref) => const BackgroundScheduler());
