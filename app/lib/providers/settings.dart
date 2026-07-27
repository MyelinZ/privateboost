import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _foregroundPollingKey = 'foreground_polling_enabled';

/// Whether the sessions screen runs the foreground wake poller while it is
/// open. Persisted across restarts and defaults to `false`: a stock install
/// advances joined rounds only through the FCM push and the WorkManager
/// periodic until the user turns this on. The cadence the poller ticks at is a
/// separate build define ([foregroundPollSeconds] in `demo_config.dart`); this
/// flag is only the on/off gate.
class ForegroundPollingNotifier extends AsyncNotifier<bool> {
  @override
  Future<bool> build() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_foregroundPollingKey) ?? false;
  }

  Future<void> set(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_foregroundPollingKey, enabled);
    state = AsyncData(enabled);
  }
}

final foregroundPollingProvider =
    AsyncNotifierProvider<ForegroundPollingNotifier, bool>(ForegroundPollingNotifier.new);
