import 'dart:io';

import 'package:battery_plus/battery_plus.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:package_info_plus/package_info_plus.dart';

/// Device and app context for a `paperSimRoundMetrics` document. Every read is
/// guarded and returns a sentinel on failure, never throwing: a metric path
/// that threw would break participation, and these plugins can fail (an
/// unsupported host, or a background isolate whose plugin registrant is not
/// yet available). The device/app fields never change for an install, so they
/// are resolved once per isolate via [resolveMetricsContext] and cached.
class MetricsContext {
  const MetricsContext({
    required this.deviceId,
    required this.deviceModel,
    required this.appVersion,
  });

  final String deviceId;
  final String deviceModel;
  final String appVersion;
}

/// Resolves the (cached) device/app context. Called once before each
/// participation; every field read is guarded, so this never throws.
Future<MetricsContext> resolveMetricsContext() async => MetricsContext(
      deviceId: await deviceId(),
      deviceModel: await deviceModel(),
      appVersion: await appVersion(),
    );

Future<String>? _deviceIdMemo;
Future<String>? _deviceModelMemo;
Future<String>? _appVersionMemo;

/// Stable per-vendor device identifier (`ANDROID_ID` on Android,
/// `identifierForVendor` on iOS). `'unknown'` when neither can be read.
Future<String> deviceId() => _deviceIdMemo ??= _readDeviceId();

/// Human-readable device model, e.g. `"Google Pixel 9 Pro"`. `'unknown'` on
/// failure or an unsupported host.
Future<String> deviceModel() => _deviceModelMemo ??= _readDeviceModel();

/// App version and build, e.g. `"1.0.0+1"`. `'unknown'` on failure.
Future<String> appVersion() => _appVersionMemo ??= _readAppVersion();

Future<String> _readDeviceId() async {
  final plugin = DeviceInfoPlugin();
  try {
    return (await plugin.androidInfo).id;
  } catch (_) {
    try {
      return (await plugin.iosInfo).identifierForVendor ?? 'unknown';
    } catch (_) {
      return 'unknown';
    }
  }
}

Future<String> _readDeviceModel() async {
  final plugin = DeviceInfoPlugin();
  try {
    final a = await plugin.androidInfo;
    return '${a.manufacturer} ${a.model}';
  } catch (_) {
    try {
      final i = await plugin.iosInfo;
      return '${i.name} ${i.model}';
    } catch (_) {
      return 'unknown';
    }
  }
}

Future<String> _readAppVersion() async {
  try {
    final info = await PackageInfo.fromPlatform();
    return '${info.version}+${info.buildNumber}';
  } catch (_) {
    return 'unknown';
  }
}

/// Active connection tag for the round about to run (`wifi`, `cellular`,
/// `none`, or `other`). Read per round, not cached, since it can change
/// between rounds. `'other'` on failure.
Future<String> networkType() async {
  try {
    final results = await Connectivity().checkConnectivity();
    if (results.contains(ConnectivityResult.mobile)) return 'cellular';
    if (results.contains(ConnectivityResult.wifi)) return 'wifi';
    if (results.contains(ConnectivityResult.none)) return 'none';
    return 'other';
  } catch (_) {
    return 'other';
  }
}

/// Battery state tag for the round about to run, the raw [BatteryState] name:
/// `charging`, `full`, or `connectedNotCharging` mean the device is on
/// external power; `discharging` means it is not. Read per round, not cached,
/// since it can change between rounds. `'unknown'` on failure.
Future<String> batteryState() async {
  try {
    return (await Battery().batteryState).name;
  } catch (_) {
    return 'unknown';
  }
}

/// Battery charge percent (0-100) for the round about to run, read per round
/// like [batteryState]. `null` when it cannot be read, so callers can omit
/// the field rather than record a fake level.
Future<int?> batteryLevel() async {
  try {
    return await Battery().batteryLevel;
  } catch (_) {
    return null;
  }
}

/// Whole-process resident set size in bytes at the moment of the call, a
/// coarse cross-platform memory sample (mach on iOS, `/proc` on Android). It
/// is the app's total RSS, not isolated crypto memory. `0` when the platform
/// cannot report it.
int currentRssBytes() {
  try {
    return ProcessInfo.currentRss;
  } catch (_) {
    return 0;
  }
}

