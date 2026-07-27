import 'dart:typed_data';

import 'package:privateboost_app/pilot_ca.dart';
import 'package:privateboost_app/src/rust/api/mobile.dart' as rust;

export 'package:privateboost_app/src/rust/api/mobile.dart'
    show ClientPlatform, RegisterResult;

/// Thin wrapper over the generated FRB bindings
/// (`lib/src/rust/api/mobile.dart`). Device registration is the only
/// remaining path through here: session participation goes through the wake
/// loop (`wake_loop.dart`), whose `RustWakeBridge` pins the CA itself instead
/// of routing through this class.
class Facade {
  const Facade();

  /// Registers this device's FCM token with the aggregator so it can be
  /// woken by a silent push when a round opens, pinning the bundled CA when
  /// the endpoint is `https`. Never throws: any failure, including a missing/
  /// placeholder CA on a TLS endpoint, is folded into
  /// `RegisterResult.lastError`.
  Future<rust.RegisterResult> registerDevice({
    required String aggEndpoint,
    required String idToken,
    required String fcmToken,
    required rust.ClientPlatform platform,
  }) async {
    final Uint8List? caPem;
    try {
      caPem = await resolvePinnedCa(aggEndpoint);
    } catch (e) {
      return rust.RegisterResult(ok: false, lastError: '$e');
    }
    return rust.registerDevice(
      aggEndpoint: aggEndpoint,
      idToken: idToken,
      fcmToken: fcmToken,
      platform: platform,
      caPem: caPem,
    );
  }
}
