import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

/// The bundled trust root for a TLS (`https`) aggregator. A debug/emulator
/// build ships the committed placeholder and never reads it (its endpoint is
/// cleartext); a pilot build has `deploy/hetzner/secrets/ca.crt` copied over
/// the placeholder before `flutter build` (see `app/README.md`).
const String caAssetPath = 'assets/pilot-ca.pem';

/// Whether [aggEndpoint] needs a pinned CA. Only `https` endpoints do; `http`
/// (the loopback emulator default) connects in cleartext with no CA.
bool endpointNeedsCa(String aggEndpoint) => Uri.parse(aggEndpoint).scheme == 'https';

/// Whether [bytes] are an actual PEM certificate rather than the committed
/// placeholder. The placeholder is comment-only text with no PEM block, so the
/// begin marker distinguishes a real CA that was copied in from a release built
/// without one.
bool looksLikeCaPem(Uint8List bytes) =>
    utf8.decode(bytes, allowMalformed: true).contains('-----BEGIN CERTIFICATE-----');

/// The CA bytes to pin for [aggEndpoint], or `null` when the endpoint is
/// cleartext (`http`) and no CA is used.
///
/// On an `https` endpoint the bundled asset must be a real certificate; a
/// still-placeholder asset means a TLS release was built without copying the
/// deployment CA in, which would otherwise fall back to the system trust store
/// and defeat the pin, so this throws instead. Reads through `rootBundle`,
/// which the FCM and WorkManager background isolates initialise (via
/// `WidgetsFlutterBinding.ensureInitialized`) before either participation path
/// runs, so the same call works foreground and background alike.
Future<Uint8List?> resolvePinnedCa(String aggEndpoint) async {
  if (!endpointNeedsCa(aggEndpoint)) return null;
  final data = await rootBundle.load(caAssetPath);
  final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  if (!looksLikeCaPem(bytes)) {
    throw StateError(
      'TLS endpoint $aggEndpoint needs a pinned CA, but $caAssetPath is still '
      'the committed placeholder. Copy deploy/hetzner/secrets/ca.crt over it '
      'before building a release.',
    );
  }
  return bytes;
}
