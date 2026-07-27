import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/demo_config.dart';
import 'package:privateboost_app/pilot_ca.dart';

void main() {
  group('endpointNeedsCa', () {
    test('https endpoints pin a CA, http endpoints do not', () {
      expect(endpointNeedsCa('https://10.0.0.1:42800'), isTrue);
      expect(endpointNeedsCa('http://127.0.0.1:42800'), isFalse);
    });

    test('the default (emulator) endpoint is cleartext, so no CA', () {
      // A stray CA requirement on the loopback demo would break the emulator
      // flow, so the default endpoint must resolve to no CA.
      expect(endpointNeedsCa(aggEndpoint), isFalse);
    });
  });

  group('looksLikeCaPem', () {
    test('accepts a real PEM certificate block', () {
      final pem = utf8.encode(
        '-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n',
      );
      expect(looksLikeCaPem(Uint8List.fromList(pem)), isTrue);
    });

    test('rejects a placeholder without a certificate block', () {
      final placeholder = utf8.encode(
        '# replace with deploy/hetzner/secrets/ca.crt before a TLS build\n',
      );
      // The guard exists so a TLS release built with the placeholder still in
      // place fails loudly rather than trusting the system store.
      expect(looksLikeCaPem(Uint8List.fromList(placeholder)), isFalse);
    });

    test('the committed asset is the real deployment CA', () async {
      // The pilot ships the real Hetzner CA (deploy/hetzner/secrets/ca.crt is
      // copied here), so the committed asset is a valid certificate the release
      // build pins. Read the committed file directly (test cwd is the package
      // root).
      final bytes = await File(caAssetPath).readAsBytes();
      expect(looksLikeCaPem(bytes), isTrue,
          reason: 'the committed $caAssetPath must be the real deployment CA');
    });
  });

  // resolvePinnedCa on an https endpoint composes endpointNeedsCa (true) with
  // looksLikeCaPem over the loaded asset; both predicates are covered above.
  // Only the cleartext short-circuit, which never touches the asset bundle, is
  // exercised here.
  group('resolvePinnedCa', () {
    test('returns null for a cleartext endpoint without touching the asset', () async {
      expect(await resolvePinnedCa('http://127.0.0.1:42800'), isNull);
    });
  });
}
