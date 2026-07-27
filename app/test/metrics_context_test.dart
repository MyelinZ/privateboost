import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/metrics/context.dart';

void main() {
  // On the test host none of the platform plugins are registered, so every
  // guarded read hits its catch and returns the sentinel. This is exactly the
  // failure mode the guards exist for (a plugin missing at runtime).
  TestWidgetsFlutterBinding.ensureInitialized();

  test('deviceModel degrades to a sentinel, never throwing', () async {
    expect(await deviceModel(), 'unknown');
  });

  test('appVersion degrades to a sentinel, never throwing', () async {
    expect(await appVersion(), 'unknown');
  });

  test('deviceId degrades to a sentinel, never throwing', () async {
    expect(await deviceId(), 'unknown');
  });

  test('networkType degrades to a sentinel, never throwing', () async {
    expect(await networkType(), 'other');
  });

  test('batteryState degrades to a sentinel, never throwing', () async {
    expect(await batteryState(), 'unknown');
  });

  test('batteryLevel degrades to null, never throwing', () async {
    expect(await batteryLevel(), isNull);
  });

  test('currentRssBytes never throws and is non-negative', () {
    expect(currentRssBytes(), greaterThanOrEqualTo(0));
  });

}
