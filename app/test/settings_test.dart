import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/providers/settings.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('foreground polling defaults to off when never set', () async {
    SharedPreferences.setMockInitialValues({});
    final container = ProviderContainer();
    addTearDown(container.dispose);

    expect(await container.read(foregroundPollingProvider.future), isFalse);
  });

  test('an existing persisted true value is read on first build', () async {
    SharedPreferences.setMockInitialValues({'foreground_polling_enabled': true});
    final container = ProviderContainer();
    addTearDown(container.dispose);

    expect(await container.read(foregroundPollingProvider.future), isTrue);
  });

  test('set(true) updates state and persists to a fresh container', () async {
    SharedPreferences.setMockInitialValues({});
    final container = ProviderContainer();
    addTearDown(container.dispose);

    await container.read(foregroundPollingProvider.future);
    await container.read(foregroundPollingProvider.notifier).set(true);
    expect(container.read(foregroundPollingProvider).value, isTrue);

    final reopened = ProviderContainer();
    addTearDown(reopened.dispose);
    expect(await reopened.read(foregroundPollingProvider.future), isTrue);
  });
}
