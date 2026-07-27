import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/joined_sessions.dart';

void main() {
  late Directory tmp;
  late JoinedSessions store;

  setUp(() async {
    tmp = await Directory.systemTemp.createTemp('pbr_joined_sessions_test');
    store = JoinedSessions(directory: tmp);
  });

  tearDown(() async {
    if (await tmp.exists()) await tmp.delete(recursive: true);
  });

  test('a joined id is returned for the same uid and not for another', () async {
    await store.join('uid-a', 'session-1');

    expect(await store.ids('uid-a'), {'session-1'});
    expect(await store.ids('uid-b'), isEmpty);
  });

  test('joining the same session twice is idempotent', () async {
    await store.join('uid-a', 'session-1');
    await store.join('uid-a', 'session-1');

    expect(await store.ids('uid-a'), {'session-1'});
  });

  test('a session marked finished drops out of active but stays in ids', () async {
    await store.join('uid-a', 'session-1');
    await store.join('uid-a', 'session-2');

    await store.markFinished('uid-a', 'session-1');

    expect(await store.ids('uid-a'), {'session-1', 'session-2'});
    expect(await store.active('uid-a'), {'session-2'});
  });

  test('clear empties one uid without touching another', () async {
    await store.join('uid-a', 'session-1');
    await store.join('uid-b', 'session-2');

    await store.clear('uid-a');

    expect(await store.ids('uid-a'), isEmpty);
    expect(await store.active('uid-a'), isEmpty);
    expect(await store.ids('uid-b'), {'session-2'});
  });

  test('a missing store file yields an empty set rather than throwing', () async {
    expect(await store.ids('uid-a'), isEmpty);
    expect(await store.active('uid-a'), isEmpty);
  });

  test('a corrupt store file yields an empty set rather than throwing', () async {
    final file = File('${tmp.path}/joined_sessions.json');
    await file.writeAsString('{not valid json');

    expect(await store.ids('uid-a'), isEmpty);
    expect(await store.active('uid-a'), isEmpty);

    // A write after a corrupt read must still succeed rather than propagate
    // the earlier corruption.
    await store.join('uid-a', 'session-1');
    expect(await store.ids('uid-a'), {'session-1'});
  });

  test('markFinished on a session never joined does not throw or create it', () async {
    await store.markFinished('uid-a', 'session-1');

    expect(await store.ids('uid-a'), isEmpty);
  });

  test("a second instance over the same directory sees the first instance's writes", () async {
    // The closest available proxy for "readable from another isolate": Dart
    // isolates share no memory, so two instances with no reference to each
    // other, both reading and writing the same on-disk file, is what a
    // foreground-UI instance and a background-handler instance actually look
    // like to each other.
    final other = JoinedSessions(directory: tmp);

    await store.join('uid-a', 'session-1');
    expect(await other.ids('uid-a'), {'session-1'});

    await other.markFinished('uid-a', 'session-1');
    expect(await store.active('uid-a'), isEmpty);
  });

  test('concurrent markFinished and join on the same store both take effect', () async {
    // A background wake marking one session finished races a foreground tap
    // joining another. Two instances over the same directory, as a background
    // handler and the foreground UI are: neither shares memory, so only a
    // cross-isolate guard on the store can stop one write clobbering the
    // other.
    final ui = JoinedSessions(directory: tmp);
    final background = JoinedSessions(directory: tmp);

    await store.join('uid-a', 'session-1');

    await Future.wait([
      background.markFinished('uid-a', 'session-1'),
      ui.join('uid-a', 'session-2'),
    ]);

    expect(
      await store.ids('uid-a'),
      {'session-1', 'session-2'},
      reason: 'the concurrent join must not be lost',
    );
    expect(
      await store.active('uid-a'),
      {'session-2'},
      reason: 'the concurrent markFinished must not be lost',
    );
  });
}
