import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/participation_lock.dart';

void main() {
  late Directory tmp;
  late File lockFile;

  setUp(() async {
    tmp = await Directory.systemTemp.createTemp('pbr_lock_test');
    lockFile = File('${tmp.path}/participation.lock');
  });

  tearDown(() async {
    if (await tmp.exists()) await tmp.delete(recursive: true);
  });

  test('a second acquire is rejected while the first is held', () async {
    final first = await ParticipationLock.tryAcquireAt(lockFile);
    expect(first, isNotNull);

    final second = await ParticipationLock.tryAcquireAt(lockFile);
    expect(second, isNull, reason: 'an overlapping participation must be excluded');

    await first!.release();

    final third = await ParticipationLock.tryAcquireAt(lockFile);
    expect(third, isNotNull, reason: 'the lock is available again after release');
    await third!.release();
  });

  test('a stale lock is reclaimed, a fresh one is not', () async {
    // A crashed isolate leaves the lock file behind; a caller must reclaim a
    // lock past the stale window rather than deadlock all future
    // participations, but must not steal a lock that a live session holds.
    final held = await ParticipationLock.tryAcquireAt(lockFile);
    expect(held, isNotNull);

    final blocked = await ParticipationLock.tryAcquireAt(
      lockFile,
      staleAfter: const Duration(hours: 1),
    );
    expect(blocked, isNull, reason: 'a fresh lock is not stale');

    final reclaimed = await ParticipationLock.tryAcquireAt(
      lockFile,
      staleAfter: Duration.zero,
    );
    expect(reclaimed, isNotNull, reason: 'a lock past the stale window is reclaimed');
    await reclaimed!.release();
  });

  test('two contenders racing the same stale lock: at most one reclaims it', () async {
    // A crashed isolate's leftover, seen as stale by two triggers that wake at
    // once. Their delete+create can interleave; without a per-lock token B's
    // delete would remove A's just-created lock and both would proceed. The
    // reclaim must grant the lock to at most one of them.
    final crashed = await ParticipationLock.tryAcquireAt(lockFile);
    expect(crashed, isNotNull);

    final results = await Future.wait([
      ParticipationLock.tryAcquireAt(lockFile, staleAfter: Duration.zero),
      ParticipationLock.tryAcquireAt(lockFile, staleAfter: Duration.zero),
    ]);

    final winners = results.whereType<ParticipationLock>().toList();
    expect(
      winners.length,
      1,
      reason: 'a stale-lock race must not grant two overlapping participations',
    );

    // The reclaimed lock is real and still exclusive.
    final newcomer = await ParticipationLock.tryAcquireAt(lockFile);
    expect(newcomer, isNull, reason: 'the reclaimed lock still excludes newcomers');

    await winners.single.release();
    final free = await ParticipationLock.tryAcquireAt(lockFile);
    expect(free, isNotNull, reason: 'the lock frees up once the winner releases');
    await free!.release();
  });

  test('release does not delete a lock another isolate reclaimed as stale', () async {
    // A slow (not crashed) participation still holds its lock when a sibling
    // reclaims it past the stale window. When the slow one finally releases, it
    // must not delete the reclaimer's live lock.
    final slow = await ParticipationLock.tryAcquireAt(lockFile);
    expect(slow, isNotNull);

    final reclaimer = await ParticipationLock.tryAcquireAt(
      lockFile,
      staleAfter: Duration.zero,
    );
    expect(reclaimer, isNotNull, reason: 'the stale lock is reclaimed');

    await slow!.release();

    final newcomer = await ParticipationLock.tryAcquireAt(lockFile);
    expect(
      newcomer,
      isNull,
      reason: "the slow releaser must not delete the reclaimer's live lock",
    );

    await reclaimer!.release();
    final free = await ParticipationLock.tryAcquireAt(lockFile);
    expect(free, isNotNull, reason: 'the lock frees up once the reclaimer releases');
    await free!.release();
  });
}
