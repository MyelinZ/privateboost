import 'package:flutter_test/flutter_test.dart';
import 'package:privateboost_app/messaging.dart';

void main() {
  group('isRoundOpenWake', () {
    test('a round_open data message drives participation', () {
      expect(isRoundOpenWake({'kind': 'round_open'}), isTrue);
      expect(
        isRoundOpenWake({'kind': 'round_open', 'roundId': '7'}),
        isTrue,
        reason: 'extra data keys do not change the decision',
      );
    });

    test('any other kind, or none, is ignored', () {
      expect(isRoundOpenWake({'kind': 'something_else'}), isFalse);
      expect(isRoundOpenWake({}), isFalse,
          reason: 'a message with no kind must not enroll the device');
      expect(isRoundOpenWake({'kind': ''}), isFalse);
    });
  });

  group('wakeLatencyMsFromData', () {
    test('subtracts the server sentAt stamp from the device clock', () {
      // FCM data-message values are always strings, so sentAt arrives as one.
      expect(wakeLatencyMsFromData({'sentAt': '1000'}, 1250), 250);
    });

    test('returns null when sentAt is absent', () {
      expect(wakeLatencyMsFromData({'kind': 'round_open'}, 1250), isNull);
    });

    test('returns null when sentAt is not a parseable integer', () {
      expect(wakeLatencyMsFromData({'sentAt': 'not-a-number'}, 1250), isNull);
      expect(wakeLatencyMsFromData({'sentAt': ''}, 1250), isNull);
    });
  });
}
