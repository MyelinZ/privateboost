import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Thrown when an operation requires a signed-in user but none is available.
///
/// An `Exception` (not an `Error`): being signed out is an expected runtime
/// state, not a programming bug, so callers may catch and handle it.
class SignedOutException implements Exception {
  const SignedOutException();
  @override
  String toString() => 'SignedOutException: no signed-in user';
}

/// A signed-in user's Firebase ID token plus their uid.
class Creds {
  const Creds({required this.idToken, required this.uid});
  final String idToken;
  final String uid;
}

/// Emits on sign-in/out *and* whenever the Firebase SDK refreshes the ID token,
/// so [credsProvider] re-derives a fresh token instead of holding an expired
/// one (tokens last ~1h; auth-state changes alone would not refresh it).
final authStateProvider = StreamProvider<User?>(
    (ref) => FirebaseAuth.instance.idTokenChanges());

/// The current credentials, or null when signed out.
final credsProvider = FutureProvider<Creds?>((ref) async {
  final user = ref.watch(authStateProvider).value;
  if (user == null) return null;
  final token = await user.getIdToken();
  if (token == null) return null;
  return Creds(idToken: token, uid: user.uid);
});
