import 'package:firebase_ui_auth/firebase_ui_auth.dart';
import 'package:flutter/material.dart';

/// Firebase email/password sign-in. `firebase_ui_auth`'s default form
/// supports in-app registration, so a tester can create an account on
/// first run without any service-account/console setup.
class LoginScreen extends StatelessWidget {
  const LoginScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SignInScreen(providers: [EmailAuthProvider()]);
  }
}
