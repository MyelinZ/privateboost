import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_ui_auth/firebase_ui_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:privateboost_app/background.dart';
import 'package:privateboost_app/messaging.dart';
import 'package:privateboost_app/providers/auth.dart';
import 'package:privateboost_app/screens/login_screen.dart';
import 'package:privateboost_app/screens/sessions_screen.dart';
import 'package:privateboost_app/src/rust/frb_generated.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  await Firebase.initializeApp();
  await initWorkManager();
  initMessaging();
  FirebaseUIAuth.configureProviders([EmailAuthProvider()]);
  runApp(const ProviderScope(child: PrivateboostApp()));
  // After the first frame: on Android 13+ this shows the notification
  // permission dialog, so it must not run during startup where an await
  // would block the first frame from rendering.
  WidgetsBinding.instance.addPostFrameCallback((_) {
    requestNotificationPermission().catchError((Object e) {
      debugPrint('privateboost: notification permission request failed: $e');
    });
  });
}

class PrivateboostApp extends ConsumerWidget {
  const PrivateboostApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Register this device's FCM token and the WorkManager periodic fallback
    // once signed in, cancel the latter on sign-out; this is a side effect
    // (an RPC call / platform channel call), so it belongs in ref.listen
    // rather than the build's render path. Fires on the initial
    // loading->data resolution and every later sign-in/out.
    ref.listen(authStateProvider, (_, next) {
      if (!next.hasValue) return; // ignore loading/error states
      final scheduler = ref.read(backgroundSchedulerProvider);
      final user = next.value;
      if (user == null) {
        scheduler.cancel().catchError((Object e) {
          debugPrint('privateboost: cancel background task failed: $e');
        });
        return;
      }
      scheduler.register().catchError((Object e) {
        debugPrint('privateboost: register background task failed: $e');
      });
      final registrar = ref.read(fcmRegistrarProvider);
      // Fire-and-forget with an error handler: an unhandled failure here is
      // a zone error, and a silently missing FCM registration just means no
      // push wakes until the next sign-in/token rotation (WorkManager still
      // runs as the fallback).
      user.getIdToken().then((t) {
        if (t != null) return registrar.register(t);
      }).catchError((Object e) {
        debugPrint('privateboost: fcm registration failed: $e');
      });
    });

    final auth = ref.watch(authStateProvider);
    return MaterialApp(
      title: 'PrivateBoost',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: auth.when(
        loading: () => const _Loading(),
        error: (e, _) => _Message('auth error: $e'),
        data: (user) => user == null ? const LoginScreen() : const SessionsScreen(),
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) =>
      const Scaffold(body: Center(child: CircularProgressIndicator()));
}

class _Message extends StatelessWidget {
  const _Message(this.text);
  final String text;
  @override
  Widget build(BuildContext context) =>
      Scaffold(body: Center(child: Text(text)));
}
