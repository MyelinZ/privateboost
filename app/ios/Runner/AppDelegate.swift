import Flutter
import UIKit
import workmanager_apple

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    // The BGTaskScheduler handler must be registered before launch completes,
    // so it cannot move into didInitializeImplicitFlutterEngine. The identifier
    // and 15-minute floor mirror the Dart registerPeriodicParticipation(); iOS
    // treats the frequency as a minimum and runs opportunistically, so this is
    // a self-wake fallback — the FCM silent push stays the primary background
    // training trigger.
    WorkmanagerPlugin.registerPeriodicTask(
      withIdentifier: "pbr-contribute",
      frequency: NSNumber(value: 15 * 60))
    // The BGTask callback runs in its own FlutterEngine, which gets no plugins
    // unless registered through this callback; without it the wake's
    // Firebase.initializeApp() dies on a missing platform channel and every
    // WorkManager wake steps 0 sessions.
    WorkmanagerPlugin.setPluginRegistrantCallback { registry in
      GeneratedPluginRegistrant.register(with: registry)
    }
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
  }
}
