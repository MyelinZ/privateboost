#!/usr/bin/env bash
# Build the Android APK for the PrivateBoost demo app.
#
# Mirrors pboost's approach: cargokit is bypassed on Android; we cross-compile
# the Rust cdylib with cargo-ndk straight into jniLibs, then let Flutter/AGP
# package the pre-placed .so. Unlike pboost we have NO C++/DuckDB dependency,
# so there is NO libc++_shared.so to bundle.
#
# ABIs: x86_64 FIRST (the Pixel_7 emulator is x86_64); arm64-v8a for real
# phones. Set PBR_ARM64_ONLY=1 to skip x86_64.
set -euo pipefail
cd "$(dirname "$0")"

NDK_HOME="${ANDROID_NDK_HOME:-/opt/android-sdk/ndk/28.2.13676358}"
export ANDROID_NDK_HOME="$NDK_HOME"
JNILIBS="$PWD/android/app/src/main/jniLibs"
SO_NAME="librust_lib_privateboost_app.so"

ABIS=(x86_64 arm64-v8a)
if [[ "${PBR_ARM64_ONLY:-0}" == "1" ]]; then ABIS=(arm64-v8a); fi

echo ">> cross-compiling Rust cdylib for: ${ABIS[*]}"
for abi in "${ABIS[@]}"; do
  (cd rust && cargo ndk -t "$abi" --platform 24 -o "$JNILIBS" build --release --lib)
done

echo ">> jniLibs contents:"
for abi in "${ABIS[@]}"; do
  ls -la "$JNILIBS/$abi/$SO_NAME" || { echo "MISSING $SO_NAME for $abi"; exit 1; }
done

echo ">> flutter build apk (debug)"
flutter build apk --debug

echo ">> APK: build/app/outputs/flutter-apk/app-debug.apk"
