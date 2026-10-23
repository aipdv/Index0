# Consumer ProGuard rules — consumed by :app when it depends on :lib.
# Keep ONNX Runtime classes referenced from :lib
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# DJL HuggingFace tokenizer — JNI bridge classes must survive R8 (native symbol names).
-keep class ai.djl.** { *; }
-keepclassmembers class ai.djl.** { *; }

# R8 missing-class warnings, from the first release build with minify on.
# DJL references desktop JDK classes (java.awt, javax.imageio, javax.sound, javax.tools,
# java.lang.management) from code paths Narad never calls. PdfBox-Android references
# optional JP2 / BouncyCastle (excluded on purpose in lib/build.gradle.kts).
# commons-compress references lang3. None of these exist on Android.
-dontwarn java.awt.**
-dontwarn javax.imageio.**
-dontwarn javax.sound.**
-dontwarn javax.tools.**
-dontwarn java.lang.management.**
-dontwarn com.gemalto.jp2.**
-dontwarn org.bouncycastle.**
-dontwarn org.apache.commons.lang3.SystemProperties
