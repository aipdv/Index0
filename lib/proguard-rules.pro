# ProGuard rules for the :lib module.
# Keep ONNX Runtime JNI classes
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }

# DJL HuggingFace tokenizer — JNI bridge classes must survive R8 (native symbol names).
-keep class ai.djl.** { *; }
-keepclassmembers class ai.djl.** { *; }
