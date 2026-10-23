plugins {
    // AGP 9 compiles Kotlin itself (built-in Kotlin); no kotlin.android plugin.
    id("com.android.library")
}

// ─────────────────────────────────────────────────────────────────────────────
// F1: PdfBox-Android AAR asset extraction for host JVM unit tests.
//
// PdfBox-Android bundles its resources (glyphlist, AFM, CMaps) as Android
// assets (assets/com/tom_roush/…). On the host JVM these are not on the
// classpath, so GlyphList.<clinit> and PDFTextStripper.<clinit> fail. The fix:
// resolve only the AAR (transitive=false), unzip its assets/ into a build
// directory, and register that directory as a test resource root. No files are
// committed to source control (Gates: D7 / N2).
// ─────────────────────────────────────────────────────────────────────────────
val pdfboxAar by configurations.creating {
    isCanBeResolved = true
    isCanBeConsumed = false
}

val extractPdfboxTestResources by tasks.registering(Sync::class) {
    from(
        pdfboxAar.elements.map { elements ->
            elements.map { f ->
                zipTree(f.asFile).matching { include("assets/**") }
            }
        }
    )
    into(layout.buildDirectory.dir("generated/pdfbox-test-resources"))
    eachFile {
        // Strip the leading "assets/" segment so paths land at the root,
        // matching the classpath layout GlyphList.load() expects.
        relativePath = RelativePath(
            true,
            *relativePath.segments.drop(1).toTypedArray()
        )
    }
    includeEmptyDirs = false
}

android {
    namespace  = "com.narad.search.lib"
    // AGP 8.13.2 supports compileSdk up to 36.1 (AGP 8.13.0 release notes).
    compileSdk = 36

    defaultConfig {
        minSdk = 29  // G4: Android 10+ floor (build-decisions.md §1/G4)

        // Native ABI targets: arm64 phones plus the x86_64 emulator. No arm32:
        // the bundled libc++_shared.so exists only for these two ABIs.
        ndk {
            abiFilters += setOf("arm64-v8a", "x86_64")
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        // PdfBox-Android calls android.util.Log at class-init; return default
        // values (0 / null / false) instead of throwing "Method not mocked".
        // Combined with extractPdfboxTestResources (above) this keeps all PDF
        // host-JVM tests real — they build and extract PDFs with PdfBox itself.
        unitTests.isReturnDefaultValues = true
    }

    sourceSets {
        // Wire the unzipped PdfBox AAR assets as a JVM-test resource root.
        getByName("test") {
            // AGP 9 rejects Provider source dirs; the afterEvaluate block below wires the task.
            resources.srcDir(layout.buildDirectory.dir("generated/pdfbox-test-resources").get().asFile)
        }
    }
}

// AGP does not auto-wire the task dependency when using srcDir(TaskProvider) for
// processXxxUnitTestJavaRes. Add it explicitly so the resource copy is always up-to-date.
afterEvaluate {
    tasks.matching { it.name.endsWith("UnitTestJavaRes") }
        .configureEach { dependsOn(extractPdfboxTestResources) }
}

// Kotlin 2.x: compilerOptions replaces the deprecated kotlinOptions DSL.
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

// One version for the runtime dep and the host-test resource unzip, so they cannot drift.
val pdfboxVersion = "2.0.27.0"

dependencies {
    // -------------------------------------------------------------------------
    // ONNX Runtime for Android. 1.30.0 is the minimum for EmbeddingGemma 2
    // (opset 21 + GatherBlockQuantized); 1.17.3 fails to load it. Sessions use the
    // default CPU EP only (no NNAPI), so the old NNAPI-API pin no longer applies.
    // -------------------------------------------------------------------------
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")

    // -------------------------------------------------------------------------
    // Apache POI is test-only: DocumentExtractorTest uses it as the DOCX parity
    // oracle. Runtime DOCX extraction is ZipInputStream + SAX (U8).
    // Exclude stax and woodstox-core: they clash with the JDK XML parser.
    // -------------------------------------------------------------------------
    testImplementation("org.apache.poi:poi-ooxml:5.2.5") {
        exclude(group = "stax",                       module = "stax-api")
        exclude(group = "stax",                       module = "stax")
        exclude(group = "com.fasterxml.woodstox",     module = "woodstox-core")
    }

    // -------------------------------------------------------------------------
    // PDF extraction — PdfBox-Android (Apache-2.0, replaces AGPL iText7).
    // Declared `implementation` (not `api`); NaradApplication calls the lib-side
    // DocumentExtractor.init(context) helper which calls PDFBoxResourceLoader.init
    // so the app module never sees com.tom_roush.* on its compile classpath.
    // ⚠ Exclude org.bouncycastle — BC conflicts with Android crypto on API 28+.
    //   Certificate-encrypted PDFs throw NoClassDefFoundError (caught in Throwable
    //   catch inside extractPdf); user-password PDFs (RC4 or AES) need no BC.
    // -------------------------------------------------------------------------
    implementation("com.tom-roush:pdfbox-android:$pdfboxVersion") {
        exclude(group = "org.bouncycastle")
    }

    // Same AAR for host-JVM test resource extraction (transitive=false — just the file).
    pdfboxAar("com.tom-roush:pdfbox-android:$pdfboxVersion") {
        isTransitive = false
    }

    // WorkManager — used by IndexWorker
    implementation("androidx.work:work-runtime-ktx:2.9.0")

    // AndroidX Core — provides ServiceCompat (API-34 startForeground fix) and
    // NotificationManagerCompat (progress notification updates from IndexWorker).
    implementation("androidx.core:core-ktx:1.13.1")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // -------------------------------------------------------------------------
    // DJL HuggingFace tokenizers — production (main sourceset, ships in APK).
    //
    // ai.djl.huggingface:tokenizers provides the Java API (HuggingFaceTokenizer).
    // ai.djl.android:tokenizer-native provides the Android AAR that bundles
    // libdjl_tokenizer.so for arm64-v8a + x86_64 (and armeabi-v7a + x86 which
    // are filtered out by abiFilters above). DJL detects Android at runtime and
    // calls System.loadLibrary("djl_tokenizer") — no network access required.
    //
    // EmbeddingGemma 2 uses the same 0.33.0 Rust tokenizer engine on Android and
    // the host JVM so token IDs stay in parity.
    //
    // Bundled native lib provenance:
    //   lib/src/main/jniLibs/{arm64-v8a,x86_64}/libc++_shared.so is the genuine
    //   NDK 26.3.11579264 sysroot libc++_shared.so (SHA256-verified identical to the
    //   NDK), required because libdjl_tokenizer.so dynamically links it and Android
    //   does not ship it in the system image. AGP strips native libs at packaging so
    //   the committed unstripped copy is intentional.
    // -------------------------------------------------------------------------
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
    runtimeOnly("ai.djl.android:tokenizer-native:0.33.0")

    // -------------------------------------------------------------------------
    // Host-platform (JVM) test deps — NOT shipped in the APK.
    // These resolve the macOS-arm64 native lib problem described in §2d of
    // build-decisions.md: onnxruntime-android ships Android ELF binaries that
    // cannot load on the macOS JVM, so the JVM unit-test sourceset needs the
    // desktop variant instead.
    // -------------------------------------------------------------------------
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.30.0")
    testImplementation("junit:junit:4.13.2")
    // Real org.json on the JVM: android.jar only ships stubs ("Method ... not mocked").
    testImplementation("org.json:json:20240303")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.0")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")

    // -------------------------------------------------------------------------
    // DJL HuggingFace tokenizers — host-only (macOS arm64 native, test scope).
    // Used in HuggingFaceTokenizer.kt (test sourceset) to tokenise text for the
    // EmbeddingModel test gate.  NOT added as `implementation` — would risk
    // pulling native binaries into the Android APK.
    // -------------------------------------------------------------------------
    // 0.28.0 fails to parse the TemplateProcessing post-processor in this
    // tokenizer.json (it uses a newer format).  0.33.0 (latest) supports it.
    testImplementation("ai.djl.huggingface:tokenizers:0.33.0")

    // -------------------------------------------------------------------------
    // Instrumented test dependencies
    // -------------------------------------------------------------------------
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:core:1.5.0")
    androidTestImplementation("androidx.work:work-testing:2.9.0")

    // MockWebServer for ModelProvisionerTest (serves synthetic payloads;
    // no real ONNX bytes needed — just a few KB per test).
    androidTestImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
}
