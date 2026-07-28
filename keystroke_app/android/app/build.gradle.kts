import java.util.Properties

plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

val keystoreProperties = Properties()
val keystorePropertiesFile = rootProject.file("key.properties")
if (keystorePropertiesFile.exists()) {
    keystorePropertiesFile.inputStream().use(keystoreProperties::load)
}

fun signingSetting(propertyName: String, environmentName: String): String? =
    keystoreProperties.getProperty(propertyName)?.takeIf { it.isNotBlank() }
        ?: System.getenv(environmentName)?.takeIf { it.isNotBlank() }

val releaseStorePath = signingSetting("storeFile", "KEYSTROKE_KEYSTORE_PATH")
val releaseStorePassword = signingSetting("storePassword", "KEYSTROKE_KEYSTORE_PASSWORD")
val releaseKeyAlias = signingSetting("keyAlias", "KEYSTROKE_KEY_ALIAS")
val releaseKeyPassword = signingSetting("keyPassword", "KEYSTROKE_KEY_PASSWORD")
val hasReleaseSigning =
    listOf(
        releaseStorePath,
        releaseStorePassword,
        releaseKeyAlias,
        releaseKeyPassword,
    ).all { !it.isNullOrBlank() }

android {
    namespace = "com.blu.keystrokeanalyzer"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.blu.keystrokeanalyzer"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    signingConfigs {
        create("release") {
            if (hasReleaseSigning) {
                storeFile = file(releaseStorePath!!)
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
            }
        }
    }

    buildTypes {
        release {
            signingConfig =
                if (hasReleaseSigning) signingConfigs.getByName("release") else null
        }
    }
}

gradle.taskGraph.whenReady {
    val releaseBuildRequested =
        allTasks.any { task ->
            task.name == "assembleRelease" || task.name == "bundleRelease"
        }
    if (releaseBuildRequested && !hasReleaseSigning) {
        throw GradleException(
            "Release signing is not configured. Copy android/key.properties.example " +
                "to android/key.properties or set the KEYSTROKE_KEYSTORE_* environment variables.",
        )
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
