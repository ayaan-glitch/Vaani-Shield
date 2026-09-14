package com.example.vaanishield

import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        // Register the VAANI-SHIELD detection MethodChannel
        DetectionChannel(flutterEngine.dartExecutor.binaryMessenger)
    }
}
