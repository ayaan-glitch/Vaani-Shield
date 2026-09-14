package com.example.vaanishield

import android.util.Log
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

// ─────────────────────────────────────────────────────────────
// DetectionChannel – Flutter ↔ Kotlin MethodChannel bridge
//
// Channel name: "vaanishield/detection"
//
// Methods:
//   initialize  → unit    (load ONNX model, prepare session)
//   analyzeChunk → Map    (run ONNX inference on 1s audio chunk)
//   dispose     → unit    (release ONNX resources)
//
// Expected output from analyzeChunk:
//   {
//     "probability":    0.87,
//     "state":          "SUSPICIOUS",
//     "modelVersion":   "v3",
//     "inferenceTimeMs": 18
//   }
//
// TODO: Replace stub with real OnnxRuntimeHelper calls once
//       the VAANI-SHIELD ONNX model is available.
// ─────────────────────────────────────────────────────────────
class DetectionChannel(messenger: BinaryMessenger) : MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "vaanishield/detection"
        private const val TAG = "VaaniShield"
    }

    private val channel = MethodChannel(messenger, CHANNEL)

    init {
        channel.setMethodCallHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        Log.d(TAG, "MethodCall: ${call.method}")
        when (call.method) {
            "initialize" -> {
                // TODO: OnnxRuntimeHelper.initialize(context, "vaanishield.onnx")
                Log.w(TAG, "[STUB] initialize called – ONNX model not yet integrated.")
                result.success(null)
            }
            "analyzeChunk" -> {
                // TODO: Pass Float32Array to OnnxRuntimeHelper.runInference(chunk)
                // and return the real result map.
                Log.w(TAG, "[STUB] analyzeChunk called – returning stub result.")
                result.error(
                    "NOT_IMPLEMENTED",
                    "ONNX inference not yet implemented. Awaiting model delivery.",
                    null
                )
            }
            "dispose" -> {
                // TODO: OnnxRuntimeHelper.dispose()
                Log.w(TAG, "[STUB] dispose called.")
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }
}
