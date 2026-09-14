package com.example.vaanishield

import android.content.Context
import android.util.Log

// ─────────────────────────────────────────────────────────────
// OnnxRuntimeHelper – STUB
//
// Future implementation will:
//   1. Load model from assets:  "vaanishield.onnx"
//   2. Create OrtSession
//   3. Accept Float32Array (16kHz, mono, 1-second chunk)
//   4. Run inference → return Map with probability, state, etc.
//
// ONNX Input Contract (to be finalised by ML team):
//   Name:   "input"
//   Shape:  [1, 16000]  (batch=1, samples=16000 at 16kHz)
//   Type:   FLOAT32
//
// ONNX Output Contract:
//   Name:   "output"
//   Shape:  [1, 1]
//   Type:   FLOAT32  (spoof probability 0.0–1.0)
//
// TODO: Add dependency in build.gradle:
//   implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'
// ─────────────────────────────────────────────────────────────
object OnnxRuntimeHelper {
    private const val TAG = "OnnxRuntimeHelper"

    fun initialize(context: Context, modelFileName: String) {
        Log.w(TAG, "[STUB] initialize – ONNX Runtime not yet integrated.")
        throw UnsupportedOperationException("OnnxRuntimeHelper not yet implemented.")
    }

    fun runInference(audioChunk: FloatArray): Map<String, Any> {
        Log.w(TAG, "[STUB] runInference – ONNX Runtime not yet integrated.")
        throw UnsupportedOperationException("OnnxRuntimeHelper not yet implemented.")
    }

    fun dispose() {
        Log.w(TAG, "[STUB] dispose.")
    }
}
