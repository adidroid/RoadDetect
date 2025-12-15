package com.example.roadmobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

class SegmentationModel(context: Context, modelName: String = "weatherAndroid_embedded.onnx") {
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

    // Model input dimensions
    private val inputWidth = 512
    private val inputHeight = 512

    init {
        val modelBytes = context.assets.open(modelName).readBytes()
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.addNnapi()   // enable NNAPI hardware acceleration
        ortSession = ortEnvironment.createSession(modelBytes, sessionOptions)

    }

    fun segment(bitmap: Bitmap): IntArray {

        fun cropCenterSquare(bmp: Bitmap): Bitmap {
            val size = minOf(bmp.width, bmp.height)
            val xOffset = (bmp.width - size) / 2
            val yOffset = (bmp.height - size) / 2
            return Bitmap.createBitmap(bmp, xOffset, yOffset, size, size)
        }

        // Resize bitmap to model input size
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)


        // Convert to float array (normalized)
        val floatBuffer = preprocessImage(resized)

        // Create ONNX tensor
        val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)

        // Run inference
        val outputs = ortSession.run(mapOf("input" to inputTensor))

        // Output shape is [1, 1, 512, 512] - class indices
        val rawOutput = outputs[0].value

        return when (rawOutput) {
            is Array<*> -> {
                val batch = rawOutput as Array<Array<Array<LongArray>>>
                // Extract [1, 512, 512] -> flatten to IntArray
                flattenToIntArray(batch[0][0])
            }
            else -> {
                android.util.Log.e("SegmentationModel", "Unexpected output type: ${rawOutput?.javaClass}")
                IntArray(inputWidth * inputHeight)
            }
        }
    }

    private fun flattenToIntArray(output: Array<LongArray>): IntArray {
        val segmentationMap = IntArray(inputWidth * inputHeight)
        for (h in output.indices) {
            for (w in output[h].indices) {
                segmentationMap[h * inputWidth + w] = output[h][w].toInt()
            }
        }
        return segmentationMap
    }

    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        val buffer = FloatBuffer.allocate(3 * inputWidth * inputHeight)
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // Match Python: simple 0-1 normalization (divide by 255)
        // Order: RGB (same as Python after cv2.cvtColor(BGR2RGB))
        for (c in 0..2) {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val value = when (c) {
                    0 -> ((pixel shr 16) and 0xFF) / 255f  // R
                    1 -> ((pixel shr 8) and 0xFF) / 255f   // G
                    else -> (pixel and 0xFF) / 255f         // B
                }
                buffer.put(value)
            }
        }
        buffer.rewind()
        return buffer
    }

    fun createColoredBitmap(segmentationMap: IntArray): Bitmap {
        val bitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)

        // Get unique classes in the map
        val uniqueClasses = segmentationMap.toSet()
        android.util.Log.d("SegmentationModel", "Found classes: $uniqueClasses")

        // Match Python color mapping exactly
        for (i in segmentationMap.indices) {
            val classId = segmentationMap[i]
            val color = when (classId) {
                0 -> 0xFF000000.toInt() // class 0 - background (black)
                1 -> 0xFFFF0000.toInt() // class 1 - road (red)
                2 -> 0xFF00FF00.toInt() // class 2 - green
                3 -> 0xFF0000FF.toInt() // class 3 - blue
                4 -> 0xFFFFFF00.toInt() // class 4 - yellow
                else -> 0xFFFFFFFF.toInt() // unknown - white
            }
            bitmap.setPixel(i % inputWidth, i / inputWidth, color)
        }
        return bitmap
    }

    fun close() {
        ortSession.close()
    }
}