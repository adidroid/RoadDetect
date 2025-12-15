package com.example.roadmobile

import android.Manifest
import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.google.accompanist.permissions.*
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun SegmentationScreen() {

    val context1 = androidx.compose.ui.platform.LocalContext.current
    val locationManager = remember { LocationManager(context1) }
    val viewModel: LocationViewModel = viewModel(
        factory = object : androidx.lifecycle.ViewModelProvider.Factory {
            override fun <T : androidx.lifecycle.ViewModel> create(modelClass: Class<T>): T {
                return LocationViewModel(locationManager) as T
            }
        }
    )
    val locationState by viewModel.locationState.collectAsState()
    val locationPermissionState = rememberPermissionState(
        permission = Manifest.permission.ACCESS_FINE_LOCATION
    )
    val context = LocalContext.current
    val cameraPermissionState = rememberPermissionState(
        android.Manifest.permission.CAMERA
    )

    var segmentationBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var maskLeft by remember { mutableStateOf(0) }
    var maskRight by remember { mutableStateOf(0) }


    val segmentationModel = remember { SegmentationModel(context) }

    DisposableEffect(Unit) {
        onDispose {
            segmentationModel.close()
        }
    }

    fun rotateMask(mask: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return mask
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotation.toFloat())
        return Bitmap.createBitmap(mask, 0, 0, mask.width, mask.height, matrix, true)
    }

    Column(modifier = Modifier.fillMaxSize()) {
        when {
            cameraPermissionState.status.isGranted&&locationPermissionState.status.isGranted -> {
                Box(modifier = Modifier.fillMaxSize()) {

                    CameraPreview(
                        onFrameAnalyzed = { bitmap, rotation ->
                            val originalWidth = bitmap.width
                            val originalHeight = bitmap.height

                            val segMap = segmentationModel.segment(bitmap)
                            val smallMask = segmentationModel.createColoredBitmap(segMap)

                            var fullMask = resizeMask(smallMask, originalWidth, originalHeight)
                            fullMask = rotateMask(fullMask, rotation)

                            segmentationBitmap = fullMask

                            // compute vertical mask area
                            val (left, right) = findMaskBoundsHorizontal(fullMask)
                            maskLeft = left
                            maskRight = right

                        },
                        modifier = Modifier.fillMaxSize()
                    )

                    // Segmentation mask overlay
                    segmentationBitmap?.let { bitmap ->
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Segmentation overlay",
                            modifier = Modifier.fillMaxSize(),
                            alpha = 0.5f
                        )
                    }

                  // UI aligned with mask
                    Row(modifier = Modifier.fillMaxSize()) {

                        // top UI area
                        Box(
                            modifier = Modifier
                                .fillMaxHeight()
                                .weight(1f),
                            contentAlignment = Alignment.Center
                        ) {
                            LocationScreen()
                        }

                        Spacer(
                            modifier = Modifier
                                .width(((maskRight - maskLeft)).dp)
                        )

                        // bottom UI area
                        Box(
                            modifier = Modifier
                                .fillMaxHeight()
                                .weight(1f),
                            contentAlignment = Alignment.Center
                        ) {

                            Card(
                                modifier = Modifier
                                    .fillMaxSize()

                            ) {
                                Column(modifier = Modifier.padding(16.dp).fillMaxSize(),
                                    verticalArrangement = Arrangement.Center) {

                                    Spacer(modifier = Modifier.height(8.dp))
                                    LocationRow("Speed", "${String.format("%.1f", locationState.speed)} m/s")
                                    LocationRow("Altitude", "${String.format("%.1f", locationState.altitude)}m")

                                }
                            }


                        }
                    }
                }
            }

            else -> {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text("Camera permission is required")
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(onClick = { cameraPermissionState.launchPermissionRequest()
                                       locationPermissionState.launchPermissionRequest()}) {
                        Text("Grant Permissions to use app")
                    }
                }
            }
        }
    }
}

fun resizeMask(mask: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
    return Bitmap.createScaledBitmap(mask, targetWidth, targetHeight, true)
}

// -------------------- Detect mask top & bottom --------------------
fun findMaskBoundsHorizontal(mask: Bitmap): Pair<Int, Int> {
    val width = mask.width
    val height = mask.height

    var left = width
    var right = 0

    for (x in 0 until width) {
        for (y in 0 until height) {

            val pixel = mask.getPixel(x, y)

            if (pixel != 0x00000000) {
                if (x < left) left = x
                if (x > right) right = x
                break
            }
        }
    }

    return Pair(left, right)
}



