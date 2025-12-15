package com.example.roadmobile

import android.Manifest
import android.content.Context
import android.media.MediaPlayer
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue

import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import org.osmdroid.config.Configuration

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun LocationScreen() {

    var mediaPlayer: MediaPlayer? by remember { mutableStateOf(null) }
    var isPlaying by remember { mutableStateOf(false) }

    var userStatus by remember { mutableStateOf("") }
    val context = LocalContext.current
    val locationManager = remember { LocationManager(context) }
    val viewModel: LocationViewModel = viewModel(
        factory = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return LocationViewModel(locationManager) as T
            }
        }
    )

    val locationState by viewModel.locationState.collectAsState()

    // Initialize osmdroid configuration
    LaunchedEffect(Unit) {
        Configuration.getInstance().load(
            context,
            context.getSharedPreferences("osmdroid", Context.MODE_PRIVATE)
        )
    }

    // Request location permissions
    val permissionState = rememberMultiplePermissionsState(
        permissions = listOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
    )

    LaunchedEffect(permissionState.allPermissionsGranted) {
        if (permissionState.allPermissionsGranted) {
            viewModel.startLocationUpdates()
        }
    }

    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        when {
            !permissionState.allPermissionsGranted -> {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text("Location permission is required")
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = { permissionState.launchMultiplePermissionRequest() }) {
                        Text("Grant Permission")
                    }
                }
            }
            locationState.isLoading -> {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator()
                    Text("Getting location...", modifier = Modifier.padding(top = 16.dp))
                }
            }
            locationState.error != null -> {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text("Error: ${locationState.error}", color = MaterialTheme.colorScheme.error)
                }
            }
            else -> {
                // Map View
                Box(modifier = Modifier.weight(1f)) {
                    OpenStreetMap(
                        modifier = Modifier.fillMaxSize(),
                        latitude = locationState.latitude,
                        longitude = locationState.longitude
                    )
                }




                // Location Info Card
               /* Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            "GPS Coordinates",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        LocationRow("Latitude", String.format("%.6f", locationState.latitude))
                        LocationRow("Longitude", String.format("%.6f", locationState.longitude))
                        LocationRow("Accuracy", "${String.format("%.1f", locationState.accuracy)}m")
                        LocationRow("Speed", "${String.format("%.1f", locationState.speed)} m/s")
                        LocationRow("Altitude", "${String.format("%.1f", locationState.altitude)}m")

                    }
                }*/
            }
        }
    }
}

@Composable
fun LocationRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}