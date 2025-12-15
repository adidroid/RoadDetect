package com.example.roadmobile

import android.content.Context
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.mylocation.GpsMyLocationProvider
import org.osmdroid.views.overlay.mylocation.MyLocationNewOverlay
import kotlin.apply
import kotlin.collections.none
import kotlin.collections.removeAll
import kotlin.text.format

@Composable
fun rememberMapViewWithLifecycle(context: Context): MapView {
    val mapView = remember {
        MapView(context).apply {
            Configuration.getInstance().userAgentValue = context.packageName
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            controller.setZoom(15.0)
        }
    }

    DisposableEffect(mapView) {
        mapView.onResume()
        onDispose {
            mapView.onPause()
        }
    }

    return mapView
}

@Composable
fun OpenStreetMap(
    modifier: Modifier = Modifier,
    latitude: Double,
    longitude: Double,
    onMapReady: (MapView) -> Unit = {}
) {
    val context = LocalContext.current
    val mapView = rememberMapViewWithLifecycle(context)

    LaunchedEffect(latitude, longitude) {
        if (latitude != 0.0 && longitude != 0.0) {
            val geoPoint = GeoPoint(latitude, longitude)
            mapView.controller.animateTo(geoPoint)
        }
    }

    AndroidView(
        factory = {
            mapView.apply {
                onMapReady(this)
            }
        },
        modifier = modifier,
        update = { map ->
            if (latitude != 0.0 && longitude != 0.0) {
                val geoPoint = GeoPoint(latitude, longitude)

                // Clear existing markers
                map.overlays.removeAll { it is Marker }

                // Add marker at current location
                val marker = Marker(map).apply {
                    position = geoPoint
                    setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                    title = "Current Location"
                    snippet = "Lat: ${"%.6f".format(latitude)}, Lng: ${"%.6f".format(longitude)}"
                }
                map.overlays.add(marker)

                // Add my location overlay (blue dot)
                if (map.overlays.none { it is MyLocationNewOverlay }) {
                    val locationOverlay = MyLocationNewOverlay(
                        GpsMyLocationProvider(context),
                        map
                    ).apply {
                        enableMyLocation()
                        enableFollowLocation()
                    }
                    map.overlays.add(locationOverlay)
                }

                map.invalidate()
            }
        }
    )
}