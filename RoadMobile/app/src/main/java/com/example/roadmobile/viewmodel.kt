package com.example.roadmobile

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

data class LocationState(
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val accuracy: Float = 0f,
    val speed: Float = 0f,
    val altitude: Double = 0.0,
    val isLoading: Boolean = true,
    val error: String? = null
)

class LocationViewModel(private val locationManager: LocationManager) : ViewModel() {
    private val _locationState = MutableStateFlow(LocationState())
    val locationState: StateFlow<LocationState> = _locationState

    fun startLocationUpdates() {
        viewModelScope.launch {
            locationManager.getLocationUpdates()
                .catch { e ->
                    _locationState.value = _locationState.value.copy(
                        error = e.message,
                        isLoading = false
                    )
                }
                .collect { location ->
                    _locationState.value = LocationState(
                        latitude = location.latitude,
                        longitude = location.longitude,
                        accuracy = location.accuracy,
                        speed = location.speed,
                        altitude = location.altitude,
                        isLoading = false,
                        error = null
                    )
                }
        }
    }
}